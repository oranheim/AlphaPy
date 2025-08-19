################################################################################
#
# Package   : AlphaPy
# Module    : variables
# Created   : July 11, 2013
#
# Copyright 2020 ScottFree Analytics LLC
# Mark Conway & Robert D. Scott II
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


#
# Variables
# ---------
#
# Numeric substitution is allowed for any number in the expression.
# Offsets are allowed in event expressions but cannot be substituted.
#
# Examples
# --------
#
# Variable('rrunder', 'rr_3_20 <= 0.9')
#
# 'rrunder_2_10_0.7'
# 'rrunder_2_10_0.9'
# 'xmaup_20_50_20_200'
# 'xmaup_10_50_20_50'
#


#
# Imports
#

import ast
import builtins
import logging
import operator
import re
import sys
from collections import OrderedDict
from importlib import import_module

import numpy as np

from alphapy.alias import get_alias
from alphapy.frame import Frame, frame_name
from alphapy.globals import LOFF, ROFF, USEP
from alphapy.utilities import valid_name

#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Security: Safe Expression Evaluator
#

# Whitelist of allowed operators and functions for variable expressions
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.And: operator.and_,
    ast.Or: operator.or_,
    ast.Not: operator.not_,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}

# Whitelist of allowed functions for variable expressions
SAFE_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    # NumPy functions commonly used in financial analysis
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "power": np.power,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "mean": np.mean,
    "std": np.std,
    "var": np.var,
    "median": np.median,
    "percentile": np.percentile,
    "isnan": np.isnan,
    "isinf": np.isinf,
    "isfinite": np.isfinite,
}

# Whitelist of allowed module names for secure imports
ALLOWED_MODULES = {
    "alphapy.transforms",
    "numpy",
    "pandas",
    "scipy.stats",
    "sklearn.preprocessing",
    "talib",  # Technical Analysis Library often used in finance
}


class SafeExpressionEvaluator(ast.NodeVisitor):
    """Secure expression evaluator that uses AST parsing to safely evaluate
    mathematical expressions without allowing arbitrary code execution.

    This evaluator only allows whitelisted operators, functions, and column references.
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.result = None

    def evaluate(self, expression_str):
        """Safely evaluate a mathematical expression string.

        Parameters
        ----------
        expression_str : str
            The expression to evaluate

        Returns
        -------
        result : pandas.Series or scalar
            The result of the expression evaluation

        Raises
        ------
        SecurityError
            If the expression contains unsafe operations
        """
        try:
            # Parse the expression into an AST
            tree = ast.parse(expression_str, mode="eval")
            # Evaluate the AST safely
            return self.visit(tree.body)
        except Exception as e:
            logger.error(f"Failed to safely evaluate expression '{expression_str}': {e}")
            raise SecurityError(f"Unsafe or invalid expression: {expression_str}") from e

    def visit_Name(self, node):
        """Handle variable names (column references)."""
        if node.id in self.dataframe.columns:
            return self.dataframe[node.id]
        elif node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        else:
            raise SecurityError(f"Undefined variable or unsafe function: {node.id}")

    def visit_Constant(self, node):
        """Handle constant values (numbers, strings)."""
        return node.value

    def visit_Num(self, node):
        """Handle numeric constants (for older Python versions)."""
        return node.n

    def visit_Str(self, node):
        """Handle string constants (for older Python versions)."""
        return node.s

    def visit_BinOp(self, node):
        """Handle binary operations (+, -, *, /, etc.)."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)

        if op_type not in SAFE_OPERATORS:
            raise SecurityError(f"Unsafe binary operation: {op_type.__name__}")

        return SAFE_OPERATORS[op_type](left, right)

    def visit_UnaryOp(self, node):
        """Handle unary operations (-, +, not)."""
        operand = self.visit(node.operand)
        op_type = type(node.op)

        if op_type not in SAFE_OPERATORS:
            raise SecurityError(f"Unsafe unary operation: {op_type.__name__}")

        return SAFE_OPERATORS[op_type](operand)

    def visit_Compare(self, node):
        """Handle comparison operations (==, !=, <, >, <=, >=)."""
        left = self.visit(node.left)

        for op, comparator in zip(node.ops, node.comparators, strict=False):
            right = self.visit(comparator)
            op_type = type(op)

            if op_type not in SAFE_OPERATORS:
                raise SecurityError(f"Unsafe comparison operation: {op_type.__name__}")

            left = SAFE_OPERATORS[op_type](left, right)

        return left

    def visit_Call(self, node):
        """Handle function calls."""
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        else:
            raise SecurityError("Complex function calls not allowed")

        if func_name not in SAFE_FUNCTIONS:
            raise SecurityError(f"Unsafe function call: {func_name}")

        func = SAFE_FUNCTIONS[func_name]
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}

        return func(*args, **kwargs)

    def visit_BoolOp(self, node):
        """Handle boolean operations (and, or)."""
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise SecurityError(f"Unsafe boolean operation: {op_type.__name__}")

        values = [self.visit(value) for value in node.values]

        if op_type == ast.And:
            result = values[0]
            for value in values[1:]:
                result = result & value
            return result
        elif op_type == ast.Or:
            result = values[0]
            for value in values[1:]:
                result = result | value
            return result

    def generic_visit(self, node):
        """Reject any AST node types not explicitly handled."""
        raise SecurityError(f"Unsafe AST node type: {type(node).__name__}")


class SecurityError(Exception):
    """Exception raised for security violations in expression evaluation."""

    pass


def safe_module_import(module_name):
    """Safely import a module from the whitelist.

    Parameters
    ----------
    module_name : str
        Name of the module to import

    Returns
    -------
    module : module
        The imported module

    Raises
    ------
    SecurityError
        If the module is not in the whitelist
    """
    if module_name not in ALLOWED_MODULES:
        raise SecurityError(f"Module '{module_name}' not in security whitelist")

    try:
        return import_module(module_name)
    except ImportError as e:
        logger.error(f"Failed to import whitelisted module '{module_name}': {e}")
        raise


#
# Class Variable
#


class Variable:
    """Create a new variable as a key-value pair. All variables are stored
    in ``Variable.variables``. Duplicate keys or values are not allowed,
    unless the ``replace`` parameter is ``True``.

    Parameters
    ----------
    name : str
        Variable key.
    expr : str
        Variable value.
    replace : bool, optional
        Replace the current key-value pair if it already exists.

    Attributes
    ----------
    variables : dict
        Class variable for storing all known variables

    Examples
    --------

    >>> Variable("rrunder", "rr_3_20 <= 0.9")
    >>> Variable("hc", "higher_close")

    """

    # class variable to track all variables

    variables: dict[str, "Variable"] = {}

    # function __new__

    def __new__(cls, name, expr, replace=False):
        # code
        efound = expr in [Variable.variables[key].expr for key in Variable.variables]
        if efound:
            key = [key for key in Variable.variables if expr in Variable.variables[key].expr]
            logger.info("Expression '%s' already exists for key %s", expr, key)
            return
        else:
            if replace or name not in Variable.variables:
                if not valid_name(name):
                    logger.info("Invalid variable key: %s", name)
                    return
                try:
                    # Use ast.parse to validate the expression
                    ast.parse(expr, mode="eval")
                except (SyntaxError, ValueError):
                    logger.info("Invalid expression: %s", expr)
                    return
                return super().__new__(cls)
            else:
                logger.info("Key %s already exists", name)

    # function __init__

    def __init__(self, name, expr, replace=False):
        # code
        self.name = name
        self.expr = expr
        # add key with expression
        Variable.variables[name] = self

    # function __str__

    def __str__(self):
        return self.expr


#
# Function vparse
#


def vparse(vname):
    r"""Parse a variable name into its respective components.

    Parameters
    ----------
    vname : str
        The name of the variable.

    Returns
    -------
    vxlag : str
        Variable name without the ``lag`` component.
    root : str
        The base variable name without the parameters.
    plist : list
        The parameter list.
    lag : int
        The offset starting with the current value [0]
        and counting back, e.g., an offset [1] means the
        previous value of the variable.

    Notes
    -----

    **AlphaPy** makes feature creation easy. The syntax
    of a variable name maps to a function call:

    xma_20_50 => xma(20, 50)

    Examples
    --------

    >>> vparse("xma_20_50[1]")
    # ('xma_20_50', 'xma', ['20', '50'], 1)

    """

    # split along lag first
    lsplit = vname.split(LOFF)
    vxlag = lsplit[0]
    # if necessary, substitute any alias
    root = vxlag.split(USEP)[0]
    alias = get_alias(root)
    if alias:
        vxlag = vxlag.replace(root, alias)
    vsplit = vxlag.split(USEP)
    root = vsplit[0]
    plist = vsplit[1:]
    # extract lag
    lag = 0
    if len(lsplit) > 1:
        # lag is present
        slag = lsplit[1].replace(ROFF, "")
        if len(slag) > 0:
            lpat = r"(^-?[0-9]+$)"
            lre = re.compile(lpat)
            if lre.match(slag):
                lag = int(slag)
    # return all components
    return vxlag, root, plist, lag


#
# Function allvars
#


def allvars(expr):
    r"""Get the list of valid names in the expression.

    Parameters
    ----------
    expr : str
        A valid expression conforming to the Variable Definition Language.

    Returns
    -------
    vlist : list
        List of valid variable names.

    """
    regex = re.compile(r"\w+")
    items = regex.findall(expr)
    vlist = []
    for item in items:
        if valid_name(item):
            vlist.append(item)
    return vlist


#
# Function vtree
#


def vtree(vname):
    r"""Get all of the antecedent variables.

    Before applying a variable to a dataframe, we have to recursively
    get all of the child variables, beginning with the starting variable's
    expression. Then, we have to extract the variables from all the
    subsequent expressions. This process continues until all antecedent
    variables are obtained.

    Parameters
    ----------
    vname : str
        A valid variable stored in ``Variable.variables``.

    Returns
    -------
    all_variables : list
        The variables that need to be applied before ``vname``.

    Other Parameters
    ----------------
    Variable.variables : dict
        Global dictionary of variables

    """
    allv = []

    def vwalk(allv, vname):
        vxlag, root, plist, lag = vparse(vname)
        if root in Variable.variables:
            root_expr = Variable.variables[root].expr
            expr = vsub(vname, root_expr)
            av = allvars(expr)
            for v in av:
                vwalk(allv, v)
        else:
            for p in plist:
                if valid_name(p):
                    vwalk(allv, p)
        allv.append(vname)
        return allv

    allv = vwalk(allv, vname)
    all_variables = list(OrderedDict.fromkeys(allv))
    return all_variables


#
# Function vsub
#


def vsub(v, expr):
    r"""Substitute the variable parameters into the expression.

    This function performs the parameter substitution when
    applying features to a dataframe. It is a mechanism for
    the user to override the default values in any given
    expression when defining a feature, instead of having
    to programmatically call a function with new values.

    Parameters
    ----------
    v : str
        Variable name.
    expr : str
        The expression for substitution.

    Returns
    -------
    newexpr
        The expression with the new, substituted values.

    """
    # numbers pattern
    npat = r"[-+]?[0-9]*\.?[0-9]+"
    nreg = re.compile(npat)
    # find all number locations in variable name
    nreg.findall(v)
    viter = nreg.finditer(v)
    vlocs = []
    for match in viter:
        vlocs.append(match.span())
    # find all number locations in expression
    # find all non-number locations as well
    elen = len(expr)
    nreg.findall(expr)
    eiter = nreg.finditer(expr)
    elocs = []
    enlocs = []
    index = 0
    for match in eiter:
        eloc = match.span()
        elocs.append(eloc)
        enlocs.append((index, eloc[0]))
        index = eloc[1]
    # build new expression
    newexpr = ""
    for i, enloc in enumerate(enlocs):
        if i < len(vlocs):
            newexpr += expr[enloc[0] : enloc[1]] + v[vlocs[i][0] : vlocs[i][1]]
        else:
            newexpr += expr[enloc[0] : enloc[1]] + expr[elocs[i][0] : elocs[i][1]]
    estart = elocs[len(elocs) - 1][1] if elocs else 0
    newexpr += expr[estart:elen]
    return newexpr


#
# Function vexec
#


def vexec(f, v, vfuncs=None):
    r"""Add a variable to the given dataframe.

    This is the core function for adding a variable to a dataframe.
    The default variable functions are already defined locally
    in ``alphapy.transforms``; however, you may want to define your
    own variable functions. If so, then the ``vfuncs`` parameter
    will contain the list of modules and functions to be imported
    and applied by the ``vexec`` function.

    To write your own variable function, your function must have
    a pandas *DataFrame* as an input parameter and must return
    a pandas *DataFrame* with the new variable(s).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe to contain the new variable.
    v : str
        Variable to add to the dataframe.
    vfuncs : dict, optional
        Dictionary of external modules and functions.

    Returns
    -------
    f : pandas.DataFrame
        Dataframe with the new variable.

    Other Parameters
    ----------------
    Variable.variables : dict
        Global dictionary of variables

    """
    vxlag, root, plist, lag = vparse(v)
    logger.debug("vexec : %s", v)
    logger.debug("vxlag : %s", vxlag)
    logger.debug("root  : %s", root)
    logger.debug("plist : %s", plist)
    logger.debug("lag   : %s", lag)
    if vxlag not in f.columns:
        if root in Variable.variables:
            logger.debug("Found variable %s: ", root)
            vroot = Variable.variables[root]
            expr = vroot.expr
            expr_new = vsub(vxlag, expr)
            estr = f"{expr_new}"
            logger.debug("Expression: %s", estr)
            # Secure expression evaluation (replacing unsafe f.eval)
            try:
                evaluator = SafeExpressionEvaluator(f)
                f[vxlag] = evaluator.evaluate(estr)
                logger.debug("Successfully evaluated expression securely: %s", estr)
            except SecurityError as e:
                logger.error("Security violation in variable expression: %s", e)
                raise
            except Exception as e:
                logger.error("Failed to evaluate variable expression '%s': %s", estr, e)
                raise ValueError(f"Invalid variable expression: {estr}") from e
        else:
            logger.debug("Did not find variable: %s", root)
            # Must be a function call
            func_name = root
            # Convert the parameter list and prepend the data frame
            newlist = []
            for p in plist:
                try:
                    newlist.append(int(p))
                except ValueError:
                    try:
                        newlist.append(float(p))
                    except ValueError:
                        newlist.append(p)
            newlist.insert(0, f)
            # Find the module and function
            module = None
            if vfuncs:
                for m in vfuncs:
                    funcs = vfuncs[m]
                    if func_name in funcs:
                        module = m
                        break
            # If the module was found, import the external transform function,
            # else search the local namespace and AlphaPy.
            if module:
                try:
                    ext_module = safe_module_import(module)
                    func = getattr(ext_module, func_name)
                    logger.debug("Successfully imported function '%s' from secure module '%s'", func_name, module)
                except SecurityError as e:
                    logger.error("Security violation in module import: %s", e)
                    raise
                except Exception as e:
                    logger.error("Failed to import function '%s' from module '%s': %s", func_name, module, e)
                    raise ImportError(f"Cannot import function {func_name} from module {module}") from e
            else:
                modname = globals()["__name__"]
                module = sys.modules[modname]
                if func_name in dir(module):
                    func = getattr(module, func_name)
                else:
                    try:
                        ap_module = safe_module_import("alphapy.transforms")
                        func = getattr(ap_module, func_name)
                        logger.debug("Successfully imported function '%s' from alphapy.transforms", func_name)
                    except (SecurityError, ImportError, AttributeError) as e:
                        logger.debug("Could not import function '%s' from alphapy.transforms: %s", func_name, e)
                        func = None
            if func:
                # Create the variable by calling the function
                f[v] = func(*newlist)
            elif func_name not in dir(builtins):
                module_error = f"*** Could not find module to execute function {func_name} ***"
                logger.error(module_error)
                sys.exit(module_error)
    # if necessary, add the lagged variable
    if lag > 0 and vxlag in f.columns:
        f[v] = f[vxlag].shift(lag)
    # output frame
    return f


#
# Function vapply
#


def vapply(group, vname, vfuncs=None):
    r"""Apply a variable to multiple dataframes.

    Parameters
    ----------
    group : alphapy.Group
        The input group.
    vname : str
        The variable to apply to the ``group``.
    vfuncs : dict, optional
        Dictionary of external modules and functions.

    Returns
    -------
    None : None

    Other Parameters
    ----------------
    Frame.frames : dict
        Global dictionary of dataframes

    See Also
    --------
    vunapply

    """
    # get all frame names to apply variables
    gnames = [item.lower() for item in group.members]
    # get all the precedent variables
    allv = vtree(vname)
    # apply the variables to each frame
    for g in gnames:
        fname = frame_name(g, group.space)
        if fname in Frame.frames:
            f = Frame.frames[fname].df
            if not f.empty:
                for v in allv:
                    logger.debug("Applying variable %s to %s", v, g)
                    f = vexec(f, v, vfuncs)
            else:
                logger.debug("Frame for %s is empty", g)
        else:
            logger.debug("Frame not found: %s", fname)


#
# Function vmapply
#


def vmapply(group, vs, vfuncs=None):
    r"""Apply multiple variables to multiple dataframes.

    Parameters
    ----------
    group : alphapy.Group
        The input group.
    vs : list
        The list of variables to apply to the ``group``.
    vfuncs : dict, optional
        Dictionary of external modules and functions.

    Returns
    -------
    None : None

    See Also
    --------
    vmunapply

    """
    for v in vs:
        logger.info("Applying variable: %s", v)
        vapply(group, v, vfuncs)


#
# Function vunapply
#


def vunapply(group, vname):
    r"""Remove a variable from multiple dataframes.

    Parameters
    ----------
    group : alphapy.Group
        The input group.
    vname : str
        The variable to remove from the ``group``.

    Returns
    -------
    None : None

    Other Parameters
    ----------------
    Frame.frames : dict
        Global dictionary of dataframes

    See Also
    --------
    vapply

    """
    # get all frame names to apply variables
    gnames = [item.lower() for item in group.all_members()]
    # apply the variables to each frame
    for g in gnames:
        fname = frame_name(g, group.space)
        if fname in Frame.frames:
            f = Frame.frames[fname].df
            logger.info("Unapplying variable %s from %s", vname, g)
            if vname not in f.columns:
                logger.info("Variable %s not in %s frame", vname, g)
            else:
                estr = f"Frame.frames['{fname}'].df = f.df.drop('{vname}', axis=1)"
                exec(estr)
        else:
            logger.info("Frame not found: %s", fname)


#
# Function vmunapply
#


def vmunapply(group, vs):
    r"""Remove a list of variables from multiple dataframes.

    Parameters
    ----------
    group : alphapy.Group
        The input group.
    vs : list
        The list of variables to remove from the ``group``.

    Returns
    -------
    None : None

    See Also
    --------
    vmapply

    """
    for v in vs:
        vunapply(group, v)

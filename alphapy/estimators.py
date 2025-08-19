################################################################################
#
# Package   : AlphaPy
# Module    : estimators
# Created   : July 11, 2013
#
# Copyright 2019 ScottFree Analytics LLC
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
# Imports
#

import ast
import logging

from alphapy.globals import SSEP, ModelType, Objective

# Try to import Keras/TensorFlow components
try:
    from keras.models import Sequential

    KERAS_AVAILABLE = True
except ImportError:
    # Keras/TensorFlow not installed
    Sequential = None
    KERAS_AVAILABLE = False

try:
    # Try newer scikeras package (recommended for scikit-learn integration)
    from scikeras.wrappers import KerasClassifier, KerasRegressor
except ImportError:
    try:
        # Fallback to tensorflow.keras for older setups
        from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
    except ImportError:
        # If both fail, provide a helpful error message
        import warnings

        if KERAS_AVAILABLE:
            warnings.warn(
                "Keras scikit-learn wrappers not found. Install 'scikeras' package for Keras support.", stacklevel=2
            )
        KerasClassifier = None
        KerasRegressor = None
import builtins
import contextlib

import yaml
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, LinearSVC

#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Security: Safe Parameter Parser
#

# Whitelist of allowed literal types for ML algorithm parameters
SAFE_PARAMETER_TYPES = (int, float, str, bool, type(None))

# Whitelist of allowed parameter values that can be parsed from strings
SAFE_PARAMETER_VALUES = {
    "true",
    "false",
    "none",
    "null",
    "auto",
    "scale",
    "balanced",
    "uniform",
    "normal",
    "l1",
    "l2",
    "elasticnet",
    "newton-cg",
    "lbfgs",
    "liblinear",
    "sag",
    "saga",
    "gini",
    "entropy",
    "best",
    "random",
    "sqrt",
    "log2",
    "relu",
    "tanh",
    "sigmoid",
    "linear",
    "softmax",
    "rmsprop",
    "adam",
    "sgd",
    "adagrad",
    "adadelta",
}


class ParameterSecurityError(Exception):
    """Exception raised for security violations in parameter parsing."""

    pass


def safe_parameter_parser(param_value):
    """Safely parse algorithm parameter values from configuration.

    This function replaces the unsafe eval() call for parsing ML algorithm
    parameters. It only allows safe literal values and whitelisted strings.

    Parameters
    ----------
    param_value : str
        The parameter value to parse

    Returns
    -------
    parsed_value : int, float, str, bool, or None
        The safely parsed parameter value

    Raises
    ------
    ParameterSecurityError
        If the parameter contains unsafe values
    """
    if not isinstance(param_value, str):
        # If it's already a safe type, return as-is
        if isinstance(param_value, SAFE_PARAMETER_TYPES):
            return param_value
        else:
            raise ParameterSecurityError(f"Unsafe parameter type: {type(param_value)}")

    # Strip whitespace and convert to lowercase for comparison
    param_lower = param_value.strip().lower()

    # Handle boolean values
    if param_lower == "true":
        return True
    elif param_lower == "false":
        return False

    # Handle None/null values
    if param_lower in ("none", "null"):
        return None

    # Try to parse as number (int or float)
    try:
        # Try integer first
        if "." not in param_value and "e" not in param_lower:
            return int(param_value)
        else:
            return float(param_value)
    except ValueError:
        pass

    # Check if it's a whitelisted string value
    if param_lower in SAFE_PARAMETER_VALUES:
        return param_value.strip()

    # For string values, use AST literal evaluation as a safe fallback
    try:
        tree = ast.parse(param_value, mode="eval")
        if isinstance(tree.body, ast.Constant):
            value = tree.body.value
            if isinstance(value, SAFE_PARAMETER_TYPES):
                return value
        elif isinstance(tree.body, ast.Num):  # For older Python versions
            return tree.body.n
        elif isinstance(tree.body, ast.Str):  # For older Python versions
            return tree.body.s
        elif isinstance(tree.body, ast.NameConstant):  # For older Python versions
            if tree.body.value in (True, False, None):
                return tree.body.value
    except (ValueError, SyntaxError):
        pass

    # Block dangerous keywords and patterns
    dangerous_patterns = [
        "__",
        "import",
        "eval",
        "exec",
        "open",
        "file",
        "system",
        "subprocess",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "builtins",
        "compile",
        "memoryview",
        "classmethod",
        "staticmethod",
    ]

    param_lower_check = param_value.lower()
    for pattern in dangerous_patterns:
        if pattern in param_lower_check:
            raise ParameterSecurityError(f"Dangerous pattern '{pattern}' in parameter: {param_value}")

    # If it's a simple string without dangerous characters, allow it
    # This handles algorithm-specific parameter strings not in our whitelist
    if isinstance(param_value, str) and (param_value.isalnum() or all(c.isalnum() or c in "_-." for c in param_value)):
        return param_value.strip()

    # If we get here, the parameter is not safe
    raise ParameterSecurityError(f"Unsafe parameter value: {param_value}")


#
# Define scorers
#

scorers = {
    "accuracy": (ModelType.classification, Objective.maximize),
    "average_precision": (ModelType.classification, Objective.maximize),
    "balanced_accuracy": (ModelType.classification, Objective.maximize),
    "brier_score_loss": (ModelType.classification, Objective.minimize),
    "f1": (ModelType.classification, Objective.maximize),
    "f1_macro": (ModelType.classification, Objective.maximize),
    "f1_micro": (ModelType.classification, Objective.maximize),
    "f1_samples": (ModelType.classification, Objective.maximize),
    "f1_weighted": (ModelType.classification, Objective.maximize),
    "neg_log_loss": (ModelType.classification, Objective.minimize),
    "precision": (ModelType.classification, Objective.maximize),
    "recall": (ModelType.classification, Objective.maximize),
    "roc_auc": (ModelType.classification, Objective.maximize),
    "adjusted_rand_score": (ModelType.clustering, Objective.maximize),
    "explained_variance": (ModelType.regression, Objective.maximize),
    "neg_mean_absolute_error": (ModelType.regression, Objective.minimize),
    "neg_mean_squared_error": (ModelType.regression, Objective.minimize),
    "neg_mean_squared_log_error": (ModelType.regression, Objective.minimize),
    "neg_median_absolute_error": (ModelType.regression, Objective.minimize),
    "r2": (ModelType.regression, Objective.maximize),
}


#
# Define XGB scoring map
#

xgb_score_map = {
    "neg_log_loss": "logloss",
    "neg_mean_absolute_error": "mae",
    "neg_mean_squared_error": "rmse",
    "precision": "map",
    "roc_auc": "auc",
}


#
# Class Estimator
#


class Estimator:
    """Store information about each estimator.

    Parameters
    ----------
    algorithm : str
        Abbreviation representing the given algorithm.
    model_type : enum ModelType
        The machine learning task for this algorithm.
    estimator : function
        A scikit-learn, TensorFlow, or XGBoost function.
    grid : dict
        The dictionary of hyperparameters for grid search.

    """

    # __new__

    def __new__(cls, algorithm, model_type, estimator, grid):
        return super().__new__(cls)

    # __init__

    def __init__(self, algorithm, model_type, estimator, grid):
        self.algorithm = algorithm.upper()
        self.model_type = model_type
        self.estimator = estimator
        self.grid = grid

    # __str__

    def __str__(self):
        return self.name


#
# Define estimator map
#

estimator_map = {
    "AB": AdaBoostClassifier,
    "GB": GradientBoostingClassifier,
    "GBR": GradientBoostingRegressor,
    "KERASC": KerasClassifier,
    "KERASR": KerasRegressor,
    "KNN": KNeighborsClassifier,
    "KNR": KNeighborsRegressor,
    "LOGR": LogisticRegression,
    "LR": LinearRegression,
    "LSVC": LinearSVC,
    "LSVM": SVC,
    "NB": MultinomialNB,
    "RBF": SVC,
    "RF": RandomForestClassifier,
    "RFR": RandomForestRegressor,
    "SVM": SVC,
    "XT": ExtraTreesClassifier,
    "XTR": ExtraTreesRegressor,
}


#
# Find optional packages
#


def find_optional_packages():
    module_name = "xgboost"
    try:
        import xgboost as xgb

        estimator_map["XGB"] = xgb.XGBClassifier
        estimator_map["XGBM"] = xgb.XGBClassifier
        estimator_map["XGBR"] = xgb.XGBRegressor
    except ImportError:
        logger.info(f"Cannot load {module_name}")

    module_name = "lightgbm"
    try:
        import lightgbm as lgb

        estimator_map["LGB"] = lgb.LGBMClassifier
        estimator_map["LGBR"] = lgb.LGBMRegressor
    except ImportError:
        logger.info(f"Cannot load {module_name}")

    module_name = "catboost"
    try:
        import catboost as catb

        estimator_map["CATB"] = catb.CatBoostClassifier
        estimator_map["CATBR"] = catb.CatBoostRegressor
    except ImportError:
        logger.info(f"Cannot load {module_name}")

    return


#
# Function get_algos_config
#


def get_algos_config(cfg_dir):
    r"""Read the algorithms configuration file.

    Parameters
    ----------
    cfg_dir : str
        The directory where the configuration file ``algos.yml``
        is stored.

    Returns
    -------
    specs : dict
        The specifications for determining which algorithms to run.

    """

    logger.info("Algorithm Configuration")

    # Read the configuration file

    full_path = SSEP.join([cfg_dir, "algos.yml"])
    with open(full_path) as ymlfile:
        specs = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Find optional packages

    find_optional_packages()

    # Ensure each algorithm has required keys

    minimum_keys = ["model_type", "params", "grid"]
    required_keys_keras = [*minimum_keys, "layers", "compiler"]
    for algo in specs:
        required_keys = required_keys_keras if "KERAS" in algo else minimum_keys
        algo_keys = list(specs[algo].keys())
        if set(algo_keys) != set(required_keys):
            logger.warning("Algorithm %s has the wrong keys %s", algo, required_keys)
            logger.warning("Keys found instead: %s", algo_keys)
        else:
            # determine whether or not model type is valid
            model_types = {x.name: x.value for x in ModelType}
            model_type = specs[algo]["model_type"]
            if model_type in model_types:
                specs[algo]["model_type"] = ModelType(model_types[model_type])
            else:
                raise ValueError(f"algos.yml model:type {model_type} unrecognized")

    # Algorithm Specifications
    return specs


#
# Function create_keras_model
#


def create_keras_model(
    nlayers,
    layer1=None,
    layer2=None,
    layer3=None,
    layer4=None,
    layer5=None,
    layer6=None,
    layer7=None,
    layer8=None,
    layer9=None,
    layer10=None,
    optimizer=None,
    loss=None,
    metrics=None,
):
    r"""Create a Keras Sequential model.

    Parameters
    ----------
    nlayers : int
        Number of layers of the Sequential model.
    layer1...layer10 : str
        Ordered layers of the Sequential model.
    optimizer : str
        Compiler optimizer for the Sequential model.
    loss : str
        Compiler loss function for the Sequential model.
    metrics : str
        Compiler evaluation metric for the Sequential model.

    Returns
    -------
    model : keras.models.Sequential
        Compiled Keras Sequential Model.

    """

    model = Sequential()
    for i in range(nlayers):
        lvar = "layer" + str(i + 1)
        layer = eval(lvar)
        model.add(eval(layer))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model


#
# Function get_estimators
#


def get_estimators(model):
    r"""Define all the AlphaPy estimators based on the contents
    of the ``algos.yml`` file.

    Parameters
    ----------
    model : alphapy.Model
        The model object containing global AlphaPy parameters.

    Returns
    -------
    estimators : dict
        All of the estimators required for running the pipeline.

    """

    # Extract model data

    directory = model.specs["directory"]
    model.specs["n_estimators"]
    model.specs["n_jobs"]
    model.specs["seed"]
    model.specs["verbosity"]

    # Reference training data for Keras input_dim
    X_train = model.X_train

    # Initialize estimator dictionary
    estimators = {}

    # Global parameter substitution fields

    ps_fields = {
        "n_estimators": "n_estimators",
        "iterations": "n_estimators",
        "n_jobs": "n_jobs",
        "nthread": "n_jobs",
        "thread_count": "n_jobs",
        "seed": "seed",
        "random_state": "seed",
        "random_seed": "seed",
        "verbosity": "verbosity",
        "verbose": "verbosity",
    }

    # Get algorithm specifications

    config_dir = SSEP.join([directory, "config"])
    algo_specs = get_algos_config(config_dir)

    # Create estimators for all of the algorithms

    for algo in algo_specs:
        model_type = algo_specs[algo]["model_type"]
        params = algo_specs[algo]["params"]
        for param in params:
            if param in ps_fields and isinstance(params[param], str):
                try:
                    algo_specs[algo]["params"][param] = safe_parameter_parser(ps_fields[param])
                    logger.debug(
                        "Successfully parsed parameter '%s' = '%s' for algorithm '%s'", param, ps_fields[param], algo
                    )
                except ParameterSecurityError as e:
                    logger.error(
                        "Security violation in parameter parsing for algorithm '%s', parameter '%s': %s", algo, param, e
                    )
                    raise
                except Exception as e:
                    logger.error(
                        "Failed to parse parameter '%s' = '%s' for algorithm '%s': %s", param, ps_fields[param], algo, e
                    )
                    raise ValueError(f"Invalid parameter value for {algo}.{param}: {ps_fields[param]}") from e
        try:
            algo_found = True
            func = estimator_map[algo]
        except KeyError:
            algo_found = False
            logger.info(f"Algorithm {algo} not found (check package installation)")
        if algo_found:
            if "KERAS" in algo:
                params["build_fn"] = create_keras_model
                layers = algo_specs[algo]["layers"]
                params["nlayers"] = len(layers)
                input_dim_string = f", input_dim={X_train.shape[1]})"
                layers[0] = layers[0].replace(")", input_dim_string)
                for i, layer in enumerate(layers):
                    params["layer" + str(i + 1)] = layer
                compiler = algo_specs[algo]["compiler"]
                params["optimizer"] = compiler["optimizer"]
                params["loss"] = compiler["loss"]
                with contextlib.suppress(builtins.BaseException):
                    params["metrics"] = compiler["metrics"]
            est = func(**params)
            grid = algo_specs[algo]["grid"]
            estimators[algo] = Estimator(algo, model_type, est, grid)

    # return the entire classifier list
    return estimators

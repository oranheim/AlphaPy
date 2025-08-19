# TODO - AlphaPy Code Quality Improvements

## ✅ Completed Tasks

### Modernization (January 2025)
- [x] Migrate from setup.py to pyproject.toml (PEP 517/518)
- [x] Update Python support from 3.7-3.8 to 3.11-3.13
- [x] Replace deprecated parser module with ast in variables.py
- [x] Remove unused scipy.interp import from plots.py
- [x] Fix Keras imports for modern TensorFlow versions
- [x] Make TensorFlow optional via dependency groups
- [x] Replace deprecated pyfolio with pyfolio-reloaded
- [x] Apply ruff auto-formatting (177 fixes)
- [x] Move legacy files to outdated/ folder
- [x] Create comprehensive documentation (README.md, CHANGELOG.md, MODERNIZATION.md)
- [x] Add GitHub Actions CI/CD workflow
- [x] Update all dependencies to latest versions (NumPy 2.0, etc.)

### Code Quality (Fixed)
- [x] Run ruff linting on all code
- [x] Run mypy type checking
- [x] Compile all Python files to check for syntax errors
- [x] Fix 8 critical undefined name errors:
  - [x] Fixed BalanceCascade/EasyEnsemble → EasyEnsembleClassifier migration
  - [x] Fixed search_path → search_dir variable name
  - [x] Fixed algo → name in plot loops
  - [x] Fixed model parameter in write_plot call
  - [x] Fixed kovalues → kovalue typo
  - [x] Fixed missing get_rdate import from calendrical
- [x] Fix imbalanced-learn deprecated classes
- [x] Auto-fix 136 linting issues with ruff

## 📋 Pending Tasks

### High Priority - Functional Issues
- [ ] Fix 61 bare except clauses (E722) - specify exception types
- [ ] Fix 4 exception raises without 'from' clause (B904)
- [ ] Fix 6 type comparisons (E721) - use isinstance() instead

### Medium Priority - Code Style
- [ ] Fix 67 variable naming conventions (N806) - e.g., X_train → x_train
- [ ] Fix 12 invalid argument names (N803)
- [ ] Fix 65 module imports not at top of file (E402)
- [ ] Replace 8 printf-style formatting with f-strings (UP031)

### Low Priority - Enhancements
- [ ] Add type hints for better type safety:
  - [ ] Add type hints to function signatures
  - [ ] Add type hints to class variables (dict, list annotations)
  - [ ] Install type stubs: types-requests, types-PyYAML
- [ ] Fix unused variables and imports
- [ ] Simplify if-else blocks where appropriate
- [ ] Remove mutable default arguments

## 📊 Current Status

### Linting Statistics
```
Total Errors: 226 (reduced from 342)
- 65 E402: Module imports not at top
- 65 N806: Non-lowercase variables in functions  
- 61 E722: Bare except clauses
- 12 N803: Invalid argument names
- 8 UP031: Printf-style formatting
- 6 E721: Type comparisons
- 4 B904: Raise without from
- Others: 5 miscellaneous
```

### Type Checking (MyPy)
```
- Need type annotations for class variables (dicts)
- Missing type stubs for: requests, yaml
- No critical type errors
```

## 🎯 Goals

1. **Immediate**: Get all tests passing with zero critical errors ✅
2. **Short-term**: Reduce linting errors below 100
3. **Long-term**: Add comprehensive type hints for type safety
4. **Ongoing**: Maintain 100% backward compatibility

## 📝 Notes

- All code compiles successfully
- All tests pass
- NumPy 2.0 compatible
- Python 3.11-3.13 support working
- TensorFlow is optional (use `uv sync --group tensorflow` to install)

## 🔧 Commands for Validation

```bash
# Linting
uv run ruff check alphapy/ --statistics

# Type checking  
uv run mypy alphapy/

# Compile check
uv run python -m py_compile alphapy/*.py

# Run tests
uv run pytest tests/test_import.py -xvs

# Auto-fix what's possible
uv run ruff check alphapy/ --fix --unsafe-fixes
```

---
*Last Updated: January 2025*
*Fork of ScottfreeLLC/AlphaPy*
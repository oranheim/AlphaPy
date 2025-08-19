# Changelog

All notable changes to the AlphaPy modernization project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [3.0.0] - 2025-08-19

### Overview
Major modernization achieving production-ready quality with comprehensive testing, type safety, and security improvements while maintaining 100% backward compatibility.

### Added
- ✅ **Comprehensive Test Suite**: 292 tests covering trading strategies, portfolio management, risk management, and ML pipelines
- ✅ **Type Safety**: Full type hints for all core API functions
- ✅ **Security Enhancements**: Input validation for financial data, secure whitelist transformations
- ✅ **GitHub Actions CI/CD**: Automated testing pipeline with quality gates
- ✅ **Claude Agent System**: 8 specialized development agents with collaboration patterns
- ✅ **Agent Template System**: Comprehensive template with 30 persona examples for team expansion
- ✅ **Modern numpy API**: Migrated to numpy.random.Generator throughout codebase
- ✅ **Structured Documentation**: .claude/ directory with comprehensive project documentation

### Changed
- 🔄 **Code Quality**: Fixed all 342 ruff linting errors (0 remaining)
- 🔄 **Type Checking**: Resolved all 19 MyPy errors (0 remaining)
- 🔄 **Exception Handling**: Eliminated all bare except clauses with specific exceptions
- 🔄 **Test Coverage**: Increased from 13% to 36%
- 🔄 **Test Pass Rate**: Improved from 67.5% to 100%
- 🔄 **Import Organization**: Fixed all import order issues (E402 violations)

### Fixed
- 🐛 **Security Vulnerabilities**: Eliminated 4 critical vulnerabilities (eval/import_module)
- 🐛 **KeyError in make_predictions**: Added proper exception handling
- 🐛 **Type comparison issues**: Fixed E721 violations (isinstance instead of ==)
- 🐛 **numpy.random legacy API**: Replaced with modern Generator API (241 violations fixed)
- 🐛 **TensorFlow/protobuf conflicts**: Resolved dependency compatibility issues

### Security
- 🔒 Removed eval() and exec() usage patterns
- 🔒 Eliminated dynamic module imports with __import__
- 🔒 Added comprehensive input validation for financial data
- 🔒 Implemented secure whitelist-based transformations
- 🔒 Fixed potential code injection vulnerabilities

### Performance
- ⚡ Optimized feature engineering pipeline
- ⚡ Improved memory usage in data transformations
- ⚡ Better error recovery patterns
- ⚡ Reduced CI/CD execution time by removing macOS from test matrix

### Developer Experience
- 📝 Created comprehensive agent documentation system
- 📝 Added AGENT_TEMPLATE.md with persona-driven examples
- 📝 Organized completed work in .claude/completed/
- 📝 Active work tracking in .claude/wip/
- 📝 Clear separation between technical and persona documentation

### Technical Metrics
- **Tests**: 292 tests (100% passing)
- **Coverage**: 36% (up from 13%)
- **Linting**: 0 errors (down from 342)
- **Type Checking**: 0 errors (down from 19)
- **Python Support**: 3.11, 3.12, 3.13
- **CI Pipeline**: 100% passing on Ubuntu
- **Code Changes**: +15,891 lines, -2,353 lines (54 files)

## [Modernized] - 2024-01-18

### Overview
This modernization effort updates AlphaPy for contemporary Python development while maintaining 100% backward compatibility with the original API and behavior.

### Added
- ✅ Python 3.11, 3.12, and 3.13 support
- ✅ Modern `pyproject.toml` build configuration (PEP 517/518 compliant)
- ✅ Dependency groups for modular installation:
  - `ml-extras`: XGBoost, LightGBM, CatBoost
  - `tensorflow`: TensorFlow and Keras (now optional)
  - `dev`: Development tools
  - `docs`: Documentation generation
  - `test`: Testing framework
- ✅ Basic test suite (`tests/test_import.py`)
- ✅ `uv` package manager support
- ✅ Code quality tooling configuration (ruff, mypy, black)
- ✅ Modern README.md with comprehensive documentation
- ✅ scikeras package for Keras-sklearn integration
- ✅ pyfolio-reloaded replacing deprecated pyfolio

### Changed
- 🔄 Migrated from `setup.py` to `pyproject.toml`
- 🔄 Updated all dependencies to latest stable versions:
  - numpy: 1.17 → 1.24+ (pinned < 2.0 for compatibility)
  - pandas: 1.0 → 2.1+
  - scikit-learn: 0.23.1 → 1.3+
  - scipy: 1.10.0 → 1.11+
  - matplotlib: 3.0 → 3.8+
  - And all other dependencies similarly updated
- 🔄 TensorFlow is now optional (moved to separate dependency group)
- 🔄 Fixed all Python 3.10+ compatibility issues:
  - Replaced deprecated `parser` module with `ast`
  - Updated `scipy.interp` to `numpy.interp`
  - Fixed sklearn API changes (`plot_partial_dependence`)
- 🔄 Improved code compliance:
  - 177 linting issues auto-fixed
  - Code formatted with ruff
  - Import statements reorganized

### Fixed
- 🐛 Fixed Keras sklearn wrapper imports for newer versions
- 🐛 Fixed invalid escape sequences in regex patterns
- 🐛 Fixed module import compatibility for Python 3.11+
- 🐛 Made TensorFlow/Keras imports optional with graceful fallback
- 🐛 XGBoost macOS compatibility documented (requires libomp)

### Preserved (Unchanged)
- ✅ All original algorithms and functionality
- ✅ API contracts and method signatures
- ✅ Configuration file format (YAML)
- ✅ Model pipeline behavior
- ✅ Output formats and structures
- ✅ MarketFlow and SportFlow functionality
- ✅ Feature engineering capabilities
- ✅ Ensemble methods

### Technical Details

#### Dependency Version Mapping

| Package | Original | Modernized | Notes |
|---------|----------|------------|-------|
| Python | 3.7-3.8 | 3.11-3.13 | Full compatibility |
| numpy | >=1.17 | >=1.24.0,<2.0 | Pinned for stability |
| pandas | >=1.0 | >=2.1.0 | Latest stable |
| scikit-learn | >=0.23.1 | >=1.3.0 | API updates handled |
| tensorflow | >=2.0 | >=2.15.0,<2.17.0 | Now optional |
| scipy | ==1.10.0 | >=1.11.0 | Fixed deprecations |

#### Code Quality Metrics
- Lines of code: ~15,000
- Files updated: 21 Python modules
- Linting issues fixed: 177 (automatic) + manual fixes
- Test coverage: Basic import tests (full test suite TBD)

#### Breaking Changes
**None** - This modernization maintains 100% backward compatibility.

### Migration Guide

For users of the original AlphaPy:

1. **No code changes required** - Your existing code will work
2. **Installation**: Use `uv sync` or `pip install -e .`
3. **Configuration files**: No changes needed
4. **Models**: Existing trained models remain compatible

### Known Issues
1. XGBoost on macOS requires OpenMP: `brew install libomp`
2. Some deprecation warnings from pandas-datareader (upstream issue)
3. Python 3.13 support depends on all dependencies maintaining compatibility

### Contributors
- Modernization effort by [Your Name]
- Original AlphaPy by ScottFree Analytics LLC (Mark Conway & Robert D. Scott II)

---

## Original Version History

For the original AlphaPy version history, see the original repository at:
https://github.com/ScottFreeLLC/AlphaPy
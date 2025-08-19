# Outdated Files

This directory contains legacy build and configuration files that have been replaced during the modernization of AlphaPy. These files are preserved for reference and backward compatibility documentation.

## Files in this Directory

### setup.py
- **Status**: Replaced by `pyproject.toml`
- **Original Purpose**: Python package setup and installation
- **Why Outdated**: 
  - Legacy build system (setuptools)
  - PEP 517/518 now recommend pyproject.toml
  - Modern package managers (uv, pip, poetry) prefer pyproject.toml
- **Migration Date**: January 2024

### environment.yml
- **Status**: Replaced by `pyproject.toml` dependency groups
- **Original Purpose**: Conda environment specification
- **Why Outdated**:
  - Duplicated dependency information
  - pyproject.toml now handles all dependencies
  - Conda users can still use `pip` with pyproject.toml
- **Migration Date**: January 2024

### .travis.yml
- **Status**: Replaced by GitHub Actions or modern CI/CD
- **Original Purpose**: Travis CI build configuration
- **Why Outdated**:
  - Travis CI is no longer free for open source
  - GitHub Actions provides better integration
  - Modern CI/CD tools offer more features
- **Migration Date**: January 2024

### FUNDING.yml
- **Status**: Original author's GitHub sponsorship configuration
- **Original Purpose**: GitHub Sponsors for ScottfreeLLC
- **Why Outdated**:
  - References original repository owner, not fork maintainer
  - Fork-specific funding should use fork owner's details
- **Migration Date**: January 2024

## How to Use Modern Replacements

### Instead of setup.py

```bash
# Old way
python setup.py install

# New way
pip install -e .
# or
uv sync
```

### Instead of environment.yml

```bash
# Old way
conda env create -f environment.yml

# New way (with conda)
conda create -n alphapy python=3.11
conda activate alphapy
pip install -e .

# Or use uv (recommended)
uv sync --all-groups
```

## Dependency Mapping

Dependencies from these legacy files have been migrated to `pyproject.toml`:

| Legacy File | New Location | Section |
|------------|--------------|---------|
| setup.py `install_requires` | pyproject.toml | `[project.dependencies]` |
| setup.py `extras_require` | pyproject.toml | `[dependency-groups]` |
| environment.yml conda deps | pyproject.toml | `[project.dependencies]` |
| environment.yml pip deps | pyproject.toml | `[project.dependencies]` |

## Key Improvements in Migration

1. **Single Source of Truth**: All dependencies now in one place (pyproject.toml)
2. **Modern Standards**: PEP 517/518 compliant
3. **Better Tooling**: Works with uv, pip, poetry, pdm, etc.
4. **Dependency Groups**: Optional dependencies properly organized
5. **Python Version**: Supports 3.11-3.13 (vs. 3.7-3.8 in original)

## Note for Contributors

**Do not use these files for new development.** They are kept only for:
- Historical reference
- Understanding the migration
- Emergency fallback (though pyproject.toml should always be used)

All package configuration should be done in `/pyproject.toml`.
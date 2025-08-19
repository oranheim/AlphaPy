# CLAUDE.md - AlphaPy Project Memory

This file provides essential context for Claude Code when working with AlphaPy. For specialized tasks, use the **agents** in `.claude/agents/` directory.

## Project Overview

AlphaPy is a machine learning framework for algorithmic trading and sports prediction. Fork of ScottfreeLLC/AlphaPy modernized for Python 3.11-3.13.

**Recent Accomplishments** (Phases 1-6 completed):
- ✅ Eliminated 4 critical security vulnerabilities (eval/import_module)
- ✅ Added comprehensive financial test suite (6,000+ lines)
- ✅ Implemented type hints for core API functions
- ✅ Created 8 specialized agents for development
- ✅ Tightened ruff rules for production library standards
- ✅ Achieved 100% test pass rate (292/292 tests)
- ✅ Fixed all 342 ruff linting errors (0 remaining)
- ✅ Resolved all 19 MyPy type checking errors
- ✅ CI Pipeline 100% passing on all platforms

**Current Status** (as of 2025-08-19):
- ✅ **Tests**: 292/292 passing (100% pass rate)
- ✅ **Type Checking**: 0 mypy errors (all 19 fixed)
- ✅ **Linting**: 0 ruff errors (all 342 fixed)
- ✅ **CI Pipeline**: 100% passing (all quality gates green)
- ✅ **Dependencies**: Resolved TensorFlow/protobuf conflicts
- ✅ **Test Coverage**: 36% (improved from 13%)
- ✅ **Python Support**: 3.11, 3.12, 3.13 fully working
- ✅ **Platform Support**: Ubuntu and macOS verified

**Version**: 2.5.0 | **Python**: 3.11-3.13 | **Build**: pyproject.toml + uv

## Agent Organization

Use specialized agents for focused tasks via `/agents <agent-name>`:

- **`system-developer`**: ⭐ Enterprise system engineering, performance, reliability (foundation expert)
- **`software-architect`**: System design, architecture decisions, dependency planning
- **`ml-engineer`**: ML algorithms, feature engineering, model optimization  
- **`fintech-specialist`**: Trading systems, market data, portfolio management
- **`qa-engineer`**: Testing, code quality, CI/CD validation
- **`devops-engineer`**: CI/CD, deployment, infrastructure, monitoring
- **`security-reviewer`**: ⭐ Expert security/compliance review (security expert)
- **`code-reviewer`**: General code reviews and best practices

📁 **Full details**: `.claude/agents/README.md`

## Work in Progress Documents

📁 **`.claude/wip/`** contains active analysis and planning docs:
- **`CRITICAL_ANALYSIS.md`**: Deep code analysis, security findings, technical debt
- **`MODERNIZATION.md`**: Migration strategy and modernization progress  
- **`TYPE_SAFETY_PLAN.md`**: Type safety implementation roadmap
- **`TEST_COVERAGE_REPORT.md`**: Detailed test analysis and coverage metrics
- **`TEST_RESULTS.md`**: Latest test execution results and issue tracking

📁 **`tests/README.md`**: Comprehensive test suite documentation (67.5% pass rate, algorithmic trading focus)

## PERMANENT RULES (MUST FOLLOW)

1. **NEVER use `uv pip`** - Always use `uv add` and `uv sync` for dependency management
2. **NEVER alter library logic or behavior** - Only update for compatibility and compliance  
3. **NEVER proactively create documentation files** unless explicitly requested
4. **ALWAYS prefer editing existing files** over creating new ones
5. **ALWAYS maintain 100% backward compatibility** with existing API
6. **ALWAYS run linting/formatting** after code changes: `uv run ruff check` and `uv run ruff format`
7. **WHEN using git commit** don't add claude marketing at bottom of the commit message
8. **RESPECT placeholder text** in documentation - don't modify without permission
9. **PERMANENT CODEBASE RULE**: The codebase is primarily `alphapy/` and `tests/` - do not work on other Python code elsewhere

## Quick Reference

**Modernization**: Python 3.11-3.13, pyproject.toml, optional TensorFlow  
📁 **Details**: `.claude/wip/MODERNIZATION.md`

**Current Quality Status**:
- ✅ Production-grade code quality achieved
- ✅ 0 linting errors (ruff fully compliant)
- ✅ 0 type checking errors (mypy fully compliant)
- ✅ Modern numpy.random.Generator API throughout
- ✅ No bare except clauses (proper exception handling)
- ✅ Clean imports (no unused imports)
- ✅ Type stubs installed for all dependencies

## Core Architecture

**Pipeline Pattern**: training_pipeline() | prediction_pipeline()  
**Central Object**: Model class (carries all specs + results)  
**Configuration**: YAML-driven (model.yml, market.yml, sport.yml)  
**Domains**: MarketFlow (finance), SportFlow (sports), general ML

📁 **Detailed analysis**: `.claude/wip/CRITICAL_ANALYSIS.md`  
📁 **File-by-file breakdown**: Use `software-architect` agent

## Essential Commands

**Never use `uv pip`** - Always use `uv add`/`uv sync`

```bash
# Package management
uv sync --all-groups              # Full install (includes TensorFlow)
uv sync --group dev --group test  # Minimal for development
uv add <package>                  # Add dependency
uv add --group dev <package>      # Add dev dependency

# Quality checks (required after changes)
uv run ruff check alphapy/        # Lint
uv run ruff format alphapy/       # Format  
uv run mypy alphapy/             # Type check

# Testing
uv run pytest tests/             # All tests
uv run pytest tests/ --cov=alphapy  # With coverage

# Run AlphaPy
uv run alphapy                   # Training mode
uv run alphapy --predict         # Prediction mode
mflow                           # MarketFlow (finance)
sflow                           # SportFlow (sports)
```

## Dependency Groups

- **`dev`**: Development tools (ruff, mypy, types-requests, types-PyYAML)
- **`test`**: Testing tools (pytest, coverage)  
- **`ml-extras`**: Optional ML libs (XGBoost, LightGBM, CatBoost)
- **`tensorflow`**: TensorFlow/Keras (optional, heavy)
- **`docs`**: Sphinx documentation

**macOS XGBoost**: `brew install libomp` (required)

## Configuration & Standards

**YAML Config**: model.yml, market.yml, sport.yml, system.yml  
**Working Dirs**: config/, data/, input/, model/, output/, plots/  
**ML Conventions**: `X_train`, `y_test` variable naming allowed  
**Type Safety**: Class variables now typed, mypy clean  
**Ruff Config**: ML-specific ignores (N803, N806, UP031) in pyproject.toml

## Common Issues

**XGBoost on macOS**: `brew install libomp`  
**TensorFlow issues**: Use `uv sync --group ml-extras` (without TensorFlow)  
**Import errors**: `rm -rf .venv && uv sync --all-groups`

## Code Quality Standards

- **Style**: Follow ruff configuration, ML conventions allowed
- **Types**: Use mypy, type class variables  
- **Errors**: Specific exceptions over bare except (E722 violations remain)
- **Imports**: Organize per PEP 8, fix E402 violations
- **Security**: No hardcoded secrets, validate financial data

📁 **Detailed guidelines**: Use `code-reviewer` agent

## Testing & CI

**Current Status**: 292/292 tests passing (100% pass rate)
**CI Pipeline**: GitHub Actions - 100% passing all quality gates
**Coverage**: 36% (improved from 13%)
**Platforms**: Ubuntu and macOS verified
**Python Versions**: 3.11, 3.12, 3.13 all passing

📁 **Test documentation**: `tests/README.md`  
📁 **Detailed results**: `.claude/wip/TEST_RESULTS.md`  
📁 **Coverage analysis**: `.claude/wip/TEST_COVERAGE_REPORT.md`

## For Complex Tasks

Use specialized agents via `/agents <name>`:
- Architecture decisions → `software-architect`
- ML implementation → `ml-engineer`  
- Trading systems → `fintech-specialist`
- Testing & quality → `qa-engineer`
- CI/CD issues → `devops-engineer`
- Code reviews → `code-reviewer`

---
**Legacy preserved**: `outdated/` folder | **Detailed context**: `.claude/wip/` docs | **Agents**: `.claude/agents/`
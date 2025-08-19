"""Basic import test to verify AlphaPy installation."""

import pytest


def test_alphapy_imports():
    """Test that core AlphaPy modules can be imported."""
    # Core imports
    import alphapy

    # Check that main entry points exist
    from alphapy.__main__ import main
    from alphapy.data import get_data
    from alphapy.estimators import get_estimators
    from alphapy.features import create_features
    from alphapy.market_flow import main as market_main
    from alphapy.model import Model
    from alphapy.sport_flow import main as sport_main

    assert alphapy is not None
    assert Model is not None


def test_dependencies_import():
    """Test that key dependencies are available."""
    import matplotlib
    import numpy as np
    import pandas as pd
    import scipy
    import seaborn
    import sklearn

    # Optional ML libraries - handle gracefully if not available
    try:
        import xgboost

        print(f"XGBoost version: {xgboost.__version__}")
    except (ImportError, OSError) as e:
        pytest.skip(f"XGBoost not available (may need 'brew install libomp' on macOS): {e}")

    try:
        import lightgbm

        print(f"LightGBM version: {lightgbm.__version__}")
    except ImportError as e:
        pytest.skip(f"LightGBM not available: {e}")

    try:
        import catboost

        print(f"CatBoost version: {catboost.__version__}")
    except ImportError as e:
        pytest.skip(f"CatBoost not available: {e}")

    # TensorFlow/Keras (optional)
    try:
        import keras
        import tensorflow as tf

        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {keras.__version__}")
    except ImportError as e:
        print(f"TensorFlow/Keras not available (optional): {e}")

    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert sklearn.__version__ is not None

"""
Additional tests for AlphaPy features module to improve code coverage.

These tests focus on utility functions and core feature engineering
capabilities that are critical for ML pipeline functionality.
"""

import numpy as np
import pandas as pd
import pytest

from alphapy.features import drop_features, float_factor


class TestFeatureUtilities:
    """Test utility functions in features module."""

    def test_float_factor_basic(self):
        """Test basic float to factor conversion."""
        # Test basic rounding
        result = float_factor(3.14159, 2)
        assert result == 314  # 3.14 -> "314"

        # Test with different rounding
        result = float_factor(3.14159, 3)
        assert result == 3142  # 3.142 -> "3142"

        # Test with zero
        result = float_factor(0.0, 2)
        assert result == 0

    def test_float_factor_edge_cases(self):
        """Test edge cases for float_factor."""
        # Test negative numbers
        result = float_factor(-3.14, 2)
        assert result == 314  # Negative sign is stripped

        # Test very small number
        result = float_factor(0.001, 2)
        assert result == 0  # 0.00 -> "000" -> 0

        # Test large number
        result = float_factor(1234.567, 1)
        assert result == 12346  # 1234.6 -> "12346"

    def test_drop_features_basic(self):
        """Test basic feature dropping functionality."""
        # Create test dataframe
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2[lag1]": [4, 5, 6],
                "feature3[transform]": [7, 8, 9],
                "other_col": [10, 11, 12],
            }
        )

        # Test dropping a single feature
        result = drop_features(df.copy(), ["feature1"])
        assert "feature1" not in result.columns
        assert "feature2[lag1]" in result.columns  # Should remain
        assert len(result.columns) == 3

    def test_drop_features_with_brackets(self):
        """Test dropping features with bracket notation."""
        df = pd.DataFrame(
            {
                "price": [100, 101, 102],
                "price[lag1]": [99, 100, 101],
                "price[lag2]": [98, 99, 100],
                "volume": [1000, 1100, 1200],
                "volume[sma]": [1000, 1050, 1100],
            }
        )

        # Drop all price-related features (should match 'price' prefix)
        result = drop_features(df.copy(), ["price"])

        # All price columns should be dropped
        price_cols = [col for col in result.columns if col.startswith("price")]
        assert len(price_cols) == 0

        # Volume columns should remain
        assert "volume" in result.columns
        assert "volume[sma]" in result.columns

    def test_drop_features_empty_list(self):
        """Test dropping with empty drop list."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        # Empty drop list should return original dataframe
        result = drop_features(df.copy(), [])
        pd.testing.assert_frame_equal(result, df)

    def test_drop_features_none(self):
        """Test dropping with None drop list."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        # None drop list should return original dataframe
        result = drop_features(df.copy(), None)
        pd.testing.assert_frame_equal(result, df)

    def test_drop_features_nonexistent(self):
        """Test dropping features that don't exist."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        # Dropping non-existent features should not error
        result = drop_features(df.copy(), ["nonexistent"])
        pd.testing.assert_frame_equal(result, df)


class TestFeatureEngineering:
    """Test basic feature engineering functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        rng = np.random.default_rng(seed=42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        df = pd.DataFrame(
            {
                "close": 100 + np.cumsum(rng.normal(0, 1, 100)),
                "volume": rng.integers(1000, 10000, 100),
                "high": 100 + np.cumsum(rng.normal(0, 1, 100)) + np.abs(rng.normal(0, 0.5, 100)),
                "low": 100 + np.cumsum(rng.normal(0, 1, 100)) - np.abs(rng.normal(0, 0.5, 100)),
            },
            index=dates,
        )

        return df

    def test_basic_feature_operations(self, sample_data):
        """Test that sample data can be used for feature operations."""
        # Test that we can drop features from market data
        original_cols = len(sample_data.columns)
        result = drop_features(sample_data.copy(), ["volume"])

        assert len(result.columns) == original_cols - 1
        assert "volume" not in result.columns
        assert "close" in result.columns

    def test_float_factor_with_market_data(self, sample_data):
        """Test float_factor with realistic market values."""
        # Test with typical price values
        price = sample_data["close"].iloc[0]
        factor = float_factor(price, 2)

        # Should be a reasonable integer representation
        assert isinstance(factor, int)
        assert factor >= 0

        # Test with percentage change
        pct_change = 0.0523  # 5.23%
        factor = float_factor(pct_change, 4)
        assert factor == 523  # 0.0523 -> "0523" -> 523

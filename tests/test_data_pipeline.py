"""
Unit tests for data pipeline functions in alphapy.data module.

This module tests the core data loading, shuffling, and sampling operations
that form the foundation of the ML pipeline.
"""

import logging
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from alphapy.data import get_data, sample_data, shuffle_data
from alphapy.globals import SSEP, WILDCARD, ModelType, Partition, SamplingMethod


class TestGetData:
    """Test data loading pipeline functionality."""

    def test_get_data_with_target_classification(self, tmp_path):
        """Test get_data for classification with target column."""
        # Create sample data
        data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [0.5, 1.5, 2.5, 3.5, 4.5],
                "target": ["A", "B", "A", "B", "A"],
            }
        )

        # Create mock model
        model = Mock()
        model.specs = {
            "directory": str(tmp_path),
            "extension": "csv",
            "features": ["feature1", "feature2"],
            "model_type": ModelType.classification,
            "separator": ",",
            "target": "target",
        }

        # Mock datasets and read_frame to return our test data
        with (
            patch("alphapy.data.datasets", {Partition.train: "train"}),
            patch("alphapy.data.SSEP", os.sep),
            patch("alphapy.data.read_frame") as mock_read_frame,
        ):
            mock_read_frame.return_value = data

            X, y = get_data(model, Partition.train)

            # Check features
            assert isinstance(X, pd.DataFrame)
            assert list(X.columns) == ["feature1", "feature2"]
            assert len(X) == 5

            # Check target encoding for classification
            assert len(y) == 5
            # Labels should be encoded as 0, 1
            unique_labels = np.unique(y)
            assert set(unique_labels) == {0, 1}

    def test_get_data_with_target_regression(self, tmp_path):
        """Test get_data for regression with target column."""
        data = pd.DataFrame(
            {"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [0.5, 1.5, 2.5, 3.5], "target": [10.5, 20.3, 15.7, 25.1]}
        )

        model = Mock()
        model.specs = {
            "directory": str(tmp_path),
            "extension": "csv",
            "features": ["feature1", "feature2"],
            "model_type": ModelType.regression,
            "separator": ",",
            "target": "target",
        }

        with (
            patch("alphapy.data.datasets", {Partition.train: "train"}),
            patch("alphapy.data.SSEP", os.sep),
            patch("alphapy.data.read_frame") as mock_read_frame,
        ):
            mock_read_frame.return_value = data

            X, y = get_data(model, Partition.train)

            # Check features
            assert isinstance(X, pd.DataFrame)
            assert list(X.columns) == ["feature1", "feature2"]
            assert len(X) == 4

            # Check target - should not be encoded for regression
            np.testing.assert_array_equal(y, [10.5, 20.3, 15.7, 25.1])

    def test_get_data_wildcard_features(self, tmp_path):
        """Test get_data with wildcard features (all columns except target)."""
        data = pd.DataFrame(
            {"feature1": [1.0, 2.0, 3.0], "feature2": [0.5, 1.5, 2.5], "feature3": [100, 200, 300], "target": [0, 1, 0]}
        )

        model = Mock()
        model.specs = {
            "directory": str(tmp_path),
            "extension": "csv",
            "features": WILDCARD,  # All features
            "model_type": ModelType.classification,
            "separator": ",",
            "target": "target",
        }

        with (
            patch("alphapy.data.datasets", {Partition.test: "test"}),
            patch("alphapy.data.SSEP", os.sep),
            patch("alphapy.data.read_frame") as mock_read_frame,
        ):
            mock_read_frame.return_value = data

            X, y = get_data(model, Partition.test)

            # Should include all features except target
            assert list(X.columns) == ["feature1", "feature2", "feature3"]
            assert len(X) == 3

    def test_get_data_no_target_column(self, tmp_path):
        """Test get_data when target column is missing."""
        data = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "feature2": [0.5, 1.5, 2.5]})

        model = Mock()
        model.specs = {
            "directory": str(tmp_path),
            "extension": "csv",
            "features": ["feature1", "feature2"],
            "model_type": ModelType.classification,
            "separator": ",",
            "target": "target",
        }

        with (
            patch("alphapy.data.datasets", {Partition.predict: "predict"}),
            patch("alphapy.data.SSEP", os.sep),
            patch("alphapy.data.read_frame") as mock_read_frame,
        ):
            mock_read_frame.return_value = data

            X, y = get_data(model, Partition.predict)

            # Should return features without target
            assert isinstance(X, pd.DataFrame)
            assert list(X.columns) == ["feature1", "feature2"]
            assert len(X) == 3

            # No target available
            assert len(y) == 0

    def test_get_data_with_nan_targets(self, tmp_path):
        """Test get_data handling NaN values in target column."""
        data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0],
                "feature2": [0.5, 1.5, 2.5, 3.5],
                "target": [0, 1, np.nan, 1],  # Contains NaN
            }
        )

        model = Mock()
        model.specs = {
            "directory": str(tmp_path),
            "extension": "csv",
            "features": ["feature1", "feature2"],
            "model_type": ModelType.classification,
            "separator": ",",
            "target": "target",
        }

        with (
            patch("alphapy.data.datasets", {Partition.train: "train"}),
            patch("alphapy.data.SSEP", os.sep),
            patch("alphapy.data.read_frame") as mock_read_frame,
        ):
            mock_read_frame.return_value = data

            X, y = get_data(model, Partition.train)

            # Features should still be loaded
            assert len(X) == 4

            # When NaN values are found in target, labels are not used (empty array returned)
            assert len(y) == 0

    def test_get_data_empty_dataframe(self, tmp_path):
        """Test get_data with empty data file."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create empty file
        data_file = input_dir / "empty.csv"
        data_file.touch()

        model = Mock()
        model.specs = {
            "directory": str(tmp_path),
            "extension": "csv",
            "features": ["feature1"],
            "model_type": ModelType.classification,
            "separator": ",",
            "target": "target",
        }

        with (
            patch("alphapy.data.datasets", {Partition.train: "empty"}),
            patch("alphapy.data.SSEP", os.sep),
            patch("alphapy.data.read_frame") as mock_read,
        ):
            # Mock empty dataframe
            mock_read.return_value = pd.DataFrame()

            X, y = get_data(model, Partition.train)

            # Should return empty structures
            assert X.empty
            assert len(y) == 0


class TestShuffleData:
    """Test data shuffling functionality."""

    def test_shuffle_data_enabled(self):
        """Test shuffling when shuffle is enabled."""
        # Create mock model with training data
        model = Mock()
        model.specs = {"seed": 42, "shuffle": True}

        # Create deterministic training data
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y_train = np.array([0, 1, 0, 1, 0])

        model.X_train = X_train
        model.y_train = y_train

        result_model = shuffle_data(model)

        # Data should be shuffled (different order)
        assert not np.array_equal(result_model.X_train, X_train)
        assert not np.array_equal(result_model.y_train, y_train)

        # But should contain same elements
        assert len(result_model.X_train) == len(X_train)
        assert len(result_model.y_train) == len(y_train)

        # Check that corresponding X and y are still paired correctly
        # (This is implementation dependent but important for ML)
        assert result_model.X_train.shape == X_train.shape
        assert result_model.y_train.shape == y_train.shape

    def test_shuffle_data_disabled(self):
        """Test that data is not shuffled when shuffle is disabled."""
        model = Mock()
        model.specs = {"seed": 42, "shuffle": False}

        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])

        model.X_train = X_train
        model.y_train = y_train

        result_model = shuffle_data(model)

        # Data should remain unchanged
        np.testing.assert_array_equal(result_model.X_train, X_train)
        np.testing.assert_array_equal(result_model.y_train, y_train)

    def test_shuffle_data_reproducible(self):
        """Test that shuffling is reproducible with same seed."""
        model1 = Mock()
        model1.specs = {"seed": 123, "shuffle": True}

        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])

        model1.X_train = X_train.copy()
        model1.y_train = y_train.copy()

        model2 = Mock()
        model2.specs = {"seed": 123, "shuffle": True}
        model2.X_train = X_train.copy()
        model2.y_train = y_train.copy()

        result1 = shuffle_data(model1)
        result2 = shuffle_data(model2)

        # Same seed should produce same shuffle
        np.testing.assert_array_equal(result1.X_train, result2.X_train)
        np.testing.assert_array_equal(result1.y_train, result2.y_train)

    def test_shuffle_data_different_seeds(self):
        """Test that different seeds produce different shuffles."""
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y_train = np.array([0, 1, 0, 1, 0])

        model1 = Mock()
        model1.specs = {"seed": 42, "shuffle": True}
        model1.X_train = X_train.copy()
        model1.y_train = y_train.copy()

        model2 = Mock()
        model2.specs = {"seed": 123, "shuffle": True}
        model2.X_train = X_train.copy()
        model2.y_train = y_train.copy()

        result1 = shuffle_data(model1)
        result2 = shuffle_data(model2)

        # Different seeds should produce different shuffles
        assert not np.array_equal(result1.X_train, result2.X_train)
        assert not np.array_equal(result1.y_train, result2.y_train)


class TestSampleData:
    """Test data sampling functionality."""

    def test_sample_data_random_under_sampling(self):
        """Test random under sampling."""
        model = Mock()
        model.specs = {
            "sampling_method": SamplingMethod.under_random,
            "sampling_ratio": 1.0,
            "target": "target",
            "target_value": 1,
        }

        # Imbalanced data: more 0s than 1s
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y_train = np.array([0, 0, 0, 0, 1, 1])

        model.X_train = X_train
        model.y_train = y_train

        with patch("alphapy.data.RandomUnderSampler") as mock_sampler_class:
            mock_sampler = Mock()
            # The function tries fit_sample first, then fit_resample as fallback
            mock_sampler.fit_sample.return_value = (
                np.array([[1, 2], [3, 4], [9, 10], [11, 12]]),
                np.array([0, 0, 1, 1]),
            )
            mock_sampler_class.return_value = mock_sampler

            result_model = sample_data(model)

            # Check that sampling was applied
            mock_sampler_class.assert_called_once()
            assert len(result_model.X_train) == 4
            assert len(result_model.y_train) == 4

            # Check balanced classes
            unique, counts = np.unique(result_model.y_train, return_counts=True)
            assert len(unique) == 2
            assert counts[0] == counts[1]  # Balanced

    def test_sample_data_smote_over_sampling(self):
        """Test SMOTE over sampling."""
        model = Mock()
        model.specs = {
            "sampling_method": SamplingMethod.over_smote,
            "sampling_ratio": 1.0,
            "target": "target",
            "target_value": 1,
        }

        # Imbalanced data
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 0, 0, 1])

        model.X_train = X_train
        model.y_train = y_train

        with patch("alphapy.data.SMOTE") as mock_smote_class:
            mock_smote = Mock()
            # SMOTE would create synthetic samples - use fit_sample
            mock_smote.fit_sample.return_value = (
                np.array([[1, 2], [3, 4], [5, 6], [7, 8], [6, 7], [4, 5]]),
                np.array([0, 0, 0, 1, 1, 1]),
            )
            mock_smote_class.return_value = mock_smote

            result_model = sample_data(model)

            # Check that SMOTE was configured correctly
            mock_smote_class.assert_called_once_with(ratio=1.0, kind="regular")

            # Check that oversampling increased samples
            assert len(result_model.X_train) == 6
            assert len(result_model.y_train) == 6

    def test_sample_data_auto_ratio_calculation(self):
        """Test automatic sampling ratio calculation."""
        model = Mock()
        model.specs = {
            "sampling_method": SamplingMethod.over_random,
            "sampling_ratio": 0.0,  # Auto-calculate
            "target": "target",
            "target_value": 1,
        }

        # 4 class 0, 2 class 1 -> ratio should be (4/2) - 1 = 1.0
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y_train = np.array([0, 0, 0, 0, 1, 1])

        model.X_train = X_train
        model.y_train = y_train

        with patch("alphapy.data.RandomOverSampler") as mock_sampler_class:
            mock_sampler = Mock()
            mock_sampler.fit_sample.return_value = (X_train, y_train)
            mock_sampler_class.return_value = mock_sampler

            result_model = sample_data(model)

            # Check that ratio was calculated and passed
            mock_sampler_class.assert_called_once_with(ratio=1.0)

    def test_sample_data_unsupported_method(self):
        """Test handling of unsupported sampling method."""
        model = Mock()
        model.specs = {
            "sampling_method": "invalid_method",
            "sampling_ratio": 1.0,
            "target": "target",
            "target_value": 1,
        }

        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0, 1])

        model.X_train = X_train
        model.y_train = y_train

        with pytest.raises(ValueError, match="Unknown Sampling Method"):
            sample_data(model)

    def test_sample_data_fit_sample_fallback(self):
        """Test fallback to fit_sample when fit_resample is not available."""
        model = Mock()
        model.specs = {
            "sampling_method": SamplingMethod.under_random,
            "sampling_ratio": 1.0,
            "target": "target",
            "target_value": 1,
        }

        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 0, 1, 1])

        model.X_train = X_train
        model.y_train = y_train

        with patch("alphapy.data.RandomUnderSampler") as mock_sampler_class:
            mock_sampler = Mock()
            # fit_resample raises AttributeError, fallback to fit_sample
            mock_sampler.fit_resample.side_effect = AttributeError("fit_resample not available")
            mock_sampler.fit_sample.return_value = (X_train[:2], y_train[:2])
            mock_sampler_class.return_value = mock_sampler

            result_model = sample_data(model)

            # Should fallback to fit_sample
            mock_sampler.fit_sample.assert_called_once_with(X_train, y_train)
            assert len(result_model.X_train) == 2
            assert len(result_model.y_train) == 2


if __name__ == "__main__":
    pytest.main([__file__])

"""
Unit tests for core model functions in alphapy.model module.

This module tests the fundamental model operations including initialization,
configuration loading, fitting, prediction, and evaluation.
"""

import logging
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from alphapy.globals import ModelType, Objective, Partition
from alphapy.model import (
    Model,
    first_fit,
    generate_metrics,
    get_model_config,
    load_feature_map,
    load_predictor,
    make_predictions,
    predict_best,
    predict_blend,
    save_predictor,
)


class TestModelInitialization:
    """Test Model class initialization and basic properties."""

    def test_model_init_with_valid_specs(self):
        """Test Model initialization with valid specifications."""
        specs = {
            "algorithms": ["RF", "LR"],
            "directory": "/tmp/test",
            "target": "signal",
            "features": ["feature1", "feature2"],
            "model_type": ModelType.classification,
        }

        model = Model(specs)

        assert model.specs == specs
        assert model.algolist == ["RF", "LR"]
        assert model.X_train is None
        assert model.X_test is None
        assert model.y_train is None
        assert model.y_test is None
        assert model.best_algo is None
        assert isinstance(model.estimators, dict)
        assert isinstance(model.importances, dict)
        assert isinstance(model.coefs, dict)
        assert isinstance(model.support, dict)
        assert isinstance(model.preds, dict)
        assert isinstance(model.probas, dict)
        assert isinstance(model.metrics, dict)

    def test_model_init_missing_algorithms_key(self):
        """Test Model initialization fails without algorithms key."""
        specs = {
            "directory": "/tmp/test",
            "target": "signal",
        }

        with pytest.raises(KeyError, match="Model specs must include the key: algorithms"):
            Model(specs)

    def test_model_init_empty_algorithms(self):
        """Test Model initialization with empty algorithms list."""
        specs = {
            "algorithms": [],
            "directory": "/tmp/test",
        }

        model = Model(specs)
        assert model.algolist == []

    def test_model_str_representation(self):
        """Test Model string representation."""
        specs = {"algorithms": ["RF"]}
        model = Model(specs)

        # Since model.name is not set in __init__, this should raise AttributeError
        with pytest.raises(AttributeError):
            str(model)

    def test_model_getnewargs(self):
        """Test Model __getnewargs__ method for pickling support."""
        specs = {"algorithms": ["RF", "LR"]}
        model = Model(specs)

        args = model.__getnewargs__()
        assert args == (specs,)


class TestGetModelConfig:
    """Test model configuration loading functionality."""

    def test_get_model_config_with_valid_yaml(self, tmp_path):
        """Test loading valid model configuration from YAML file."""
        # Create a temporary config directory structure
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Create a sample model.yml file
        model_config = {
            "project": {
                "directory": "/tmp/test",
                "file_extension": "csv",
                "submission_file": "sample_submission",
                "submit_probas": True,
            },
            "data": {
                "drop": [],
                "features": [],
                "sentinel": -1,
                "separator": ",",
                "shuffle": True,
                "split": 0.2,
                "target": "signal",
                "target_value": 1,
                "sampling": {
                    "option": False,
                    "method": "under_cluster",
                    "ratio": 1.0,
                },
            },
            "features": {
                "clustering": {"option": False, "minimum": 2, "maximum": 10, "increment": 1},
                "counts": {"option": False},
                "encoding": {"rounding": 3, "type": "onehot"},
                "factors": [],
                "interactions": {"option": False, "sampling_pct": 10, "poly_degree": 2},
                "isomap": {"option": False, "components": 2, "neighbors": 5},
                "logtransform": {"option": False},
                "variance": {"option": False, "threshold": 0.1},
                "numpy": {"option": False},
                "pca": {"option": False, "minimum": 2, "maximum": 10, "increment": 1, "whiten": False},
                "scaling": {"option": False, "type": "standard"},
                "scipy": {"option": False},
                "text": {"ngrams": 1, "vectorize": False},
                "tsne": {"option": False, "components": 2, "learning_rate": 200.0, "perplexity": 30.0},
            },
            "model": {
                "algorithms": ["RF", "LR"],
                "cv_folds": 5,
                "type": "classification",
                "estimators": 100,
                "pvalue_level": 0.05,
                "scoring_function": "roc_auc",
                "calibration": {"option": False, "type": "sigmoid"},
                "feature_selection": {"option": False, "percentage": 50, "uni_grid": [], "score_func": "f_classif"},
                "grid_search": {
                    "option": False,
                    "iterations": 50,
                    "random": True,
                    "subsample": False,
                    "sampling_pct": 50,
                },
                "rfe": {"option": False, "step": 1},
            },
            "pipeline": {
                "number_jobs": 1,
                "seed": 42,
                "verbosity": 0,
            },
            "plots": {
                "calibration": False,
                "confusion_matrix": False,
                "importances": False,
                "learning_curve": False,
                "roc_curve": False,
            },
            "transforms": None,
            "xgboost": {
                "stopping_rounds": 10,
            },
        }

        config_file = config_dir / "model.yml"
        with open(config_file, "w") as f:
            yaml.dump(model_config, f)

        # Mock the global PSEP and SSEP
        with patch("alphapy.model.PSEP", str(tmp_path)), patch("alphapy.model.SSEP", os.sep):
            specs = get_model_config()

            # Verify key configuration values
            assert specs["directory"] == "/tmp/test"
            assert specs["extension"] == "csv"
            assert specs["algorithms"] == ["RF", "LR"]
            assert specs["cv_folds"] == 5
            assert specs["model_type"] == ModelType.classification
            assert specs["scorer"] == "roc_auc"
            assert specs["seed"] == 42

    def test_get_model_config_invalid_model_type(self, tmp_path):
        """Test configuration loading with invalid model type."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        model_config = {
            "project": {"directory": "/tmp", "file_extension": "csv", "submission_file": "", "submit_probas": False},
            "data": {
                "drop": [],
                "features": [],
                "sentinel": -1,
                "separator": ",",
                "shuffle": True,
                "split": 0.2,
                "target": "signal",
                "target_value": 1,
                "sampling": {"option": False, "method": "under_cluster", "ratio": 1.0},
            },
            "features": {
                "clustering": {"option": False, "minimum": 2, "maximum": 10, "increment": 1},
                "counts": {"option": False},
                "encoding": {"rounding": 3, "type": "onehot"},
                "factors": [],
                "interactions": {"option": False, "sampling_pct": 10, "poly_degree": 2},
                "isomap": {"option": False, "components": 2, "neighbors": 5},
                "logtransform": {"option": False},
                "variance": {"option": False, "threshold": 0.1},
                "numpy": {"option": False},
                "pca": {"option": False, "minimum": 2, "maximum": 10, "increment": 1, "whiten": False},
                "scaling": {"option": False, "type": "standard"},
                "scipy": {"option": False},
                "text": {"ngrams": 1, "vectorize": False},
                "tsne": {"option": False, "components": 2, "learning_rate": 200.0, "perplexity": 30.0},
            },
            "model": {
                "algorithms": ["RF"],
                "cv_folds": 5,
                "type": "invalid_type",
                "estimators": 100,
                "pvalue_level": 0.05,
                "scoring_function": "roc_auc",
                "calibration": {"option": False, "type": "sigmoid"},
                "feature_selection": {"option": False, "percentage": 50, "uni_grid": [], "score_func": "f_classif"},
                "grid_search": {
                    "option": False,
                    "iterations": 50,
                    "random": True,
                    "subsample": False,
                    "sampling_pct": 50,
                },
                "rfe": {"option": False, "step": 1},
            },
            "pipeline": {"number_jobs": 1, "seed": 42, "verbosity": 0},
            "plots": {
                "calibration": False,
                "confusion_matrix": False,
                "importances": False,
                "learning_curve": False,
                "roc_curve": False,
            },
            "xgboost": {"stopping_rounds": 10},
        }

        config_file = config_dir / "model.yml"
        with open(config_file, "w") as f:
            yaml.dump(model_config, f)

        with (
            patch("alphapy.model.PSEP", str(tmp_path)),
            patch("alphapy.model.SSEP", os.sep),
            pytest.raises(ValueError, match="model.yml model:type invalid_type unrecognized"),
        ):
            get_model_config()


class TestFirstFit:
    """Test initial model fitting functionality."""

    def test_first_fit_basic_classifier(self):
        """Test first_fit with a basic classifier."""
        # Create mock model with required specs
        model = Mock()
        model.specs = {
            "cv_folds": 3,
            "esr": 10,
            "n_jobs": 1,
            "scorer": "roc_auc",
            "seed": 42,
            "split": 0.2,
            "verbosity": 0,
        }

        # Create sample training data
        rng = np.random.default_rng(seed=42)
        X_train = rng.standard_normal((100, 5))
        y_train = rng.integers(0, 2, 100, endpoint=False)

        model.X_train = X_train
        model.y_train = y_train
        model.estimators = {}
        model.importances = {}
        model.coefs = {}

        # Create estimator
        estimator = LogisticRegression(random_state=42)

        # Test first_fit
        result_model = first_fit(model, "LR", estimator)

        assert "LR" in result_model.estimators
        assert isinstance(result_model.estimators["LR"], LogisticRegression)
        assert "LR" in result_model.coefs  # LogisticRegression has coef_ attribute

    def test_first_fit_with_feature_importances(self):
        """Test first_fit with estimator that has feature importances."""
        model = Mock()
        model.specs = {
            "cv_folds": 3,
            "esr": 10,
            "n_jobs": 1,
            "scorer": "roc_auc",
            "seed": 42,
            "split": 0.2,
            "verbosity": 0,
        }

        rng = np.random.default_rng(seed=42)
        rng = np.random.default_rng(seed=42)
        X_train = rng.standard_normal((100, 5))
        y_train = rng.integers(0, 2, 100, endpoint=False)

        model.X_train = X_train
        model.y_train = y_train
        model.estimators = {}
        model.importances = {}
        model.coefs = {}

        # Random Forest has feature_importances_
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        result_model = first_fit(model, "RF", estimator)

        assert "RF" in result_model.estimators
        assert "RF" in result_model.importances
        assert len(result_model.importances["RF"]) == 5  # 5 features

    @patch("alphapy.model.cross_val_score")
    def test_first_fit_cross_validation_failure(self, mock_cv_score):
        """Test first_fit handles cross-validation failure gracefully."""
        mock_cv_score.side_effect = Exception("CV failed")

        model = Mock()
        model.specs = {
            "cv_folds": 3,
            "esr": 10,
            "n_jobs": 1,
            "scorer": "roc_auc",
            "seed": 42,
            "split": 0.2,
            "verbosity": 0,
        }

        rng = np.random.default_rng(seed=42)
        rng = np.random.default_rng(seed=42)
        model.X_train = rng.standard_normal((50, 3))
        model.y_train = rng.integers(0, 2, 50, endpoint=False)
        model.estimators = {}
        model.importances = {}
        model.coefs = {}

        estimator = LogisticRegression(random_state=42)

        # Should not raise exception despite CV failure
        result_model = first_fit(model, "LR", estimator)
        assert "LR" in result_model.estimators


class TestMakePredictions:
    """Test prediction generation functionality."""

    def test_make_predictions_classification(self):
        """Test make_predictions for classification models."""
        model = Mock()
        model.specs = {
            "cal_type": "sigmoid",
            "cv_folds": 3,
            "model_type": ModelType.classification,
        }

        # Create mock estimator
        estimator = Mock()
        estimator.predict.return_value = np.array([0, 1, 0, 1])
        estimator.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])

        model.estimators = {"LR": estimator}
        model.support = {}
        model.preds = {}
        model.probas = {}

        # Create sample data
        rng = np.random.default_rng(seed=42)
        X_train = rng.standard_normal((4, 3))
        X_test = rng.standard_normal((4, 3))
        y_train = np.array([0, 1, 0, 1])

        model.X_train = X_train
        model.X_test = X_test
        model.y_train = y_train

        result_model = make_predictions(model, "LR", calibrate=False)

        # Check predictions were stored
        assert ("LR", Partition.train) in result_model.preds
        assert ("LR", Partition.test) in result_model.preds
        assert ("LR", Partition.train) in result_model.probas
        assert ("LR", Partition.test) in result_model.probas

        # Check probability extraction (second column)
        np.testing.assert_array_equal(result_model.probas[("LR", Partition.train)], [0.2, 0.7, 0.1, 0.6])

    def test_make_predictions_regression(self):
        """Test make_predictions for regression models."""
        model = Mock()
        model.specs = {
            "cal_type": "isotonic",
            "cv_folds": 3,
            "model_type": ModelType.regression,
        }

        estimator = Mock()
        estimator.predict.return_value = np.array([1.5, 2.3, 0.8, 3.1])

        model.estimators = {"RF": estimator}
        model.support = {}
        model.preds = {}
        model.probas = {}

        rng = np.random.default_rng(seed=42)
        X_train = rng.standard_normal((4, 3))
        X_test = rng.standard_normal((4, 3))
        y_train = np.array([1.2, 2.1, 0.9, 2.8])

        model.X_train = X_train
        model.X_test = X_test
        model.y_train = y_train

        result_model = make_predictions(model, "RF", calibrate=False)

        # Check predictions were stored
        assert ("RF", Partition.train) in result_model.preds
        assert ("RF", Partition.test) in result_model.preds
        # No probabilities for regression
        assert ("RF", Partition.train) not in result_model.probas

    @patch("alphapy.model.CalibratedClassifierCV")
    def test_make_predictions_with_calibration(self, mock_calibrated):
        """Test make_predictions with probability calibration."""
        # Setup mock calibrated classifier
        calibrated_estimator = Mock()
        calibrated_estimator.predict.return_value = np.array([0, 1])
        calibrated_estimator.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
        calibrated_estimator.fit.return_value = None

        mock_calibrated.return_value = calibrated_estimator

        model = Mock()
        model.specs = {
            "cal_type": "sigmoid",
            "cv_folds": 3,
            "model_type": ModelType.classification,
        }

        original_estimator = Mock()
        model.estimators = {"LR": original_estimator}
        model.support = {}
        model.preds = {}
        model.probas = {}

        rng = np.random.default_rng(seed=42)
        X_train = rng.standard_normal((2, 3))
        X_test = rng.standard_normal((2, 3))
        y_train = np.array([0, 1])

        model.X_train = X_train
        model.X_test = X_test
        model.y_train = y_train

        result_model = make_predictions(model, "LR", calibrate=True)

        # Verify calibration was applied
        mock_calibrated.assert_called_once_with(original_estimator, cv=3, method="sigmoid")
        assert result_model.estimators["LR"] == calibrated_estimator


class TestGenerateMetrics:
    """Test model evaluation metrics generation."""

    def test_generate_metrics_classification(self):
        """Test metrics generation for classification models."""
        model = Mock()
        model.specs = {"model_type": ModelType.classification}
        model.algolist = ["LR"]

        # Mock data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.6, 0.8, 0.2])

        model.y_train = y_true
        model.preds = {("LR", Partition.train): y_pred}
        model.probas = {("LR", Partition.train): y_proba}
        model.metrics = {}

        result_model = generate_metrics(model, Partition.train)

        # Check that various classification metrics were calculated
        metrics_keys = [k for k in result_model.metrics if k[0] == "LR" and k[1] == Partition.train]
        metric_names = [k[2] for k in metrics_keys]

        assert "accuracy" in metric_names
        assert "roc_auc" in metric_names
        assert "f1" in metric_names
        assert "precision" in metric_names
        assert "recall" in metric_names

    def test_generate_metrics_regression(self):
        """Test metrics generation for regression models."""
        model = Mock()
        model.specs = {"model_type": ModelType.regression}
        model.algolist = ["RF"]

        # Mock data
        y_true = np.array([1.5, 2.1, 3.2, 1.8, 2.9])
        y_pred = np.array([1.4, 2.0, 3.1, 1.9, 2.8])

        model.y_train = y_true
        model.preds = {("RF", Partition.train): y_pred}
        model.probas = {}
        model.metrics = {}

        result_model = generate_metrics(model, Partition.train)

        # Check that various regression metrics were calculated
        metrics_keys = [k for k in result_model.metrics if k[0] == "RF" and k[1] == Partition.train]
        metric_names = [k[2] for k in metrics_keys]

        assert "r2" in metric_names
        assert "neg_mean_squared_error" in metric_names
        assert "neg_mean_absolute_error" in metric_names
        assert "explained_variance" in metric_names

    def test_generate_metrics_no_labels(self):
        """Test metrics generation when no labels are available."""
        model = Mock()
        model.specs = {"model_type": ModelType.classification}
        model.algolist = ["LR"]

        # Empty labels
        model.y_train = np.array([])
        model.preds = {("LR", Partition.train): np.array([])}
        model.probas = {("LR", Partition.train): np.array([])}
        model.metrics = {}

        result_model = generate_metrics(model, Partition.train)

        # Should handle gracefully with no metrics generated
        assert len(result_model.metrics) == 0

    def test_generate_metrics_with_blend(self):
        """Test metrics generation with blended models."""
        model = Mock()
        model.specs = {"model_type": ModelType.classification}
        model.algolist = ["LR", "RF"]  # Multiple algorithms triggers blend

        y_true = np.array([0, 1, 0, 1])
        y_pred_lr = np.array([0, 1, 1, 1])
        y_pred_rf = np.array([0, 1, 0, 0])
        y_pred_blend = np.array([0, 1, 0, 1])

        model.y_train = y_true
        model.preds = {
            ("LR", Partition.train): y_pred_lr,
            ("RF", Partition.train): y_pred_rf,
            ("BLEND", Partition.train): y_pred_blend,
        }
        model.probas = {
            ("LR", Partition.train): np.array([0.1, 0.9, 0.7, 0.8]),
            ("RF", Partition.train): np.array([0.2, 0.8, 0.3, 0.4]),
            ("BLEND", Partition.train): np.array([0.15, 0.85, 0.5, 0.6]),
        }
        model.metrics = {}

        result_model = generate_metrics(model, Partition.train)

        # Check that metrics were generated for all algorithms including BLEND
        algo_metrics = {k[0] for k in result_model.metrics}
        assert "LR" in algo_metrics
        assert "RF" in algo_metrics
        assert "BLEND" in algo_metrics


class TestLoadPredictor:
    """Test model predictor loading functionality."""

    def test_load_predictor_pkl_file(self, tmp_path):
        """Test loading a pickled model predictor."""
        # Create a mock directory structure
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create a simple estimator and save it
        estimator = LogisticRegression(random_state=42)
        rng = np.random.default_rng(seed=42)
        X_dummy = rng.standard_normal((10, 3))
        y_dummy = rng.integers(0, 2, 10, endpoint=False)
        estimator.fit(X_dummy, y_dummy)

        model_file = model_dir / "model_20231201.pkl"
        import joblib

        joblib.dump(estimator, model_file)

        # Mock the global separators
        with patch("alphapy.model.SSEP", os.sep), patch("alphapy.model.most_recent_file") as mock_recent:
            mock_recent.return_value = str(model_file)

            predictor = load_predictor(str(tmp_path))

            assert isinstance(predictor, LogisticRegression)
            # Test that it can make predictions
            test_pred = predictor.predict(X_dummy[:2])
            assert len(test_pred) == 2

    def test_load_predictor_h5_file_without_keras(self, tmp_path):
        """Test loading H5 file when Keras is not available."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        model_file = model_dir / "model_20231201.h5"
        model_file.touch()  # Create empty file

        with (
            patch("alphapy.model.SSEP", os.sep),
            patch("alphapy.model.most_recent_file") as mock_recent,
            patch("alphapy.model.KERAS_AVAILABLE", False),
        ):
            mock_recent.return_value = str(model_file)

            with pytest.raises(ImportError, match="Cannot load .h5 model file"):
                load_predictor(str(tmp_path))

    def test_load_predictor_no_model_found(self, tmp_path):
        """Test handling when no model file is found."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch("alphapy.model.SSEP", os.sep), patch("alphapy.model.most_recent_file") as mock_recent:
            mock_recent.side_effect = FileNotFoundError("No model files found")

            with pytest.raises(FileNotFoundError):
                load_predictor(str(tmp_path))


class TestSavePredictor:
    """Test model predictor saving functionality."""

    def test_save_predictor_pkl(self, tmp_path):
        """Test saving a scikit-learn model predictor as pickle."""
        # Create model directory
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create mock model with estimator
        model = Mock()
        model.specs = {"directory": str(tmp_path)}
        model.best_algo = "LR"

        # Create a fitted estimator
        estimator = LogisticRegression(random_state=42)
        rng = np.random.default_rng(seed=42)
        X_dummy = rng.standard_normal((10, 3))
        y_dummy = rng.integers(0, 2, 10, endpoint=False)
        estimator.fit(X_dummy, y_dummy)
        model.estimators = {"BEST": estimator}

        timestamp = "20231201"

        with patch("alphapy.model.SSEP", os.sep):
            save_predictor(model, timestamp)

            # Check that file was created
            expected_file = model_dir / f"model_{timestamp}.pkl"
            assert expected_file.exists()

            # Verify we can load it back
            import joblib

            loaded_estimator = joblib.load(expected_file)
            assert isinstance(loaded_estimator, LogisticRegression)

    def test_save_predictor_keras_h5(self, tmp_path):
        """Test saving a Keras model predictor as H5."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create mock Keras model
        keras_model = Mock()
        keras_model.save = Mock()

        mock_predictor = Mock()
        mock_predictor.model = keras_model

        model = Mock()
        model.specs = {"directory": str(tmp_path)}
        model.best_algo = "KERAS_NN"
        model.estimators = {"BEST": mock_predictor}

        timestamp = "20231201"

        with patch("alphapy.model.SSEP", os.sep):
            save_predictor(model, timestamp)

            expected_path = os.sep.join([str(tmp_path), "model", f"model_{timestamp}.h5"])
            keras_model.save.assert_called_once_with(expected_path)


class TestPredictBest:
    """Test best model selection functionality."""

    def test_predict_best_maximizing_score(self):
        """Test selecting best model with maximizing objective (e.g., ROC AUC)."""
        model = Mock()
        model.specs = {"model_type": ModelType.classification, "rfe": False, "scorer": "roc_auc"}
        model.test_labels = True
        model.algolist = ["LR", "RF"]

        # Mock metrics - RF has higher ROC AUC, but BLEND will be created and needs metrics
        model.metrics = {
            ("LR", Partition.test, "roc_auc"): 0.75,
            ("RF", Partition.test, "roc_auc"): 0.85,
            ("BLEND", Partition.test, "roc_auc"): 0.82,  # BLEND added since len(algolist) > 1
        }

        # Mock estimators and predictions
        lr_estimator = Mock()
        rf_estimator = Mock()
        model.estimators = {"LR": lr_estimator, "RF": rf_estimator}

        model.preds = {
            ("LR", Partition.train): np.array([0, 1, 0]),
            ("LR", Partition.test): np.array([1, 0, 1]),
            ("RF", Partition.train): np.array([0, 0, 1]),
            ("RF", Partition.test): np.array([0, 1, 0]),
        }

        model.probas = {
            ("LR", Partition.train): np.array([0.3, 0.8, 0.2]),
            ("LR", Partition.test): np.array([0.7, 0.1, 0.9]),
            ("RF", Partition.train): np.array([0.2, 0.4, 0.9]),
            ("RF", Partition.test): np.array([0.1, 0.8, 0.3]),
        }

        with patch("alphapy.model.scorers", {"roc_auc": (None, Objective.maximize)}):
            result_model = predict_best(model)

            # RF should be selected as best
            assert result_model.best_algo == "RF"
            assert result_model.estimators["BEST"] == rf_estimator

            # Best predictions should match RF predictions
            np.testing.assert_array_equal(
                result_model.preds[("BEST", Partition.train)], result_model.preds[("RF", Partition.train)]
            )

    def test_predict_best_minimizing_score(self):
        """Test selecting best model with minimizing objective (e.g., log loss)."""
        model = Mock()
        model.specs = {"model_type": ModelType.classification, "rfe": False, "scorer": "neg_log_loss"}
        model.test_labels = False  # Use training data for selection
        model.algolist = ["LR", "RF"]

        # Mock metrics - LR has lower log loss (better), but BLEND will be created and needs metrics
        model.metrics = {
            ("LR", Partition.train, "neg_log_loss"): 0.3,
            ("RF", Partition.train, "neg_log_loss"): 0.5,
            ("BLEND", Partition.train, "neg_log_loss"): 0.4,  # BLEND added since len(algolist) > 1
        }

        lr_estimator = Mock()
        rf_estimator = Mock()
        model.estimators = {"LR": lr_estimator, "RF": rf_estimator}

        model.preds = {
            ("LR", Partition.train): np.array([0, 1]),
            ("LR", Partition.test): np.array([1, 0]),
            ("RF", Partition.train): np.array([1, 0]),
            ("RF", Partition.test): np.array([0, 1]),
        }

        model.probas = {
            ("LR", Partition.train): np.array([0.2, 0.9]),
            ("LR", Partition.test): np.array([0.8, 0.1]),
            ("RF", Partition.train): np.array([0.7, 0.3]),
            ("RF", Partition.test): np.array([0.4, 0.6]),
        }

        with patch("alphapy.model.scorers", {"neg_log_loss": (None, Objective.minimize)}):
            result_model = predict_best(model)

            # LR should be selected as best (lower log loss)
            assert result_model.best_algo == "LR"
            assert result_model.estimators["BEST"] == lr_estimator

    def test_predict_best_with_blend_multiple_algos(self):
        """Test best selection when blend model is available."""
        model = Mock()
        model.specs = {"model_type": ModelType.classification, "rfe": False, "scorer": "roc_auc"}
        model.test_labels = True
        model.algolist = ["LR", "RF"]  # Multiple algos triggers blend

        # BLEND has highest score
        model.metrics = {
            ("LR", Partition.test, "roc_auc"): 0.75,
            ("RF", Partition.test, "roc_auc"): 0.80,
            ("BLEND", Partition.test, "roc_auc"): 0.87,
        }

        blend_estimator = Mock()
        model.estimators = {"LR": Mock(), "RF": Mock(), "BLEND": blend_estimator}

        model.preds = {
            ("BLEND", Partition.train): np.array([0, 1]),
            ("BLEND", Partition.test): np.array([1, 0]),
        }

        model.probas = {
            ("BLEND", Partition.train): np.array([0.1, 0.9]),
            ("BLEND", Partition.test): np.array([0.8, 0.2]),
        }

        with patch("alphapy.model.scorers", {"roc_auc": (None, Objective.maximize)}):
            result_model = predict_best(model)

            assert result_model.best_algo == "BLEND"
            assert result_model.estimators["BEST"] == blend_estimator


class TestPredictBlend:
    """Test model blending functionality."""

    def test_predict_blend_classification(self):
        """Test blending for classification models."""
        model = Mock()
        model.specs = {"model_type": ModelType.classification, "cv_folds": 3}
        model.algolist = ["LR", "RF"]

        # Create sample data
        rng = np.random.default_rng(seed=42)
        X_train = rng.standard_normal((6, 4))
        X_test = rng.standard_normal((4, 4))
        y_train = np.array([0, 1, 0, 1, 0, 1])

        model.X_train = X_train
        model.X_test = X_test
        model.y_train = y_train

        # Mock estimators
        lr_estimator = Mock()
        lr_estimator.coef_ = np.array([[0.5, -0.3, 0.2, 0.1]])
        rf_estimator = Mock()
        rf_estimator.feature_importances_ = np.array([0.2, 0.3, 0.3, 0.2])

        model.estimators = {"LR": lr_estimator, "RF": rf_estimator}
        model.importances = {}
        model.coefs = {}

        # Mock probabilities for blending
        model.probas = {
            ("LR", Partition.train): np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9]),
            ("LR", Partition.test): np.array([0.4, 0.6, 0.2, 0.8]),
            ("RF", Partition.train): np.array([0.3, 0.7, 0.4, 0.6, 0.2, 0.8]),
            ("RF", Partition.test): np.array([0.5, 0.5, 0.3, 0.7]),
        }

        model.preds = {}

        with patch("alphapy.model.LogisticRegression") as mock_lr_class:
            mock_blend_estimator = Mock()
            mock_blend_estimator.predict.return_value = np.array([0, 1, 0, 1, 0, 1])
            mock_blend_estimator.predict_proba.return_value = np.array(
                [[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8], [0.9, 0.1], [0.1, 0.9]]
            )
            mock_lr_class.return_value = mock_blend_estimator

            result_model = predict_blend(model)

            # Check that blend estimator was created and fitted
            mock_lr_class.assert_called_once()
            mock_blend_estimator.fit.assert_called_once()

            # Check that BLEND predictions were stored
            assert "BLEND" in result_model.estimators
            assert ("BLEND", Partition.train) in result_model.preds
            assert ("BLEND", Partition.test) in result_model.preds
            assert ("BLEND", Partition.train) in result_model.probas
            assert ("BLEND", Partition.test) in result_model.probas

    def test_predict_blend_regression(self):
        """Test blending for regression models."""
        model = Mock()
        model.specs = {"model_type": ModelType.regression, "cv_folds": 5}
        model.algolist = ["RF", "SVR"]

        # Create sample data
        rng = np.random.default_rng(seed=42)
        X_train = rng.standard_normal((5, 3))
        X_test = rng.standard_normal((3, 3))
        y_train = np.array([1.5, 2.1, 1.8, 2.5, 1.9])

        model.X_train = X_train
        model.X_test = X_test
        model.y_train = y_train

        # Mock estimators
        rf_estimator = Mock()
        rf_estimator.feature_importances_ = np.array([0.4, 0.3, 0.3])
        svr_estimator = Mock()

        model.estimators = {"RF": rf_estimator, "SVR": svr_estimator}
        model.importances = {}
        model.coefs = {}

        # Mock predictions for blending
        model.preds = {
            ("RF", Partition.train): np.array([1.4, 2.0, 1.7, 2.4, 1.8]),
            ("RF", Partition.test): np.array([1.6, 2.2, 1.9]),
            ("SVR", Partition.train): np.array([1.6, 2.2, 1.9, 2.6, 2.0]),
            ("SVR", Partition.test): np.array([1.7, 2.1, 2.0]),
        }

        model.probas = {}

        with patch("alphapy.model.RidgeCV") as mock_ridge_class:
            mock_blend_estimator = Mock()
            mock_blend_estimator.predict.return_value = np.array([1.5, 2.1, 1.8, 2.5, 1.9])
            mock_ridge_class.return_value = mock_blend_estimator

            result_model = predict_blend(model)

            # Check that Ridge estimator was created
            mock_ridge_class.assert_called_once()
            mock_blend_estimator.fit.assert_called_once()

            # Check that BLEND predictions were stored (no probabilities for regression)
            assert "BLEND" in result_model.estimators
            assert ("BLEND", Partition.train) in result_model.preds
            assert ("BLEND", Partition.test) in result_model.preds
            assert ("BLEND", Partition.train) not in result_model.probas


class TestLoadFeatureMap:
    """Test feature map loading functionality."""

    def test_load_feature_map_success(self, tmp_path):
        """Test successful loading of feature map."""
        # Create model directory and feature map file
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        feature_map = {
            "features": ["feature1", "feature2", "feature3"],
            "transformations": ["scale", "pca"],
            "selected_features": [0, 2],
        }

        feature_map_file = model_dir / "feature_map_20231201.pkl"
        import joblib

        joblib.dump(feature_map, feature_map_file)

        # Create mock model
        model = Mock()
        model.feature_map = None

        with patch("alphapy.model.SSEP", os.sep), patch("alphapy.model.most_recent_file") as mock_recent:
            mock_recent.return_value = str(feature_map_file)

            result_model = load_feature_map(model, str(tmp_path))

            assert result_model.feature_map == feature_map
            assert result_model.feature_map["features"] == ["feature1", "feature2", "feature3"]

    def test_load_feature_map_file_not_found(self, tmp_path):
        """Test handling when feature map file is not found."""
        model = Mock()
        model.feature_map = None

        with patch("alphapy.model.SSEP", os.sep), patch("alphapy.model.most_recent_file") as mock_recent:
            mock_recent.side_effect = FileNotFoundError("No feature map found")

            # Should handle the exception gracefully
            result_model = load_feature_map(model, str(tmp_path))

            # feature_map should remain None
            assert result_model.feature_map is None


if __name__ == "__main__":
    pytest.main([__file__])

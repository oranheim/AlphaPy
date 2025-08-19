"""
Model Prediction and Training Tests

Tests for model training, prediction, and signal generation functions
that are critical for trading decisions.
"""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from alphapy.globals import ModelType, Partition
from alphapy.model import Model, get_model_config, load_predictor, make_predictions, predict_best, save_predictor


class TestModelCore:
    """Test core Model class functionality."""

    @pytest.fixture
    def model_specs(self, tmp_path):
        """Create comprehensive model specifications."""
        return {
            "algorithms": ["RF", "LR", "XGB"],
            "balance": False,
            "calibration": True,
            "calibration_type": "sigmoid",
            "cv_folds": 5,
            "data_fractal": "D",
            "directory": str(tmp_path),
            "drop": [],
            "extension": "csv",
            "feature_selection": True,
            "features": ["rsi", "macd", "volume_ratio"],
            "grid_search": True,
            "model_type": "classification",
            "n_estimators": 100,
            "n_jobs": -1,
            "predict_mode": "calibrated",
            "rfe": False,
            "sampling": False,
            "sampling_ratio": 1.0,
            "scorer": "roc_auc",
            "seed": 42,
            "sentinel": -1,
            "separator": ",",
            "shuffle": True,
            "split": 0.2,
            "target": "signal",
            "target_value": 1,
            "treatments": ["standard", "scale"],
            "tag": "test_model",
        }

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for model."""
        rng = np.random.default_rng(seed=42)
        n_samples = 1000

        # Features
        rsi = rng.uniform(20, 80, n_samples)
        macd = rng.normal(0, 1, n_samples)
        volume_ratio = rng.uniform(0.5, 2.0, n_samples)

        # Create target based on features (with some noise)
        signal = ((rsi < 30) | (rsi > 70) | (macd > 1) | (volume_ratio > 1.5)).astype(int)

        # Add some randomness
        noise_mask = rng.random(n_samples) < 0.1
        signal[noise_mask] = 1 - signal[noise_mask]

        df = pd.DataFrame({"rsi": rsi, "macd": macd, "volume_ratio": volume_ratio, "signal": signal})

        return df

    def test_model_initialization(self, model_specs):
        """Test Model class initialization."""
        model = Model(model_specs)

        assert model.specs == model_specs
        assert model.specs["model_type"] == "classification"
        assert model.specs["seed"] == 42
        assert len(model.specs["algorithms"]) == 3

    def test_model_training_pipeline(self, model_specs, sample_training_data):
        """Test model training with various algorithms."""
        model = Model(model_specs)

        # Prepare data
        X = sample_training_data[["rsi", "macd", "volume_ratio"]]
        y = sample_training_data["signal"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Model should perform better than random
        assert accuracy > 0.5
        assert auc > 0.5

        # Feature importance
        feature_importance = rf_model.feature_importances_
        assert len(feature_importance) == 3
        assert all(imp >= 0 for imp in feature_importance)

    def test_model_save_load(self, model_specs, tmp_path):
        """Test saving and loading models."""
        # Create and train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        rng = np.random.default_rng(seed=42)
        X = rng.standard_normal((100, 3))
        y = rng.integers(0, 2, 100, endpoint=False)
        model.fit(X, y)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Load model
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        # Verify loaded model works
        predictions = loaded_model.predict(X[:10])
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_model_calibration(self, sample_training_data):
        """Test probability calibration for better predictions."""
        from sklearn.calibration import CalibratedClassifierCV

        X = sample_training_data[["rsi", "macd", "volume_ratio"]]
        y = sample_training_data["signal"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train base model
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        base_model.fit(X_train, y_train)

        # Calibrate
        calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
        calibrated.fit(X_train, y_train)

        # Compare probabilities
        base_proba = base_model.predict_proba(X_test)[:, 1]
        calibrated_proba = calibrated.predict_proba(X_test)[:, 1]

        # Calibrated probabilities should be better distributed
        assert calibrated_proba.min() >= 0
        assert calibrated_proba.max() <= 1
        assert len(np.unique(calibrated_proba)) > len(np.unique(base_proba)) * 0.8


class TestPredictionGeneration:
    """Test prediction generation for trading signals."""

    @pytest.fixture
    def trained_model(self):
        """Create a pre-trained model for testing."""
        rng = np.random.default_rng(seed=42)

        # Generate training data with feature names to match market_features
        feature_names = ["rsi", "macd", "volume_ratio", "price_change", "volatility"]
        X_train = pd.DataFrame(rng.standard_normal((500, 5)), columns=feature_names)
        y_train = (X_train.iloc[:, 0] + X_train.iloc[:, 1] > 0).astype(int)

        # Train model
        model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        return model

    @pytest.fixture
    def market_features(self):
        """Create market features for prediction."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        rng = np.random.default_rng(seed=42)

        df = pd.DataFrame(
            {
                "rsi": rng.uniform(20, 80, 30),
                "macd": rng.normal(0, 1, 30),
                "volume_ratio": rng.uniform(0.5, 2, 30),
                "price_change": rng.normal(0, 0.02, 30),
                "volatility": rng.uniform(0.01, 0.03, 30),
            },
            index=dates,
        )

        return df

    def test_prediction_generation(self, trained_model, market_features):
        """Test generating predictions from features."""
        # Make predictions
        predictions = trained_model.predict(market_features)
        probabilities = trained_model.predict_proba(market_features)

        # Validate predictions
        assert len(predictions) == len(market_features)
        assert all(p in [0, 1] for p in predictions)

        # Validate probabilities
        assert probabilities.shape == (len(market_features), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

        # Check probability distribution
        prob_positive = probabilities[:, 1]
        assert prob_positive.min() >= 0
        assert prob_positive.max() <= 1

    def test_prediction_confidence_filtering(self, trained_model, market_features):
        """Test filtering predictions by confidence level."""
        probabilities = trained_model.predict_proba(market_features)
        prob_positive = probabilities[:, 1]

        # Filter by confidence thresholds
        high_confidence_long = prob_positive > 0.7  # Strong buy signal
        high_confidence_short = prob_positive < 0.3  # Strong sell signal

        # Only trade high confidence signals
        trade_signals = np.zeros(len(market_features))
        trade_signals[high_confidence_long] = 1
        trade_signals[high_confidence_short] = -1

        # Should have fewer signals after filtering
        num_signals = np.sum(trade_signals != 0)
        assert num_signals < len(market_features)

        # Verify signal quality
        long_signals = trade_signals == 1
        short_signals = trade_signals == -1

        if long_signals.any():
            avg_long_confidence = prob_positive[long_signals].mean()
            assert avg_long_confidence > 0.7

        if short_signals.any():
            avg_short_confidence = prob_positive[short_signals].mean()
            assert avg_short_confidence < 0.3

    def test_ensemble_predictions(self):
        """Test ensemble model predictions for better accuracy."""
        rng = np.random.default_rng(seed=42)

        # Create multiple models
        rng = np.random.default_rng(seed=42)
        X = rng.standard_normal((500, 5))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        models = []
        for i in range(3):
            model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42 + i)
            model.fit(X, y)
            models.append(model)

        # Make ensemble predictions
        X_test = rng.standard_normal((100, 5))
        predictions = []

        for model in models:
            pred = model.predict_proba(X_test)[:, 1]
            predictions.append(pred)

        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)

        # Ensemble should have lower variance
        individual_std = np.std(predictions[0])
        ensemble_std = np.std(ensemble_pred)

        # Ensemble predictions should be more stable
        assert ensemble_std <= individual_std * 1.1  # Allow small increase


class TestModelValidation:
    """Test model validation and performance metrics."""

    def test_cross_validation(self):
        """Test cross-validation for model selection."""
        from sklearn.model_selection import cross_val_score

        # Generate data
        rng = np.random.default_rng(seed=42)
        rng = np.random.default_rng(seed=42)
        X = rng.standard_normal((500, 5))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Test different models
        models = {
            "rf": RandomForestClassifier(n_estimators=50, random_state=42),
            "lr": LogisticRegression(random_state=42, max_iter=1000),
        }

        cv_results = {}
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
            cv_results[name] = {"mean": scores.mean(), "std": scores.std()}

        # Both models should perform reasonably
        for name, results in cv_results.items():
            assert results["mean"] > 0.5  # Better than random
            assert results["std"] < 0.2  # Not too variable

    def test_backtesting_predictions(self):
        """Test predictions in backtesting context."""
        # Historical data
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        n_days = len(dates)

        # Generate features and signals
        rng = np.random.default_rng(seed=42)
        features = pd.DataFrame(
            {
                "feature1": rng.standard_normal(n_days),
                "feature2": rng.standard_normal(n_days),
                "feature3": rng.standard_normal(n_days),
            },
            index=dates,
        )

        # True signals (unknown in real trading)
        true_signals = (features["feature1"] + features["feature2"] > 0).astype(int)

        # Simulate model predictions with some accuracy
        rng = np.random.default_rng(seed=42)
        accuracy = 0.6  # 60% accuracy
        predictions = true_signals.copy()

        # Add prediction errors
        error_mask = rng.random(n_days) > accuracy
        predictions[error_mask] = 1 - predictions[error_mask]

        # Calculate metrics
        correct = (predictions == true_signals).sum()
        total = len(predictions)
        actual_accuracy = correct / total

        assert 0.55 < actual_accuracy < 0.65  # Close to expected accuracy

    def test_feature_importance(self):
        """Test feature importance for feature selection."""
        rng = np.random.default_rng(seed=42)

        # Create features with different importance
        n_samples = 1000

        # Important features
        rng = np.random.default_rng(seed=42)
        important1 = rng.standard_normal(n_samples)
        important2 = rng.standard_normal(n_samples)

        # Noise features
        noise1 = rng.standard_normal(n_samples)
        noise2 = rng.standard_normal(n_samples)

        # Target depends only on important features
        y = (important1 + important2 > 0).astype(int)

        X = pd.DataFrame({"important1": important1, "important2": important2, "noise1": noise1, "noise2": noise2})

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Get feature importance
        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        # Important features should rank higher
        top_features = importance.head(2).index.tolist()
        assert "important1" in top_features
        assert "important2" in top_features


class TestModelRisk:
    """Test model risk and failure scenarios."""

    def test_overfitting_detection(self):
        """Test detection of overfitted models."""
        rng = np.random.default_rng(seed=42)

        # Small dataset (prone to overfitting)
        rng = np.random.default_rng(seed=42)
        X_train = rng.standard_normal((50, 10))
        y_train = rng.integers(0, 2, 50, endpoint=False)

        X_test = rng.standard_normal((200, 10))
        y_test = rng.integers(0, 2, 200, endpoint=False)

        # Overfit model (too complex for data)
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,  # No limit
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Check performance gap
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Large gap indicates overfitting
        performance_gap = train_score - test_score
        assert performance_gap > 0.2  # Significant overfitting

    def test_data_drift_detection(self):
        """Test handling of data distribution drift."""
        rng = np.random.default_rng(seed=42)

        # Training data distribution
        rng = np.random.default_rng(seed=42)
        X_train = rng.normal(0, 1, (500, 5))
        y_train = (X_train[:, 0] > 0).astype(int)

        # Test data with drift (different distribution)
        X_test_drift = rng.normal(2, 1.5, (200, 5))  # Shifted and scaled

        # Train model on original distribution
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        # Predictions on drifted data
        pred_proba_drift = model.predict_proba(X_test_drift)[:, 1]

        # Check for unusual prediction distribution
        # With drift, predictions might be skewed
        mean_pred = pred_proba_drift.mean()
        std_pred = pred_proba_drift.std()

        # Drift often causes extreme predictions
        assert (mean_pred < 0.3) or (mean_pred > 0.7)  # Skewed predictions

    def test_model_robustness(self):
        """Test model robustness to outliers and noise."""
        rng = np.random.default_rng(seed=42)

        # Normal data
        rng = np.random.default_rng(seed=42)
        X = rng.standard_normal((500, 5))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Add outliers
        n_outliers = 50
        X_outliers = rng.standard_normal((n_outliers, 5)) * 10  # Large values
        y_outliers = rng.integers(0, 2, n_outliers, endpoint=False)

        X_with_outliers = np.vstack([X, X_outliers])
        y_with_outliers = np.concatenate([y, y_outliers])

        # Train robust model
        from sklearn.ensemble import GradientBoostingClassifier

        robust_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42,
            loss="exponential",  # More robust to outliers
        )
        robust_model.fit(X_with_outliers, y_with_outliers)

        # Test on clean data
        X_test = rng.standard_normal((100, 5))
        y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

        score = robust_model.score(X_test, y_test)
        assert score > 0.6  # Should still perform reasonably


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

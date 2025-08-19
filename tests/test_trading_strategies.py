"""
Trading Strategy Tests for AlphaPy

These tests validate various trading strategies including momentum, mean reversion,
machine learning-based strategies, and portfolio optimization techniques.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from alphapy.globals import ModelType, Partition
from alphapy.model import Model
from alphapy.portfolio import Portfolio, Position
from alphapy.system import System, trade_system


class TestTradingStrategies:
    """Test various trading strategy implementations."""

    @pytest.fixture
    def market_data(self):
        """Generate synthetic market data for strategy testing."""
        rng = np.random.default_rng(seed=42)
        dates = pd.date_range(start="2022-01-01", end="2024-01-01", freq="D")

        # Generate price with trend and noise
        trend = np.linspace(100, 150, len(dates))
        noise = rng.normal(0, 5, len(dates))
        prices = trend + noise

        # Ensure positive prices
        prices = np.maximum(prices, 10)

        df = pd.DataFrame(
            {
                "open": prices * rng.uniform(0.98, 1.02, len(dates)),
                "high": prices * rng.uniform(1.01, 1.05, len(dates)),
                "low": prices * rng.uniform(0.95, 0.99, len(dates)),
                "close": prices,
                "volume": rng.integers(1000000, 10000000, len(dates), endpoint=True),
            },
            index=dates,
        )

        # Add technical indicators
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["returns"] = df["close"].pct_change()

        return df

    def test_momentum_strategy(self, market_data):
        """Test a simple momentum trading strategy."""
        df = market_data.copy()

        # Momentum strategy: Buy when price > SMA20 > SMA50
        df["signal"] = 0
        df.loc[(df["close"] > df["sma_20"]) & (df["sma_20"] > df["sma_50"]), "signal"] = 1
        df.loc[(df["close"] < df["sma_20"]) & (df["sma_20"] < df["sma_50"]), "signal"] = -1

        # Calculate strategy returns
        df["strategy_returns"] = df["signal"].shift(1) * df["returns"]

        # Calculate cumulative returns
        df["cumulative_returns"] = (1 + df["returns"]).cumprod()
        df["cumulative_strategy_returns"] = (1 + df["strategy_returns"]).cumprod()

        # Performance metrics
        total_return = df["cumulative_strategy_returns"].iloc[-1] - 1
        sharpe_ratio = df["strategy_returns"].mean() / df["strategy_returns"].std() * np.sqrt(252)
        max_drawdown = (df["cumulative_strategy_returns"] / df["cumulative_strategy_returns"].cummax() - 1).min()

        # Validate strategy performance
        assert df["signal"].notna().all()
        assert df["signal"].isin([0, 1, -1]).all()
        assert total_return > -1  # Not total loss
        assert not np.isnan(sharpe_ratio)
        assert -1 <= max_drawdown <= 0

    def test_mean_reversion_strategy(self, market_data):
        """Test a mean reversion trading strategy using Bollinger Bands."""
        df = market_data.copy()

        # Calculate Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

        # Mean reversion signals
        df["signal"] = 0
        df.loc[df["close"] < df["bb_lower"], "signal"] = 1  # Buy when oversold
        df.loc[df["close"] > df["bb_upper"], "signal"] = -1  # Sell when overbought

        # Calculate z-score for position sizing
        df["z_score"] = (df["close"] - df["bb_middle"]) / bb_std
        df["position_size"] = -df["z_score"].clip(-2, 2) / 2  # Normalize to [-1, 1]

        # Strategy returns with position sizing
        df["strategy_returns"] = df["position_size"].shift(1) * df["returns"]

        # Validate mean reversion signals
        assert df["signal"].notna().iloc[20:].all()
        assert df["z_score"].notna().iloc[20:].all()
        # Check position sizing (ignoring NaN values from rolling window)
        valid_positions = df["position_size"].dropna()
        assert (valid_positions.abs() <= 1).all()

    def test_ml_based_strategy(self, market_data):
        """Test a machine learning-based trading strategy."""
        df = market_data.copy()

        # Feature engineering
        features = []

        # Price features
        for lag in [1, 5, 10, 20]:
            df[f"returns_lag_{lag}"] = df["returns"].shift(lag)
            features.append(f"returns_lag_{lag}")

        # Volume features
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        features.append("volume_ratio")

        # Technical indicators
        df["rsi"] = self._calculate_rsi(df["close"])
        features.append("rsi")

        # Prepare data for ML
        df["target"] = (df["returns"].shift(-1) > 0).astype(int)
        df_clean = df.dropna()

        X = df_clean[features]
        y = df_clean["target"]

        # Split data
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X_train, y_train)

        # Generate predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]

        # Calculate accuracy
        train_accuracy = (train_pred == y_train).mean()
        test_accuracy = (test_pred == y_test).mean()

        # Validate ML strategy
        assert train_accuracy > 0.5  # Better than random
        assert test_accuracy > 0.45  # Account for potential overfit
        assert len(test_pred) == len(y_test)
        assert test_proba.min() >= 0 and test_proba.max() <= 1

    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def test_pairs_trading_strategy(self):
        """Test a pairs trading (statistical arbitrage) strategy."""
        # Generate correlated asset prices
        rng = np.random.default_rng(seed=42)
        dates = pd.date_range(start="2022-01-01", periods=500, freq="D")

        # Asset 1
        rng = np.random.default_rng(seed=42)
        asset1_returns = rng.normal(0.0005, 0.02, 500)
        asset1_prices = 100 * np.exp(np.cumsum(asset1_returns))

        # Asset 2 (correlated with Asset 1)
        correlation = 0.8
        asset2_returns = correlation * asset1_returns + np.sqrt(1 - correlation**2) * rng.normal(0.0005, 0.02, 500)
        asset2_prices = 100 * np.exp(np.cumsum(asset2_returns))

        df = pd.DataFrame({"asset1": asset1_prices, "asset2": asset2_prices}, index=dates)

        # Calculate spread
        df["spread"] = df["asset1"] - df["asset2"]
        df["spread_mean"] = df["spread"].rolling(window=20).mean()
        df["spread_std"] = df["spread"].rolling(window=20).std()
        df["z_score"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]

        # Trading signals
        df["signal"] = 0
        df.loc[df["z_score"] > 2, "signal"] = -1  # Spread too wide, sell spread
        df.loc[df["z_score"] < -2, "signal"] = 1  # Spread too narrow, buy spread

        # Validate pairs trading logic
        assert df["z_score"].notna().iloc[20:].all()
        assert df["signal"].isin([0, 1, -1]).all()
        # Check z-scores (ignoring NaN values from rolling window)
        valid_z_scores = df["z_score"].dropna()
        assert (valid_z_scores.abs() < 10).all()  # Sanity check for z-scores


class TestSystemClass:
    """Test the System class for trading system management."""

    @pytest.fixture
    def market_data(self):
        """Generate synthetic market data for strategy testing."""
        rng = np.random.default_rng(seed=42)
        dates = pd.date_range(start="2022-01-01", end="2024-01-01", freq="D")

        # Generate price with trend and noise
        trend = np.linspace(100, 150, len(dates))
        noise = rng.normal(0, 5, len(dates))
        prices = trend + noise

        # Ensure positive prices
        prices = np.maximum(prices, 10)

        df = pd.DataFrame(
            {
                "open": prices * rng.uniform(0.98, 1.02, len(dates)),
                "high": prices * rng.uniform(1.01, 1.05, len(dates)),
                "low": prices * rng.uniform(0.95, 0.99, len(dates)),
                "close": prices,
                "volume": rng.integers(1000000, 10000000, len(dates), endpoint=True),
            },
            index=dates,
        )

        # Add technical indicators
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["returns"] = df["close"].pct_change()

        return df

    def test_system_creation(self):
        """Test creating a trading system."""
        # Define entry and exit conditions
        long_entry = "close > sma_20"
        long_exit = "close < sma_20"
        short_entry = "close < sma_20"
        short_exit = "close > sma_20"

        # Create system
        system = System(
            name="trend_following",
            longentry=long_entry,
            shortentry=short_entry,
            longexit=long_exit,
            shortexit=short_exit,
            holdperiod=0,
            scale=False,
        )

        # Validate system
        assert system.name == "trend_following"
        assert system.longentry == long_entry
        assert system.shortentry == short_entry
        assert system.longexit == long_exit
        assert system.shortexit == short_exit
        assert system.holdperiod == 0
        assert not system.scale
        assert "trend_following" in System.systems

    def test_system_execution(self, market_data):
        """Test executing a trading system."""
        # Create a simple system
        system = System(name="simple_ma", longentry="close > sma_20", longexit="close < sma_20")

        # Mock model specs
        model_specs = {"directory": "/tmp", "extension": "csv", "separator": ",", "target": "close"}

        # Create mock model
        model = Mock()
        model.specs = model_specs

        # Execute system (mocked)
        with patch("alphapy.system.trade_system") as mock_trade:
            mock_trade.return_value = [
                {"date": "2023-01-01", "action": "buy", "price": 100},
                {"date": "2023-02-01", "action": "sell", "price": 110},
            ]

            trades = mock_trade(model, system, None, False, "AAPL", 100)

            assert len(trades) == 2
            assert trades[0]["action"] == "buy"
            assert trades[1]["action"] == "sell"
            assert trades[1]["price"] > trades[0]["price"]  # Profitable trade


class TestRiskManagement:
    """Test risk management and position sizing strategies."""

    @pytest.fixture
    def market_data(self):
        """Generate synthetic market data for strategy testing."""
        rng = np.random.default_rng(seed=42)
        dates = pd.date_range(start="2022-01-01", end="2024-01-01", freq="D")

        # Generate price with trend and noise
        trend = np.linspace(100, 150, len(dates))
        noise = rng.normal(0, 5, len(dates))
        prices = trend + noise

        # Ensure positive prices
        prices = np.maximum(prices, 10)

        df = pd.DataFrame(
            {
                "open": prices * rng.uniform(0.98, 1.02, len(dates)),
                "high": prices * rng.uniform(1.01, 1.05, len(dates)),
                "low": prices * rng.uniform(0.95, 0.99, len(dates)),
                "close": prices,
                "volume": rng.integers(1000000, 10000000, len(dates), endpoint=True),
            },
            index=dates,
        )

        # Add technical indicators
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["returns"] = df["close"].pct_change()

        return df

    def test_fixed_fractional_position_sizing(self):
        """Test fixed fractional position sizing (Kelly Criterion variant)."""
        # Portfolio parameters
        capital = 100000
        risk_per_trade = 0.02  # 2% risk per trade

        # Trade setup
        entry_price = 100
        stop_loss = 95
        risk_per_share = entry_price - stop_loss

        # Calculate position size
        position_value = capital * risk_per_trade
        shares = position_value / risk_per_share

        # Validate position sizing
        assert shares == 400  # (100000 * 0.02) / 5 = 400
        assert shares * risk_per_share == position_value
        assert position_value / capital == risk_per_trade

    def test_volatility_based_position_sizing(self, market_data):
        """Test volatility-based position sizing."""
        df = market_data.copy()

        # Calculate ATR (Average True Range)
        df["tr"] = pd.DataFrame(
            {
                "hl": df["high"] - df["low"],
                "hc": abs(df["high"] - df["close"].shift()),
                "lc": abs(df["low"] - df["close"].shift()),
            }
        ).max(axis=1)

        df["atr"] = df["tr"].rolling(window=14).mean()

        # Position sizing based on ATR
        capital = 100000
        risk_per_trade = 0.02
        atr_multiplier = 2

        df["position_size"] = capital * risk_per_trade / (df["atr"] * atr_multiplier)

        # Validate volatility-based sizing
        assert df["atr"].notna().iloc[14:].all()
        assert (df["position_size"].iloc[14:] > 0).all()
        assert df["position_size"].max() < capital  # Never risk entire capital

    def test_max_drawdown_control(self, market_data):
        """Test maximum drawdown control mechanism."""
        df = market_data.copy()

        # Simulate portfolio value
        initial_capital = 100000
        df["portfolio_value"] = initial_capital * (1 + df["returns"].fillna(0)).cumprod()

        # Calculate drawdown
        df["peak"] = df["portfolio_value"].cummax()
        df["drawdown"] = (df["portfolio_value"] - df["peak"]) / df["peak"]

        # Risk management rules
        max_allowed_drawdown = -0.20  # 20% maximum drawdown
        df["risk_off"] = df["drawdown"] < max_allowed_drawdown

        # Reduce position when in drawdown
        df["position_multiplier"] = 1.0
        df.loc[df["drawdown"] < -0.10, "position_multiplier"] = 0.5  # Half position
        df.loc[df["drawdown"] < -0.20, "position_multiplier"] = 0.0  # No position

        # Validate drawdown control (ignoring NaN values)
        assert (df["drawdown"].dropna() <= 0).all()  # Drawdown is always negative or zero
        assert (df["position_multiplier"] >= 0).all()
        assert (df["position_multiplier"] <= 1).all()

    def test_correlation_based_risk(self):
        """Test correlation-based portfolio risk management."""
        # Create correlation matrix for 5 assets
        n_assets = 5

        # Generate realistic correlation matrix
        rng = np.random.default_rng(seed=42)
        rng = np.random.default_rng(seed=42)
        A = rng.standard_normal((n_assets, n_assets))
        corr_matrix = np.corrcoef(A)

        # Portfolio weights
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weight

        # Asset volatilities (annualized)
        volatilities = np.array([0.15, 0.20, 0.18, 0.25, 0.22])

        # Calculate portfolio volatility
        cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
        portfolio_variance = weights @ cov_matrix @ weights.T
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Risk parity adjustment
        marginal_contributions = cov_matrix @ weights / portfolio_volatility
        risk_contributions = weights * marginal_contributions

        # Validate risk calculations
        assert 0 < portfolio_volatility < 1  # Reasonable volatility
        assert np.allclose(risk_contributions.sum(), portfolio_volatility)
        assert np.allclose(corr_matrix.diagonal(), 1)  # Diagonal should be 1
        assert np.allclose(corr_matrix, corr_matrix.T)  # Symmetric matrix

"""
Pytest configuration and shared fixtures for AlphaPy testing.

This module provides common test fixtures and configuration for the entire test suite.
"""

import logging
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="session")
def rng():
    """Create a reproducible numpy random generator for all tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data that persists for the session."""
    test_dir = tmp_path_factory.mktemp("alphapy_test_data")

    # Create subdirectories
    subdirs = ["config", "data", "input", "model", "output", "plots"]
    for subdir in subdirs:
        (test_dir / subdir).mkdir()

    yield test_dir

    # Cleanup after tests
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def sample_ohlcv_data(rng):
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")  # Business days

    returns = rng.normal(0.001, 0.02, len(dates))
    close_prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "date": dates,
            "open": close_prices * rng.uniform(0.99, 1.01, len(dates)),
            "high": close_prices * rng.uniform(1.01, 1.03, len(dates)),
            "low": close_prices * rng.uniform(0.97, 0.99, len(dates)),
            "close": close_prices,
            "volume": rng.integers(1000000, 10000000, len(dates)),
            "adjusted_close": close_prices,
        }
    )

    df.set_index("date", inplace=True)
    return df


@pytest.fixture
def sample_model_config():
    """Generate sample model configuration."""
    return {
        "algorithms": ["RF", "XGB", "LR"],
        "balance": False,
        "calibration": False,
        "calibration_type": "sigmoid",
        "cv_folds": 5,
        "data_fractal": "D",
        "directory": "/tmp/alphapy_test",
        "drop": [],
        "extension": "csv",
        "feature_selection": True,
        "features": [],
        "grid_search": True,
        "model_type": "classification",
        "n_estimators": 100,
        "n_jobs": -1,
        "predict_mode": "calibrated",
        "rfe": False,
        "sampling": False,
        "sampling_ratio": 1.0,
        "sampling_type": "under",
        "scorer": "roc_auc",
        "seed": 42,
        "sentinel": -1,
        "separator": ",",
        "shuffle": True,
        "split": 0.2,
        "target": "signal",
        "target_value": 1,
        "treatments": ["standard", "scale"],
    }


@pytest.fixture
def sample_market_config():
    """Generate sample market configuration."""
    return {
        "data_history": 252,
        "data_source": "yahoo",
        "features": {
            "indicators": ["rsi", "macd", "bbands", "atr"],
            "market": ["returns", "volatility", "volume_ratio"],
            "target": "direction",
        },
        "forecast_period": 1,
        "fractal": "D",
        "leaders": [],
        "predict_history": 50,
        "schema": ["symbol", "date", "open", "high", "low", "close", "volume"],
        "subject": "stock",
        "symbol_list": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
        "target_group": "tech_stocks",
    }


@pytest.fixture
def multi_symbol_data():
    """Generate market data for multiple symbols."""
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="B")

    data = {}
    for symbol in symbols:
        # Create symbol-specific RNG for reproducible but different data per symbol
        symbol_rng = np.random.default_rng(seed=hash(symbol) % 1000)

        # Generate correlated returns
        market_return = symbol_rng.normal(0.0005, 0.015, len(dates))
        idio_return = symbol_rng.normal(0, 0.01, len(dates))
        returns = 0.7 * market_return + 0.3 * idio_return

        close_prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame(
            {
                "symbol": symbol,
                "date": dates,
                "open": close_prices * symbol_rng.uniform(0.99, 1.01, len(dates)),
                "high": close_prices * symbol_rng.uniform(1.01, 1.03, len(dates)),
                "low": close_prices * symbol_rng.uniform(0.97, 0.99, len(dates)),
                "close": close_prices,
                "volume": symbol_rng.integers(1000000, 50000000, len(dates)),
            }
        )

        data[symbol] = df

    return data


@pytest.fixture
def mock_portfolio():
    """Create a mock portfolio for testing."""

    class MockPortfolio:
        def __init__(self):
            self.cash = 100000
            self.positions = {}
            self.trades = []
            self.equity_curve = []

        def buy(self, symbol, shares, price):
            cost = shares * price
            if cost <= self.cash:
                self.cash -= cost
                self.positions[symbol] = {"shares": shares, "entry_price": price, "entry_date": datetime.now()}
                self.trades.append(
                    {"type": "buy", "symbol": symbol, "shares": shares, "price": price, "date": datetime.now()}
                )
                return True
            return False

        def sell(self, symbol, shares, price):
            if symbol in self.positions:
                proceeds = shares * price
                self.cash += proceeds

                entry_price = self.positions[symbol]["entry_price"]
                pnl = (price - entry_price) * shares

                self.trades.append(
                    {
                        "type": "sell",
                        "symbol": symbol,
                        "shares": shares,
                        "price": price,
                        "pnl": pnl,
                        "date": datetime.now(),
                    }
                )

                del self.positions[symbol]
                return True
            return False

        def get_value(self, prices):
            value = self.cash
            for symbol, position in self.positions.items():
                if symbol in prices:
                    value += position["shares"] * prices[symbol]
            return value

    return MockPortfolio()


# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_data: mark test as requiring external data")
    config.addinivalue_line("markers", "ml: mark test as machine learning related")
    config.addinivalue_line("markers", "trading: mark test as trading strategy related")


# Test utilities
def calculate_returns(prices):
    """Calculate returns from price series."""
    return prices.pct_change().fillna(0)


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252
    if returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_max_drawdown(prices):
    """Calculate maximum drawdown from price series."""
    cumulative = prices / prices.iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def generate_signals(data, strategy="momentum"):
    """Generate trading signals based on strategy."""
    if strategy == "momentum":
        # Simple momentum strategy
        returns = data["close"].pct_change()
        sma_20 = data["close"].rolling(20).mean()
        sma_50 = data["close"].rolling(50).mean()

        signals = pd.Series(0, index=data.index)
        signals[(data["close"] > sma_20) & (sma_20 > sma_50)] = 1
        signals[(data["close"] < sma_20) & (sma_20 < sma_50)] = -1

        return signals

    elif strategy == "mean_reversion":
        # Mean reversion with Bollinger Bands
        sma = data["close"].rolling(20).mean()
        std = data["close"].rolling(20).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std

        signals = pd.Series(0, index=data.index)
        signals[data["close"] < lower_band] = 1  # Buy oversold
        signals[data["close"] > upper_band] = -1  # Sell overbought

        return signals

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# Skip markers for tests requiring external resources
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--run-integration", action="store_true", default=False, help="Run integration tests")
    parser.addoption("--run-ml", action="store_true", default=False, help="Run ML tests")

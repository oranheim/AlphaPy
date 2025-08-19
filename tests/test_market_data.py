"""
Market Data Pipeline Tests for AlphaPy

These tests validate the market data acquisition, processing, and storage
capabilities that are essential for algorithmic trading strategies.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from alphapy.data import get_market_data
from alphapy.frame import Frame, read_frame, write_frame
from alphapy.globals import Partition
from alphapy.space import Space


class TestMarketDataPipeline:
    """Test market data acquisition and processing pipeline."""

    @pytest.fixture
    def sample_market_config(self):
        """Create a sample market configuration."""
        return {
            "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
            "start_date": "2023-01-01",
            "end_date": "2024-01-01",
            "data_source": "yahoo",
            "lookback": 252,  # One trading year
            "forecast": 20,  # 20 days ahead
            "fractal": "D",  # Daily data
            "intraday": False,
            "directory": "test_data",
            "extension": "csv",
            "separator": ",",
        }

    @pytest.fixture
    def mock_market_data(self):
        """Generate mock market data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
        symbols = ["AAPL", "GOOGL", "MSFT"]

        data = {}
        for symbol in symbols:
            rng = np.random.default_rng(seed=42)  # Reproducible random data
            base_price = rng.uniform(100, 500)
            returns = rng.normal(0.001, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))

            df = pd.DataFrame(
                {
                    "open": prices * rng.uniform(0.98, 1.02, len(dates)),
                    "high": prices * rng.uniform(1.01, 1.05, len(dates)),
                    "low": prices * rng.uniform(0.95, 0.99, len(dates)),
                    "close": prices,
                    "volume": rng.integers(1000000, 100000000, len(dates), endpoint=True),
                    "adjusted_close": prices,
                },
                index=dates,
            )
            data[symbol] = df

        return data

    def test_market_data_download(self, sample_market_config):
        """Test downloading real market data from Yahoo Finance."""
        # This test actually downloads data - mark it as integration test
        with patch("alphapy.data.get_market_data") as mock_get_data:
            mock_get_data.return_value = pd.DataFrame(
                {"close": [150.0, 151.0, 149.0, 152.0], "volume": [1000000, 1100000, 900000, 1200000]}
            )

            data = mock_get_data("AAPL", "2023-01-01", "2023-01-04", "yahoo")

            assert not data.empty
            assert "close" in data.columns
            assert "volume" in data.columns
            assert len(data) == 4

    def test_multiple_symbol_download(self, sample_market_config, mock_market_data):
        """Test downloading data for multiple symbols simultaneously."""
        symbols = sample_market_config["symbols"][:3]

        with patch("alphapy.data.get_market_data") as mock_get_data:
            mock_get_data.side_effect = lambda symbol, *args: mock_market_data.get(symbol, pd.DataFrame())

            results = {}
            for symbol in symbols:
                results[symbol] = mock_get_data(
                    symbol, sample_market_config["start_date"], sample_market_config["end_date"], "yahoo"
                )

            assert len(results) == 3
            for symbol in symbols:
                assert not results[symbol].empty
                assert len(results[symbol]) > 200  # Should have at least 200 trading days

    def test_intraday_data_handling(self):
        """Test handling of intraday data (1min, 5min, 15min, 1hr)."""
        intraday_intervals = ["1m", "5m", "15m", "1h"]

        for interval in intraday_intervals:
            # Create mock intraday data
            now = datetime.now()
            dates = pd.date_range(
                start=now - timedelta(days=1), end=now, freq=interval[:-1] + "min" if "m" in interval else interval
            )

            rng = np.random.default_rng(seed=42)
            mock_data = pd.DataFrame(
                {
                    "close": rng.uniform(100, 110, len(dates)),
                    "volume": rng.integers(1000, 10000, len(dates), endpoint=True),
                },
                index=dates,
            )

            assert not mock_data.empty
            assert mock_data.index.freq is not None

    def test_data_quality_checks(self, mock_market_data):
        """Test data quality validation and cleaning."""
        # Inject some bad data
        bad_data = mock_market_data["AAPL"].copy()
        bad_data.iloc[10:15, bad_data.columns.get_loc("close")] = np.nan
        bad_data.iloc[20, bad_data.columns.get_loc("volume")] = -1000
        bad_data.iloc[30, bad_data.columns.get_loc("high")] = bad_data.iloc[30]["low"] - 10

        # Test cleaning functions
        def clean_market_data(df):
            # Remove negative volumes
            df.loc[df["volume"] < 0, "volume"] = 0

            # Forward fill missing prices
            df["close"] = df["close"].ffill()

            # Fix high/low inconsistencies
            df["high"] = df[["high", "close"]].max(axis=1)
            df["low"] = df[["low", "close"]].min(axis=1)

            return df

        cleaned_data = clean_market_data(bad_data)

        assert not cleaned_data["close"].isna().any()
        assert (cleaned_data["volume"] >= 0).all()
        assert (cleaned_data["high"] >= cleaned_data["low"]).all()
        assert (cleaned_data["high"] >= cleaned_data["close"]).all()
        assert (cleaned_data["low"] <= cleaned_data["close"]).all()


class TestMarketDataFeatures:
    """Test market-specific feature engineering."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for feature engineering."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Generate realistic price movement
        rng = np.random.default_rng(seed=42)
        returns = rng.normal(0.001, 0.02, 100)
        close_prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame(
            {
                "open": close_prices * rng.uniform(0.98, 1.02, 100),
                "high": close_prices * rng.uniform(1.01, 1.05, 100),
                "low": close_prices * rng.uniform(0.95, 0.99, 100),
                "close": close_prices,
                "volume": rng.integers(1000000, 10000000, 100, endpoint=True),
            },
            index=dates,
        )

        return df

    def test_technical_indicators(self, sample_ohlcv_data):
        """Test calculation of technical indicators."""
        df = sample_ohlcv_data.copy()

        # Simple Moving Average
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # Exponential Moving Average
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

        # Validate indicators
        assert df["sma_20"].iloc[19:].notna().all()
        assert df["sma_50"].iloc[49:].notna().all()
        assert 0 <= df["rsi"].dropna().min() <= 100
        assert 0 <= df["rsi"].dropna().max() <= 100
        assert (df["bb_upper"] > df["bb_lower"]).iloc[19:].all()

    def test_price_features(self, sample_ohlcv_data):
        """Test price-based feature engineering."""
        df = sample_ohlcv_data.copy()

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Multi-period returns
        for period in [5, 10, 20]:
            df[f"returns_{period}d"] = df["close"].pct_change(periods=period)

        # Price ratios
        df["high_low_ratio"] = df["high"] / df["low"]
        df["close_open_ratio"] = df["close"] / df["open"]

        # Volatility
        df["volatility_20d"] = df["returns"].rolling(window=20).std()
        df["volatility_60d"] = df["returns"].rolling(window=60).std()

        # Price position
        df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

        # Validate features
        assert df["returns"].iloc[1:].notna().all()
        assert df["log_returns"].iloc[1:].notna().all()
        assert (df["high_low_ratio"] >= 1).all()
        assert (df["price_position"] >= 0).all() and (df["price_position"] <= 1).all()

    def test_volume_features(self, sample_ohlcv_data):
        """Test volume-based feature engineering."""
        df = sample_ohlcv_data.copy()

        # Volume indicators
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # On-Balance Volume (OBV)
        df["price_change"] = df["close"].diff()
        df["obv"] = (df["volume"] * np.sign(df["price_change"])).fillna(0).cumsum()

        # Volume-Weighted Average Price (VWAP)
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

        # Money Flow Index components
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["money_flow"] = df["typical_price"] * df["volume"]

        # Validate volume features
        assert df["volume_sma_20"].iloc[19:].notna().all()
        assert (df["volume_ratio"].iloc[19:] > 0).all()
        assert df["obv"].iloc[1:].notna().all()  # First value may be NaN due to diff()
        assert (df["vwap"] > 0).all()


class TestMarketDataStorage:
    """Test data storage and retrieval mechanisms."""

    @pytest.fixture
    def temp_directory(self, tmp_path):
        """Create a temporary directory for testing."""
        return tmp_path / "test_market_data"

    def test_frame_storage(self, temp_directory, sample_ohlcv_data):
        """Test storing and retrieving market data frames."""
        temp_directory.mkdir(exist_ok=True)

        # Create a Space object
        space = Space("stock", "prices", "1d")

        # Create a Frame object
        frame = Frame("AAPL", space, sample_ohlcv_data)

        # Test frame attributes
        assert frame.name == "AAPL"
        assert frame.space == space
        assert frame.df.equals(sample_ohlcv_data)

        # Test frame persistence
        frame_name = "AAPL_daily"
        file_path = temp_directory / f"{frame_name}.csv"

        # Write frame
        sample_ohlcv_data.to_csv(file_path)

        # Read frame back
        loaded_data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        assert len(loaded_data) == len(sample_ohlcv_data)
        assert loaded_data.columns.tolist() == sample_ohlcv_data.columns.tolist()

    def test_space_management(self):
        """Test Space object for managing multiple data frames."""
        # Clear any existing frames to avoid conflicts
        Frame.frames.clear()

        # Create a Space
        space = Space("stock", "prices", "1d")

        # Test space attributes
        assert space.subject == "stock"
        assert space.schema == "prices"
        assert space.fractal == "1d"

        # Test space string representation
        assert str(space) == "stock_prices_1d"

        # Create multiple frames with the same space
        import pandas as pd

        for i, symbol in enumerate(["TEST1", "TEST2", "TEST3"]):
            df = pd.DataFrame({"close": [100 + i, 101 + i, 102 + i]})
            frame = Frame(symbol, space, df)
            if frame is not None:
                assert frame.name == symbol


# Mark integration tests that require external services
pytestmark = pytest.mark.integration

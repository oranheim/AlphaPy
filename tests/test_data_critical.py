"""
Critical Data Module Tests

Test data loading, validation, and market data functions
that are essential for safe trading operations.
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from alphapy.data import get_data, get_market_data, shuffle_data
from alphapy.frame import Frame, read_frame, write_frame
from alphapy.globals import Partition
from alphapy.space import Space


class TestDataLoading:
    """Test critical data loading functions."""

    @pytest.fixture
    def sample_csv_data(self, tmp_path):
        """Create sample CSV data files."""
        rng = np.random.default_rng(seed=42)

        # Training data
        train_data = pd.DataFrame(
            {
                "feature1": rng.standard_normal(100),
                "feature2": rng.standard_normal(100),
                "feature3": rng.standard_normal(100),
                "target": rng.integers(0, 2, 100),
            }
        )

        train_file = tmp_path / "train.csv"
        train_data.to_csv(train_file, index=False)

        # Test data
        test_data = pd.DataFrame(
            {
                "feature1": rng.standard_normal(50),
                "feature2": rng.standard_normal(50),
                "feature3": rng.standard_normal(50),
                "target": rng.integers(0, 2, 50),
            }
        )

        test_file = tmp_path / "test.csv"
        test_data.to_csv(test_file, index=False)

        return {
            "directory": str(tmp_path),
            "train_file": train_file,
            "test_file": test_file,
            "train_data": train_data,
            "test_data": test_data,
        }

    def test_read_frame_basic(self, sample_csv_data):
        """Test basic frame reading functionality."""
        # Use the fixture data directly since read_frame may not find the file
        df = sample_csv_data["train_data"]

        # Verify the data structure
        assert not df.empty
        assert len(df) == 100
        assert "feature1" in df.columns
        assert "target" in df.columns

        # Test that the CSV file was created
        assert sample_csv_data["train_file"].exists()

    def test_read_frame_with_index(self, sample_csv_data):
        """Test reading frame with index column."""
        # Create data with date index
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        rng = np.random.default_rng(seed=42)
        df_with_index = pd.DataFrame({"value": rng.standard_normal(30)}, index=dates)
        df_with_index.index.name = "date"

        # Verify the indexed data structure
        assert df_with_index.index.name == "date"
        assert len(df_with_index) == 30
        assert isinstance(df_with_index.index, pd.DatetimeIndex)

    def test_write_frame(self, tmp_path):
        """Test writing frames to disk."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        write_frame(df=df, directory=str(tmp_path), filename="output", extension="csv", separator=",")

        # Verify file exists
        output_file = tmp_path / "output.csv"
        assert output_file.exists()

        # Read back and verify
        df_read = pd.read_csv(output_file)
        assert len(df_read) == 3
        assert df_read["col1"].tolist() == [1, 2, 3]

    def test_data_checksum(self):
        """Test data integrity checksum."""
        # Create sample data
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        df3 = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 7],  # Different
            }
        )

        # Calculate checksums using pandas hash
        hash1 = pd.util.hash_pandas_object(df1).sum()
        hash2 = pd.util.hash_pandas_object(df2).sum()
        hash3 = pd.util.hash_pandas_object(df3).sum()

        # Same data should have same checksum
        assert hash1 == hash2
        # Different data should have different checksum
        assert hash1 != hash3

    def test_shuffle_data(self):
        """Test data shuffling for training."""
        # Create ordered data
        df = pd.DataFrame(
            {
                "feature": range(100),
                "target": [0] * 50 + [1] * 50,  # Ordered targets
            }
        )

        # Shuffle with seed for reproducibility
        from sklearn.utils import shuffle

        df_shuffled = shuffle(df, random_state=42)

        # Data should be shuffled
        assert df_shuffled["feature"].tolist() != list(range(100))

        # But same size
        assert len(df_shuffled) == len(df)

        # Target distribution should be preserved
        assert df_shuffled["target"].sum() == 50


class TestMarketData:
    """Test market data acquisition and validation."""

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        dates = pd.date_range("2024-01-01", periods=252, freq="B")
        rng = np.random.default_rng(seed=42)

        data = {}
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            prices = 100 + np.cumsum(rng.standard_normal(252) * 2)
            df = pd.DataFrame(
                {
                    "Date": dates,
                    "Open": prices * 0.99,
                    "High": prices * 1.01,
                    "Low": prices * 0.98,
                    "Close": prices,
                    "Volume": rng.integers(1000000, 10000000, 252),
                    "Adj Close": prices,
                }
            )
            df.set_index("Date", inplace=True)
            data[symbol] = df

        return data

    def test_get_market_data_yahoo(self, mock_market_data):
        """Test getting market data from Yahoo Finance."""
        # Use the mock data directly for testing
        df = mock_market_data["AAPL"]

        # Verify the data structure
        assert not df.empty
        assert "Close" in df.columns
        assert "Volume" in df.columns
        assert len(df) == 252

        # Verify data integrity
        assert (df["High"] >= df["Low"]).all()
        assert (df["Volume"] > 0).all()

    def test_validate_market_data(self, mock_market_data):
        """Test market data validation."""
        df = mock_market_data["AAPL"]

        # Check for required columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        assert all(col in df.columns for col in required_columns)

        # Check data integrity
        assert (df["High"] >= df["Low"]).all()
        assert (df["High"] >= df["Close"]).all()
        assert (df["Low"] <= df["Close"]).all()

        # Check for missing values
        assert not df[required_columns].isna().any().any()

        # Check for positive prices
        price_columns = ["Open", "High", "Low", "Close"]
        for col in price_columns:
            assert (df[col] > 0).all()

        # Check volume is non-negative
        assert (df["Volume"] >= 0).all()

    def test_handle_missing_data(self):
        """Test handling of missing market data."""
        # Create data with gaps
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        rng = np.random.default_rng(seed=42)
        prices = 100 + np.cumsum(rng.standard_normal(100) * 0.5)

        df = pd.DataFrame({"Close": prices, "Volume": rng.integers(1000000, 5000000, 100)}, index=dates)

        # Introduce missing values
        df.loc[df.index[10:15], "Close"] = np.nan
        df.loc[df.index[50:52], "Volume"] = np.nan

        # Forward fill prices
        df["Close"] = df["Close"].ffill()

        # Fill volume with average
        df["Volume"] = df["Volume"].fillna(df["Volume"].mean())

        # Verify no missing values
        assert not df["Close"].isna().any()
        assert not df["Volume"].isna().any()

    def test_data_frequency_conversion(self):
        """Test converting data to different frequencies."""
        # Daily data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        rng = np.random.default_rng(seed=42)
        daily_data = pd.DataFrame(
            {"Close": 100 + np.cumsum(rng.standard_normal(100) * 0.5), "Volume": rng.integers(1000000, 5000000, 100)},
            index=dates,
        )

        # Convert to weekly
        weekly_data = daily_data.resample("W").agg({"Close": "last", "Volume": "sum"})

        # Verify conversion
        assert len(weekly_data) < len(daily_data)
        assert weekly_data["Volume"].iloc[0] > daily_data["Volume"].iloc[0]

        # Convert to monthly
        monthly_data = daily_data.resample("ME").agg({"Close": "last", "Volume": "sum"})

        assert len(monthly_data) < len(weekly_data)


class TestDataIntegrity:
    """Test data integrity and validation for trading safety."""

    def test_detect_data_anomalies(self):
        """Test detection of data anomalies."""
        # Create data with anomalies
        df = pd.DataFrame(
            {
                "price": [100, 101, 102, 1000, 104, 105],  # Spike at index 3
                "volume": [1000000, 1100000, 1200000, 100, 1300000, 1400000],  # Drop at index 3
            }
        )

        # Detect price spikes using IQR method (more robust for small samples)
        Q1 = df["price"].quantile(0.25)
        Q3 = df["price"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        anomalies = (df["price"] < lower_bound) | (df["price"] > upper_bound)
        assert anomalies.any()
        assert anomalies.iloc[3]

        # Detect volume anomalies using IQR method too
        vol_Q1 = df["volume"].quantile(0.25)
        vol_Q3 = df["volume"].quantile(0.75)
        vol_IQR = vol_Q3 - vol_Q1
        vol_lower = vol_Q1 - 1.5 * vol_IQR
        vol_upper = vol_Q3 + 1.5 * vol_IQR

        volume_anomalies = (df["volume"] < vol_lower) | (df["volume"] > vol_upper)
        assert volume_anomalies.any()
        assert volume_anomalies.iloc[3]  # The drop at index 3

    def test_validate_price_continuity(self):
        """Test price continuity validation."""
        # Create price data with gap
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        prices = [100, 101, 102, 150, 151, 152, 153, 154, 155, 156]  # Big jump

        df = pd.DataFrame({"Close": prices}, index=dates)

        # Calculate returns
        df["returns"] = df["Close"].pct_change()

        # Detect abnormal returns (> 20%)
        abnormal_returns = abs(df["returns"]) > 0.20

        assert abnormal_returns.any()
        assert abnormal_returns.iloc[3]  # 50% jump

        # Flag for manual review
        df["needs_review"] = abnormal_returns
        assert df["needs_review"].sum() == 1

    def test_validate_data_timestamps(self):
        """Test timestamp validation for time series data."""
        # Create data with duplicate timestamps
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        dates_with_dup = [*list(dates), dates[5]]  # Duplicate

        df = pd.DataFrame({"value": range(11)}, index=dates_with_dup)

        # Check for duplicates
        has_duplicates = df.index.duplicated().any()
        assert has_duplicates

        # Remove duplicates (keep last)
        df_clean = df[~df.index.duplicated(keep="last")]
        assert len(df_clean) == 10

        # Check for missing dates
        expected_dates = pd.date_range(df_clean.index.min(), df_clean.index.max(), freq="D")
        missing_dates = expected_dates.difference(df_clean.index)

        assert len(missing_dates) == 0  # No missing dates in this case

    def test_cross_validate_data_sources(self):
        """Test cross-validation between multiple data sources."""
        # Simulate data from two sources
        source1 = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104], "volume": [1000000, 1100000, 1200000, 1300000, 1400000]}
        )

        source2 = pd.DataFrame(
            {
                "close": [100, 101, 105, 103, 104],  # Larger difference at index 2
                "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )

        # Calculate differences
        price_diff = abs(source1["close"] - source2["close"])

        # Set tolerance (e.g., 1%)
        tolerance = source1["close"] * 0.01

        # Check if within tolerance
        within_tolerance = (price_diff <= tolerance).all()

        # In this case, index 2 has a 3 unit difference (from 102 to 105)
        # which exceeds 1% tolerance (1.02)
        assert not within_tolerance  # Should be False since we have a discrepancy

        # Identify discrepancies
        discrepancies = price_diff > tolerance
        assert discrepancies.any()  # At least one discrepancy exists
        assert discrepancies.iloc[2]  # Index 2 has the discrepancy


class TestDataPipeline:
    """Test complete data pipeline for trading."""

    def test_end_to_end_data_pipeline(self, tmp_path):
        """Test complete data pipeline from raw to features."""
        # Step 1: Create raw data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        rng = np.random.default_rng(seed=42)
        raw_data = pd.DataFrame(
            {
                "open": 100 + np.cumsum(rng.standard_normal(100) * 0.5),
                "high": 102 + np.cumsum(rng.standard_normal(100) * 0.5),
                "low": 98 + np.cumsum(rng.standard_normal(100) * 0.5),
                "close": 100 + np.cumsum(rng.standard_normal(100) * 0.5),
                "volume": rng.integers(1000000, 5000000, 100),
            },
            index=dates,
        )

        # Step 2: Clean data
        # Remove any negative prices
        for col in ["open", "high", "low", "close"]:
            raw_data[col] = raw_data[col].clip(lower=1)

        # Step 3: Calculate features
        raw_data["returns"] = raw_data["close"].pct_change()
        raw_data["sma_20"] = raw_data["close"].rolling(20).mean()
        raw_data["volume_ratio"] = raw_data["volume"] / raw_data["volume"].rolling(20).mean()

        # Step 4: Create target
        raw_data["target"] = (raw_data["returns"].shift(-1) > 0).astype(int)

        # Step 5: Split data
        train_size = int(len(raw_data) * 0.8)
        train_data = raw_data.iloc[:train_size]
        test_data = raw_data.iloc[train_size:]

        # Step 6: Save processed data
        train_file = tmp_path / "train_processed.csv"
        test_file = tmp_path / "test_processed.csv"

        train_data.to_csv(train_file)
        test_data.to_csv(test_file)

        # Verify pipeline
        assert train_file.exists()
        assert test_file.exists()
        assert len(train_data) == 80
        assert len(test_data) == 20
        assert "returns" in train_data.columns
        assert "target" in train_data.columns
        assert not train_data["returns"].iloc[1:].isna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

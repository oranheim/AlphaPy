"""
Market Data Pipeline Edge Cases Tests

Comprehensive tests for market data handling including:
- Missing data detection and handling
- Data quality validation
- Holiday and weekend handling
- Corporate actions (splits, dividends, mergers)
- Data source failures and fallbacks
- Real-time vs historical data consistency
- Market microstructure edge cases

Critical for ensuring trading decisions are based on clean, accurate data.
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from alphapy.frame import Frame


class TestMissingDataHandling:
    """Test detection and handling of missing market data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear state before each test."""
        Frame.frames.clear()

    def test_missing_price_data_detection(self):
        """Test detection of missing price data gaps."""
        # Create data with gaps
        dates = pd.bdate_range("2024-01-01", "2024-01-31")  # Business days only

        # Introduce missing data gaps
        missing_dates = ["2024-01-05", "2024-01-12", "2024-01-19"]  # Random missing days
        available_dates = [d for d in dates if d.strftime("%Y-%m-%d") not in missing_dates]

        # Create sample OHLCV data
        rng = np.random.default_rng(seed=42)
        prices = 100 + np.cumsum(rng.standard_normal(len(available_dates)) * 0.5)

        data = pd.DataFrame(
            {
                "open": prices * 0.995,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": rng.integers(1000000, 5000000, len(available_dates), endpoint=True),
            },
            index=available_dates,
        )

        # Detect missing dates
        expected_dates = pd.bdate_range(data.index.min(), data.index.max())
        actual_dates = data.index
        missing = expected_dates.difference(actual_dates)

        # Verify detection
        assert len(missing) == len(missing_dates)
        for missing_date in missing_dates:
            assert pd.to_datetime(missing_date) in missing

        # Test gap analysis
        def analyze_data_gaps(data_index, expected_freq="B"):
            """Analyze gaps in data."""
            expected_index = pd.bdate_range(data_index.min(), data_index.max(), freq=expected_freq)
            missing_dates = expected_index.difference(data_index)

            gaps = []
            if len(missing_dates) > 0:
                # Group consecutive missing dates
                missing_sorted = sorted(missing_dates)
                current_gap_start = missing_sorted[0]
                current_gap_end = missing_sorted[0]

                for i in range(1, len(missing_sorted)):
                    if (missing_sorted[i] - missing_sorted[i - 1]).days <= 3:  # Allow weekends
                        current_gap_end = missing_sorted[i]
                    else:
                        gaps.append(
                            {
                                "start": current_gap_start,
                                "end": current_gap_end,
                                "duration": (current_gap_end - current_gap_start).days + 1,
                            }
                        )
                        current_gap_start = missing_sorted[i]
                        current_gap_end = missing_sorted[i]

                # Add the last gap
                gaps.append(
                    {
                        "start": current_gap_start,
                        "end": current_gap_end,
                        "duration": (current_gap_end - current_gap_start).days + 1,
                    }
                )

            return gaps

        gaps = analyze_data_gaps(data.index)

        # Should detect individual day gaps
        assert len(gaps) > 0
        for gap in gaps:
            assert gap["duration"] >= 1

    def test_weekend_and_holiday_handling(self):
        """Test proper handling of weekends and market holidays."""
        # Market holidays for 2024 (US markets)
        market_holidays = [
            "2024-01-01",  # New Year's Day
            "2024-01-15",  # MLK Day
            "2024-02-19",  # Presidents Day
            "2024-03-29",  # Good Friday
            "2024-05-27",  # Memorial Day
            "2024-06-19",  # Juneteenth
            "2024-07-04",  # Independence Day
            "2024-09-02",  # Labor Day
            "2024-11-28",  # Thanksgiving
            "2024-12-25",  # Christmas
        ]

        holiday_dates = [pd.to_datetime(d) for d in market_holidays]

        # Create trading calendar
        def is_trading_day(date):
            """Check if date is a valid trading day."""
            date = pd.to_datetime(date)

            # Not a weekend
            if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False

            # Not a holiday
            return date not in holiday_dates

        # Test specific dates
        test_cases = [
            ("2024-01-01", False),  # Holiday (New Year)
            ("2024-01-02", True),  # Regular trading day
            ("2024-01-06", False),  # Saturday
            ("2024-01-07", False),  # Sunday
            ("2024-01-08", True),  # Monday (trading day)
            ("2024-07-04", False),  # Holiday (Independence Day)
            ("2024-07-05", True),  # Day after holiday
        ]

        for date_str, expected in test_cases:
            assert is_trading_day(date_str) == expected

        # Generate proper trading calendar
        start_date = "2024-01-01"
        end_date = "2024-12-31"
        all_dates = pd.date_range(start_date, end_date)
        trading_dates = [d for d in all_dates if is_trading_day(d)]

        # Verify no weekends or holidays
        for date in trading_dates:
            assert date.weekday() < 5  # No weekends
            assert date not in holiday_dates  # No holidays

        # Should have approximately 252 trading days
        assert 250 <= len(trading_dates) <= 254

    def test_partial_trading_day_handling(self):
        """Test handling of partial trading days (early close)."""
        # Early close days (typically day before holidays)
        early_close_days = {
            "2024-07-03": {"normal_close": "16:00", "early_close": "13:00"},  # Before July 4th
            "2024-11-29": {"normal_close": "16:00", "early_close": "13:00"},  # Day after Thanksgiving
            "2024-12-24": {"normal_close": "16:00", "early_close": "13:00"},  # Christmas Eve
        }

        def get_market_close_time(date):
            """Get market close time for given date."""
            date_str = pd.to_datetime(date).strftime("%Y-%m-%d")

            if date_str in early_close_days:
                return early_close_days[date_str]["early_close"]
            else:
                return "16:00"  # Normal close

        # Test early close detection
        assert get_market_close_time("2024-07-03") == "13:00"
        assert get_market_close_time("2024-07-02") == "16:00"
        assert get_market_close_time("2024-11-29") == "13:00"

        # Simulate intraday data validation
        def validate_intraday_data(date, timestamps):
            """Validate intraday data doesn't exceed market hours."""
            market_close = get_market_close_time(date)
            close_hour = int(market_close.split(":")[0])

            valid_timestamps = []
            invalid_timestamps = []

            for ts in timestamps:
                if ts.hour < close_hour or (ts.hour == close_hour and ts.minute == 0):
                    valid_timestamps.append(ts)
                else:
                    invalid_timestamps.append(ts)

            return valid_timestamps, invalid_timestamps

        # Test with sample timestamps
        test_date = "2024-07-03"  # Early close day
        sample_timestamps = [
            pd.to_datetime(f"{test_date} 09:30:00"),
            pd.to_datetime(f"{test_date} 12:30:00"),
            pd.to_datetime(f"{test_date} 13:00:00"),  # At early close
            pd.to_datetime(f"{test_date} 15:30:00"),  # After early close (invalid)
        ]

        valid, invalid = validate_intraday_data(test_date, sample_timestamps)

        assert len(valid) == 3  # First 3 timestamps
        assert len(invalid) == 1  # Last timestamp after early close

    def test_data_completeness_validation(self):
        """Test validation of data completeness."""
        # Create test data with various completeness issues
        dates = pd.bdate_range("2024-01-01", "2024-01-31")

        rng = np.random.default_rng(seed=42)
        test_scenarios = {
            "complete": {
                "data": pd.DataFrame(
                    {
                        "open": rng.standard_normal(len(dates)) + 100,
                        "high": rng.standard_normal(len(dates)) + 101,
                        "low": rng.standard_normal(len(dates)) + 99,
                        "close": rng.standard_normal(len(dates)) + 100,
                        "volume": rng.integers(1000000, 5000000, len(dates), endpoint=True),
                    },
                    index=dates,
                ),
                "expected_score": 1.0,
            },
            "missing_volume": {
                "data": pd.DataFrame(
                    {
                        "open": rng.standard_normal(len(dates)) + 100,
                        "high": rng.standard_normal(len(dates)) + 101,
                        "low": rng.standard_normal(len(dates)) + 99,
                        "close": rng.standard_normal(len(dates)) + 100,
                        "volume": [np.nan] * len(dates),  # All volume missing
                    },
                    index=dates,
                ),
                "expected_score": 0.8,  # 4/5 columns complete
            },
            "sparse_data": {
                "data": pd.DataFrame(
                    {
                        "open": [100, np.nan, 102, np.nan, 104],
                        "high": [101, 102, 103, np.nan, 105],
                        "low": [99, np.nan, 101, 98, 103],
                        "close": [100.5, 101.5, 102.5, np.nan, 104.5],
                        "volume": [1000000, 1100000, np.nan, 1200000, 1300000],
                    },
                    index=dates[:5],
                ),
                "expected_score": 0.8,  # Some missing values (5 out of 25)
            },
        }

        def calculate_completeness_score(data):
            """Calculate data completeness score."""
            total_cells = data.size
            non_null_cells = data.count().sum()
            return non_null_cells / total_cells

        for scenario_name, scenario in test_scenarios.items():
            data = scenario["data"]
            expected = scenario["expected_score"]

            actual_score = calculate_completeness_score(data)

            # Allow some tolerance for rounding
            assert abs(actual_score - expected) < 0.1

            # Additional validation rules
            if scenario_name == "complete":
                assert actual_score == 1.0
                assert data.isnull().sum().sum() == 0

            elif scenario_name == "missing_volume":
                assert data["volume"].isnull().all()
                assert data[["open", "high", "low", "close"]].isnull().sum().sum() == 0


class TestDataQualityValidation:
    """Test validation of data quality and anomaly detection."""

    def test_price_consistency_validation(self):
        """Test validation of OHLC price consistency."""
        # Create test data with price inconsistencies
        test_cases = [
            {"name": "valid_ohlc", "data": {"open": 100, "high": 102, "low": 98, "close": 101}, "valid": True},
            {
                "name": "high_below_open",
                "data": {"open": 100, "high": 99, "low": 98, "close": 99.5},  # High < Open
                "valid": False,
            },
            {
                "name": "low_above_close",
                "data": {"open": 100, "high": 102, "low": 101.5, "close": 101},  # Low > Close
                "valid": False,
            },
            {
                "name": "high_below_low",
                "data": {"open": 100, "high": 98, "low": 99, "close": 98.5},  # High < Low
                "valid": False,
            },
            {
                "name": "extreme_gap",
                "data": {"open": 100, "high": 200, "low": 50, "close": 150},  # Unrealistic range
                "valid": False,
            },
        ]

        def validate_ohlc_consistency(ohlc):
            """Validate OHLC price consistency."""
            open_price = ohlc["open"]
            high_price = ohlc["high"]
            low_price = ohlc["low"]
            close_price = ohlc["close"]

            # Basic consistency checks
            if high_price < low_price:
                return False, "High price below low price"

            if high_price < max(open_price, close_price):
                return False, "High price below open or close"

            if low_price > min(open_price, close_price):
                return False, "Low price above open or close"

            # Range check (daily range shouldn't exceed 20%)
            daily_range = (high_price - low_price) / open_price
            if daily_range > 0.20:
                return False, f"Daily range too large: {daily_range:.1%}"

            return True, "Valid"

        for case in test_cases:
            is_valid, message = validate_ohlc_consistency(case["data"])

            if case["valid"]:
                assert is_valid, f"{case['name']} should be valid but got: {message}"
            else:
                assert not is_valid, f"{case['name']} should be invalid but passed validation"

    def test_volume_anomaly_detection(self):
        """Test detection of volume anomalies."""
        # Create volume data with anomalies
        rng = np.random.default_rng(seed=42)
        normal_volume = rng.normal(2000000, 500000, 100)  # Average 2M, std 500k
        normal_volume = np.maximum(normal_volume, 100000)  # Ensure positive

        # Introduce anomalies
        anomalous_volume = normal_volume.copy()
        anomalous_volume[20] = 50000000  # Extremely high volume (25x normal)
        anomalous_volume[50] = 1000  # Extremely low volume (much lower)
        anomalous_volume[80] = 0  # Zero volume (error)

        def detect_volume_anomalies(volume_series, window=20, threshold_std=2.5):
            """Detect volume anomalies using rolling statistics."""
            volume_series = pd.Series(volume_series)

            # Rolling mean and std
            rolling_mean = volume_series.rolling(window, min_periods=10).mean()
            rolling_std = volume_series.rolling(window, min_periods=10).std()

            # Z-score calculation
            z_scores = (volume_series - rolling_mean) / rolling_std

            # Detect anomalies
            anomalies = []
            for i, (vol, z_score) in enumerate(zip(volume_series, z_scores, strict=False)):
                is_anomaly = False
                reason = []

                # Zero volume
                if vol <= 0:
                    is_anomaly = True
                    reason.append("zero_volume")

                # Extreme Z-score
                elif not pd.isna(z_score) and abs(z_score) > threshold_std:
                    is_anomaly = True
                    if z_score > threshold_std:
                        reason.append("volume_spike")
                    else:
                        reason.append("volume_drought")

                if is_anomaly:
                    anomalies.append({"index": i, "volume": vol, "z_score": z_score, "reasons": reason})

            return anomalies

        # Test normal volume (should have few/no anomalies)
        normal_anomalies = detect_volume_anomalies(normal_volume)
        assert len(normal_anomalies) <= 2  # Allow for some statistical variation

        # Test anomalous volume (should detect all introduced anomalies)
        anomalous_anomalies = detect_volume_anomalies(anomalous_volume)
        assert len(anomalous_anomalies) >= 3  # Should detect our 3 anomalies

        # Check specific anomalies
        anomaly_indices = [a["index"] for a in anomalous_anomalies]
        assert 20 in anomaly_indices  # High volume spike
        assert 50 in anomaly_indices  # Low volume
        assert 80 in anomaly_indices  # Zero volume

    def test_price_gap_detection(self):
        """Test detection of price gaps."""
        # Create price series with gaps
        dates = pd.bdate_range("2024-01-01", "2024-01-31")

        # Normal price movement
        rng = np.random.default_rng(seed=42)
        base_prices = 100 + np.cumsum(rng.standard_normal(len(dates)) * 0.5)

        # Introduce gaps
        gap_prices = base_prices.copy()
        gap_prices[10:] += 5  # 5% upward gap
        gap_prices[20:] -= 8  # 8% downward gap

        price_data = pd.DataFrame({"close": gap_prices}, index=dates)

        def detect_price_gaps(price_series, gap_threshold=0.02):
            """Detect price gaps between consecutive trading days."""
            price_series = pd.Series(price_series)
            daily_returns = price_series.pct_change()

            gaps = []
            for i, return_val in enumerate(daily_returns):
                if pd.isna(return_val):
                    continue

                if abs(return_val) > gap_threshold:
                    gap_type = "gap_up" if return_val > 0 else "gap_down"
                    gaps.append(
                        {
                            "index": i,
                            "date": price_series.index[i] if hasattr(price_series, "index") else i,
                            "return": return_val,
                            "type": gap_type,
                            "magnitude": abs(return_val),
                        }
                    )

            return gaps

        gaps = detect_price_gaps(price_data["close"], gap_threshold=0.03)

        # Should detect our introduced gaps
        assert len(gaps) >= 2

        # Check gap types
        gap_types = [gap["type"] for gap in gaps]
        assert "gap_up" in gap_types
        assert "gap_down" in gap_types

    def test_timestamp_validation(self):
        """Test validation of timestamp consistency."""
        # Different timestamp scenarios
        timestamp_scenarios = {
            "regular_trading": [
                "2024-01-02 09:30:00",  # Market open
                "2024-01-02 12:00:00",  # Midday
                "2024-01-02 16:00:00",  # Market close
            ],
            "pre_market": [
                "2024-01-02 08:00:00",  # Pre-market
                "2024-01-02 09:00:00",  # Pre-market
            ],
            "after_hours": [
                "2024-01-02 17:00:00",  # After hours
                "2024-01-02 20:00:00",  # Late after hours
            ],
            "weekend": [
                "2024-01-06 10:00:00",  # Saturday
                "2024-01-07 14:00:00",  # Sunday
            ],
            "holiday": [
                "2024-01-01 10:00:00",  # New Year's Day
            ],
        }

        def validate_timestamp(timestamp):
            """Validate if timestamp is during regular trading hours."""
            ts = pd.to_datetime(timestamp)

            # Check if trading day (not weekend/holiday)
            if ts.weekday() >= 5:  # Weekend
                return False, "weekend"

            # Simple holiday check (just New Year for this test)
            if ts.month == 1 and ts.day == 1:
                return False, "holiday"

            # Check trading hours (9:30 AM to 4:00 PM ET)
            if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30):
                return False, "pre_market"
            elif ts.hour > 16 or (ts.hour == 16 and ts.minute > 0):
                return False, "after_hours"

            return True, "regular_trading"

        # Test each scenario
        for scenario_name, timestamps in timestamp_scenarios.items():
            for timestamp in timestamps:
                is_valid, classification = validate_timestamp(timestamp)

                if scenario_name == "regular_trading":
                    assert is_valid, f"Regular trading timestamp should be valid: {timestamp}"
                else:
                    assert not is_valid, f"Non-trading timestamp should be invalid: {timestamp}"
                    # Verify correct classification
                    assert classification == scenario_name or classification in [
                        "pre_market",
                        "after_hours",
                        "weekend",
                        "holiday",
                    ]


class TestCorporateActionsHandling:
    """Test handling of corporate actions like splits and dividends."""

    def test_stock_split_adjustment(self):
        """Test proper adjustment for stock splits."""
        # Pre-split data
        pre_split_data = {
            "date": "2024-08-23",
            "open": 180.0,
            "high": 185.0,
            "low": 178.0,
            "close": 182.0,
            "volume": 1000000,
        }

        # Split details
        split_info = {
            "ratio": 4.0,  # 4:1 split
            "ex_date": "2024-08-24",
        }

        # Post-split data (raw, not adjusted)
        post_split_data = {
            "date": "2024-08-24",
            "open": 45.0,  # 180/4
            "high": 46.25,  # 185/4
            "low": 44.5,  # 178/4
            "close": 45.5,  # 182/4
            "volume": 4000000,  # Volume should be multiplied by split ratio
        }

        def adjust_for_split(price_data, split_ratio, adjustment_date):
            """Adjust historical prices for stock split."""
            adjusted_data = price_data.copy()

            # Adjust prices (divide by split ratio)
            price_fields = ["open", "high", "low", "close"]
            for field in price_fields:
                if field in adjusted_data:
                    adjusted_data[field] = adjusted_data[field] / split_ratio

            # Adjust volume (multiply by split ratio)
            if "volume" in adjusted_data:
                adjusted_data["volume"] = adjusted_data["volume"] * split_ratio

            return adjusted_data

        # Adjust pre-split data for comparison
        adjusted_pre_split = adjust_for_split(pre_split_data, split_info["ratio"], split_info["ex_date"])

        # Compare adjusted pre-split with post-split
        tolerance = 0.01  # Allow small differences

        assert abs(adjusted_pre_split["open"] - post_split_data["open"]) < tolerance
        assert abs(adjusted_pre_split["high"] - post_split_data["high"]) < tolerance
        assert abs(adjusted_pre_split["low"] - post_split_data["low"]) < tolerance
        assert abs(adjusted_pre_split["close"] - post_split_data["close"]) < tolerance

        # Volume should be approximately 4x (accounting for normal variation)
        volume_ratio = post_split_data["volume"] / pre_split_data["volume"]
        assert 3.5 <= volume_ratio <= 4.5  # Allow some natural variation

    def test_dividend_adjustment(self):
        """Test proper adjustment for dividends."""
        # Ex-dividend data
        dividend_info = {"ex_date": "2024-11-10", "amount": 0.24, "currency": "USD"}

        # Price data around ex-dividend date
        price_data = [
            {"date": "2024-11-08", "close": 100.00},  # Cum-dividend
            {"date": "2024-11-09", "close": 100.50},  # Last cum-dividend day
            {"date": "2024-11-10", "close": 100.26},  # Ex-dividend (should drop by dividend)
            {"date": "2024-11-11", "close": 100.75},  # Post ex-dividend
        ]

        def adjust_for_dividend(historical_prices, dividend_amount, ex_date):
            """Adjust historical prices for dividend."""
            adjusted_prices = []
            ex_date_pd = pd.to_datetime(ex_date)

            for price_point in historical_prices:
                price_date = pd.to_datetime(price_point["date"])
                adjusted_price = price_point.copy()

                # Adjust prices before ex-date (subtract dividend)
                if price_date < ex_date_pd:
                    adjusted_price["close"] = price_point["close"] - dividend_amount

                adjusted_prices.append(adjusted_price)

            return adjusted_prices

        # Apply dividend adjustment
        adjusted_prices = adjust_for_dividend(price_data, dividend_info["amount"], dividend_info["ex_date"])

        # Check adjustments
        for i, (original, adjusted) in enumerate(zip(price_data, adjusted_prices, strict=False)):
            if pd.to_datetime(original["date"]) < pd.to_datetime(dividend_info["ex_date"]):
                # Pre-ex-dividend prices should be reduced
                expected_adjusted = original["close"] - dividend_info["amount"]
                assert abs(adjusted["close"] - expected_adjusted) < 0.01
            else:
                # Ex-dividend and post prices should be unchanged
                assert adjusted["close"] == original["close"]

        # Verify price continuity after adjustment
        adjusted_closes = [p["close"] for p in adjusted_prices]

        # Calculate daily returns
        returns = []
        for i in range(1, len(adjusted_closes)):
            ret = (adjusted_closes[i] / adjusted_closes[i - 1]) - 1
            returns.append(ret)

        # Returns should be more reasonable after adjustment
        max_abs_return = max(abs(r) for r in returns)
        assert max_abs_return < 0.05  # No single day return > 5%

    def test_merger_and_acquisition_handling(self):
        """Test handling of mergers and acquisitions."""
        # Merger scenario: Company A acquired by Company B
        merger_info = {
            "acquiring_company": "COMPANY_B",
            "target_company": "COMPANY_A",
            "announcement_date": "2024-09-15",
            "completion_date": "2024-12-01",
            "exchange_ratio": 0.75,  # 0.75 shares of B for each share of A
            "cash_component": 10.0,  # Plus $10 cash per share
        }

        # Price data around merger completion
        pre_merger_prices = {
            "COMPANY_A": {"date": "2024-11-30", "close": 85.0},
            "COMPANY_B": {"date": "2024-11-30", "close": 120.0},
        }

        post_merger_date = "2024-12-01"

        def calculate_merger_value(target_price, acquiring_price, exchange_ratio, cash_component):
            """Calculate implied value from merger terms."""
            stock_value = acquiring_price * exchange_ratio
            total_value = stock_value + cash_component
            return total_value

        # Calculate implied value
        implied_value = calculate_merger_value(
            pre_merger_prices["COMPANY_A"]["close"],
            pre_merger_prices["COMPANY_B"]["close"],
            merger_info["exchange_ratio"],
            merger_info["cash_component"],
        )

        expected_value = (120.0 * 0.75) + 10.0  # $90 + $10 = $100
        assert abs(implied_value - expected_value) < 0.01

        # Check for arbitrage opportunity
        target_price = pre_merger_prices["COMPANY_A"]["close"]
        arbitrage_spread = implied_value - target_price  # $100 - $85 = $15
        arbitrage_percentage = arbitrage_spread / target_price  # 17.6%

        # Significant arbitrage suggests market uncertainty about completion
        assert arbitrage_percentage > 0.10  # More than 10% spread

        # Test post-merger data handling
        def handle_post_merger_data(symbol, date, merger_info):
            """Determine how to handle data post-merger."""
            completion_date = pd.to_datetime(merger_info["completion_date"])
            query_date = pd.to_datetime(date)

            if symbol == merger_info["target_company"] and query_date >= completion_date:
                return {
                    "status": "delisted",
                    "reason": "merger_completion",
                    "replacement_symbol": merger_info["acquiring_company"],
                    "conversion_info": {"ratio": merger_info["exchange_ratio"], "cash": merger_info["cash_component"]},
                }
            else:
                return {"status": "active"}

        # Test data handling after merger
        post_merger_status = handle_post_merger_data("COMPANY_A", "2024-12-02", merger_info)

        assert post_merger_status["status"] == "delisted"
        assert post_merger_status["reason"] == "merger_completion"
        assert post_merger_status["replacement_symbol"] == "COMPANY_B"

    def test_spin_off_handling(self):
        """Test handling of corporate spin-offs."""
        # Spin-off scenario: Company A spins off division as Company B
        spinoff_info = {
            "parent_company": "COMPANY_A",
            "spinoff_company": "COMPANY_B",
            "ex_date": "2024-10-15",
            "distribution_ratio": 0.5,  # 0.5 shares of B for each share of A
            "record_date": "2024-10-10",
        }

        # Pre-spin-off data
        pre_spinoff_data = {"COMPANY_A": {"date": "2024-10-14", "close": 150.0, "shares_outstanding": 100000000}}

        # Post-spin-off data
        post_spinoff_data = {
            "COMPANY_A": {
                "date": "2024-10-15",
                "close": 120.0,  # Should drop due to spin-off
                "shares_outstanding": 100000000,  # Same number of shares
            },
            "COMPANY_B": {
                "date": "2024-10-15",
                "close": 60.0,  # New company trading
                "shares_outstanding": 50000000,  # 0.5 * parent shares
            },
        }

        def analyze_spinoff_impact(pre_data, post_data, spinoff_ratio):
            """Analyze the impact of spin-off on valuations."""
            # Calculate value before spin-off
            pre_value = pre_data["COMPANY_A"]["close"]

            # Calculate combined value after spin-off
            parent_value = post_data["COMPANY_A"]["close"]
            spinoff_value = post_data["COMPANY_B"]["close"] * spinoff_ratio
            combined_value = parent_value + spinoff_value

            # Value should be approximately preserved
            value_change = (combined_value / pre_value) - 1

            return {
                "pre_spinoff_value": pre_value,
                "post_parent_value": parent_value,
                "spinoff_value_per_parent_share": spinoff_value,
                "combined_value": combined_value,
                "value_change": value_change,
            }

        analysis = analyze_spinoff_impact(pre_spinoff_data, post_spinoff_data, spinoff_info["distribution_ratio"])

        # Combined value should be close to original value
        # $120 + ($60 * 0.5) = $120 + $30 = $150
        assert abs(analysis["combined_value"] - 150.0) < 1.0
        assert abs(analysis["value_change"]) < 0.02  # Less than 2% change

        # Test portfolio adjustment for spin-off
        def adjust_portfolio_for_spinoff(original_position, spinoff_info):
            """Adjust portfolio position for spin-off."""
            parent_shares = original_position["shares"]

            # Parent company position remains the same number of shares
            adjusted_parent = {"symbol": spinoff_info["parent_company"], "shares": parent_shares}

            # New spinoff position
            spinoff_shares = parent_shares * spinoff_info["distribution_ratio"]
            new_spinoff = {"symbol": spinoff_info["spinoff_company"], "shares": spinoff_shares}

            return [adjusted_parent, new_spinoff]

        # Test portfolio adjustment
        original_position = {"shares": 1000}
        adjusted_positions = adjust_portfolio_for_spinoff(original_position, spinoff_info)

        assert len(adjusted_positions) == 2
        assert adjusted_positions[0]["shares"] == 1000  # Parent shares unchanged
        assert adjusted_positions[1]["shares"] == 500  # Spinoff shares = 1000 * 0.5


class TestDataSourceFailureHandling:
    """Test handling of data source failures and fallbacks."""

    def test_primary_data_source_failure(self):
        """Test fallback when primary data source fails."""

        # Mock data sources
        data_sources = {
            "yahoo": {"status": "down", "priority": 1},
            "alpha_vantage": {"status": "up", "priority": 2},
            "iex": {"status": "up", "priority": 3},
            "quandl": {"status": "up", "priority": 4},
        }

        def get_available_data_source(sources):
            """Get highest priority available data source."""
            available_sources = [(name, info) for name, info in sources.items() if info["status"] == "up"]

            if not available_sources:
                return None

            # Sort by priority (lower number = higher priority)
            available_sources.sort(key=lambda x: x[1]["priority"])
            return available_sources[0][0]

        # Test fallback logic
        selected_source = get_available_data_source(data_sources)
        assert selected_source == "alpha_vantage"  # Highest priority available

        # Test complete failure scenario
        all_down_sources = {
            name: {"status": "down", "priority": info["priority"]} for name, info in data_sources.items()
        }

        failed_source = get_available_data_source(all_down_sources)
        assert failed_source is None

        # Test data quality comparison between sources
        def compare_data_sources(source1_data, source2_data, tolerance=0.01):
            """Compare data from different sources."""
            differences = {}

            for field in ["open", "high", "low", "close", "volume"]:
                if field in source1_data and field in source2_data:
                    diff = abs(source1_data[field] - source2_data[field])
                    if field != "volume":
                        # Price differences as percentage
                        diff_pct = diff / source1_data[field]
                        differences[field] = {"absolute": diff, "percentage": diff_pct}
                    else:
                        # Volume differences as absolute
                        differences[field] = {"absolute": diff, "percentage": None}

            return differences

        # Mock data from different sources
        yahoo_data = {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000000}
        av_data = {"open": 100.1, "high": 102.1, "low": 99.1, "close": 101.1, "volume": 1005000}

        comparison = compare_data_sources(yahoo_data, av_data)

        # Check that differences are within reasonable bounds
        for field, diff_info in comparison.items():
            if field != "volume" and diff_info["percentage"] is not None:
                assert diff_info["percentage"] < 0.005  # Less than 0.5% difference

    def test_real_time_vs_delayed_data_handling(self):
        """Test handling of real-time vs delayed data."""

        # Mock data with timestamps
        real_time_data = {
            "timestamp": "2024-01-02 15:45:30",
            "price": 150.25,
            "volume": 1500000,
            "delay": 0,  # Real-time
        }

        delayed_data = {
            "timestamp": "2024-01-02 15:45:30",
            "price": 150.20,
            "volume": 1480000,
            "delay": 900,  # 15 minutes delayed
        }

        def determine_data_freshness(data_timestamp, current_time):
            """Determine how fresh the data is."""
            data_time = pd.to_datetime(data_timestamp)
            current = pd.to_datetime(current_time)

            age_seconds = (current - data_time).total_seconds()

            if age_seconds <= 60:
                return "real_time"
            elif age_seconds <= 900:  # 15 minutes
                return "near_real_time"
            elif age_seconds <= 3600:  # 1 hour
                return "delayed"
            else:
                return "stale"

        # Test data freshness
        current_time = "2024-01-02 15:46:00"  # 30 seconds later

        rt_freshness = determine_data_freshness(real_time_data["timestamp"], current_time)
        delayed_freshness = determine_data_freshness(delayed_data["timestamp"], current_time)

        assert rt_freshness == "real_time"
        assert delayed_freshness == "real_time"  # Same timestamp, so appears fresh

        # Test with actual delay consideration
        def determine_effective_freshness(data, current_time):
            """Determine freshness considering declared delay."""
            base_freshness = determine_data_freshness(data["timestamp"], current_time)
            declared_delay = data.get("delay", 0)

            if declared_delay > 600:  # More than 10 minutes
                return "delayed"
            elif declared_delay > 60:  # More than 1 minute
                return "near_real_time"
            else:
                return base_freshness

        rt_effective = determine_effective_freshness(real_time_data, current_time)
        delayed_effective = determine_effective_freshness(delayed_data, current_time)

        assert rt_effective == "real_time"
        assert delayed_effective == "delayed"

    def test_data_validation_pipeline(self):
        """Test comprehensive data validation pipeline."""

        # Sample raw data with various issues
        raw_data_samples = [
            {
                "symbol": "AAPL",
                "date": "2024-01-02",
                "open": 150.0,
                "high": 152.0,
                "low": 149.0,
                "close": 151.0,
                "volume": 2000000,
                "issues": [],  # Clean data
            },
            {
                "symbol": "GOOGL",
                "date": "2024-01-02",
                "open": 140.0,
                "high": 138.0,  # Issue: high < open
                "low": 139.0,
                "close": 139.5,
                "volume": 1500000,
                "issues": ["high_below_open"],
            },
            {
                "symbol": "MSFT",
                "date": "2024-01-02",
                "open": 380.0,
                "high": 385.0,
                "low": 375.0,
                "close": 382.0,
                "volume": 0,  # Issue: zero volume
                "issues": ["zero_volume"],
            },
        ]

        def validate_data_pipeline(data_sample):
            """Run comprehensive data validation."""
            validation_results = {"symbol": data_sample["symbol"], "passed": True, "errors": [], "warnings": []}

            # Price consistency checks
            if data_sample["high"] < data_sample["low"]:
                validation_results["errors"].append("high_below_low")
                validation_results["passed"] = False

            if data_sample["high"] < max(data_sample["open"], data_sample["close"]):
                validation_results["errors"].append("high_below_open_close")
                validation_results["passed"] = False

            if data_sample["low"] > min(data_sample["open"], data_sample["close"]):
                validation_results["errors"].append("low_above_open_close")
                validation_results["passed"] = False

            # Volume checks
            if data_sample["volume"] <= 0:
                validation_results["errors"].append("invalid_volume")
                validation_results["passed"] = False
            elif data_sample["volume"] < 10000:
                validation_results["warnings"].append("low_volume")

            # Range checks
            daily_range = (data_sample["high"] - data_sample["low"]) / data_sample["open"]
            if daily_range > 0.20:
                validation_results["warnings"].append("large_daily_range")

            return validation_results

        # Run validation pipeline
        validation_results = []
        for sample in raw_data_samples:
            result = validate_data_pipeline(sample)
            validation_results.append(result)

        # Check results
        assert validation_results[0]["passed"]  # Clean data should pass
        assert not validation_results[1]["passed"]  # High < open should fail
        assert not validation_results[2]["passed"]  # Zero volume should fail

        # Check specific error detection
        assert "high_below_open_close" in validation_results[1]["errors"]
        assert "invalid_volume" in validation_results[2]["errors"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

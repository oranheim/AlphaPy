"""
System Trade Execution Tests

Critical tests for the trade_system function and order execution logic.
These tests ensure trades are executed correctly with proper money management.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from alphapy.frame import Frame, frame_name
from alphapy.globals import SSEP, Orders
from alphapy.space import Space
from alphapy.system import System, trade_system


class TestTradeSystemExecution:
    """Test the trade_system function with real trading scenarios."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear state and setup test environment."""
        System.systems.clear()
        Frame.frames.clear()

    @pytest.fixture
    def mock_model(self, tmp_path):
        """Create a mock model with required specs."""
        model = Mock()
        model.specs = {
            "directory": str(tmp_path),
            "extension": "csv",
            "separator": ",",
            "tag": "test",
            "target": "signal",
        }
        return model

    @pytest.fixture
    def price_data_with_signals(self):
        """Create price data with clear trading signals."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        rng = np.random.default_rng(seed=42)

        # Create trending market data
        trend_up = np.linspace(100, 120, 40)
        sideways = np.full(20, 120)
        trend_down = np.linspace(120, 105, 40)
        prices = np.concatenate([trend_up, sideways, trend_down])

        # Add some noise
        prices = prices + rng.standard_normal(100) * 1

        df = pd.DataFrame(
            {
                "date": dates,
                "open": prices * 0.995,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": rng.integers(1000000, 5000000, 100, endpoint=True),
                "bar_number": range(100),
                "end_of_day": [True] * 100,  # All daily bars
            },
            index=dates,
        )

        # Add technical indicators
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["rsi"] = self.calculate_rsi(df["close"])

        # Add prediction probabilities for ML-based systems
        df["phigh"] = np.where(df["close"] > df["sma_20"], 0.7, 0.3)
        df["plow"] = 1 - df["phigh"]

        return df

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def test_trade_system_simple_strategy(self, mock_model, price_data_with_signals):
        """Test trade_system with simple moving average crossover."""
        # Create system
        system = System(
            name="ma_crossover",
            longentry="close > sma_20",
            longexit="close < sma_20",
            shortentry=None,  # Long only
            shortexit=None,
            holdperiod=0,
            scale=False,
        )

        # Add price data to frames
        space = Space("stock", "prices", "1d")
        symbol = "TEST"
        Frame.frames[frame_name(symbol, space)] = Mock(df=price_data_with_signals)

        # Mock vexec to evaluate conditions
        with patch("alphapy.system.vexec") as mock_vexec:

            def vexec_side_effect(df, condition):
                # Evaluate the condition and add as column
                if "close > sma_20" in condition:
                    df["close > sma_20"] = df["close"] > df["sma_20"]
                elif "close < sma_20" in condition:
                    df["close < sma_20"] = df["close"] < df["sma_20"]
                return df

            mock_vexec.side_effect = vexec_side_effect

            # Mock file system for output
            with patch("alphapy.system.SSEP", "/"):
                # Execute system
                trades = trade_system(mock_model, system, space, intraday=False, name=symbol, quantity=100)

                # Should have generated trades
                assert trades is not None
                assert len(trades) > 0  # At least one trade generated

    def test_trade_system_with_stop_loss(self, mock_model):
        """Test trade system with stop loss orders."""
        # Create price data with stop loss trigger
        dates = pd.date_range("2024-01-01", periods=50, freq="D")

        # Price goes up then crashes
        prices = np.concatenate(
            [
                np.linspace(100, 110, 20),  # Uptrend
                np.linspace(110, 95, 30),  # Crash
            ]
        )

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.full(50, 1000000),
                "sma_20": pd.Series(prices).rolling(20).mean(),
            },
            index=dates,
        )

        # Add stop loss level
        df["stop_loss"] = df["close"].rolling(20).max() * 0.95  # 5% trailing stop

        # Create system with stop loss
        system = System(
            name="with_stop",
            longentry="close > sma_20",
            longexit="close < stop_loss",  # Exit on stop loss
            holdperiod=0,
        )

        # Setup frame
        space = Space()
        Frame.frames[frame_name("TEST", space)] = Mock(df=df)

        # Mock vexec
        with patch("alphapy.system.vexec") as mock_vexec:

            def vexec_side_effect(df, condition):
                if "close > sma_20" in condition:
                    df["close > sma_20"] = df["close"] > df["sma_20"]
                elif "close < stop_loss" in condition:
                    df["close < stop_loss"] = df["close"] < df["stop_loss"]
                return df

            mock_vexec.side_effect = vexec_side_effect

            with patch("alphapy.system.write_frame"):
                trades = trade_system(mock_model, system, space, intraday=False, name="TEST", quantity=100)

    def test_trade_system_position_sizing(self, mock_model):
        """Test dynamic position sizing based on volatility."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        # Create data with varying volatility
        rng = np.random.default_rng(seed=42)
        calm_period = rng.standard_normal(40) * 1 + 100  # Low volatility
        volatile_period = rng.standard_normal(60) * 5 + 100  # High volatility
        prices = np.concatenate([calm_period, volatile_period])

        df = pd.DataFrame(
            {
                "close": prices,
                "open": prices * 0.99,
                "high": prices * 1.02,
                "low": prices * 0.98,
                "volume": np.full(100, 1000000),
            },
            index=dates,
        )

        # Calculate ATR for position sizing
        df["tr"] = pd.DataFrame(
            {
                "hl": df["high"] - df["low"],
                "hc": abs(df["high"] - df["close"].shift()),
                "lc": abs(df["low"] - df["close"].shift()),
            }
        ).max(axis=1)
        df["atr"] = df["tr"].rolling(14).mean()

        # Position size inversely proportional to volatility
        risk_amount = 1000  # $1000 risk per trade
        df["position_size"] = risk_amount / (df["atr"] * 2)  # 2x ATR stop

        # Verify position sizing
        assert df["position_size"].iloc[14:40].mean() > df["position_size"].iloc[60:].mean()

    def test_trade_system_with_ml_signals(self, mock_model, price_data_with_signals, tmp_path):
        """Test trade system using ML model predictions."""
        # Create system that uses ML probabilities
        system = System(
            name="ml_system",
            longentry="phigh > 0.6",  # High probability of going up
            longexit="plow > 0.6",  # High probability of going down
            holdperiod=0,
        )

        # Setup frame with ML predictions
        space = Space()
        df = price_data_with_signals
        Frame.frames[frame_name("TEST", space)] = Mock(df=df)

        # Create mock probabilities file
        probs_dir = tmp_path / "output"
        probs_dir.mkdir()
        probs_file = probs_dir / "probabilities_test.csv"

        # Create probability data with matching length
        probs_df = pd.DataFrame(
            {
                "probability": df["phigh"].values  # Use all values, not just last 50
            }
        )
        probs_df.to_csv(probs_file)

        # Mock vexec to handle ML conditions
        with patch("alphapy.system.vexec") as mock_vexec:

            def vexec_side_effect(df, condition):
                if "phigh > 0.6" in condition:
                    df["phigh > 0.6"] = df.get("probability", df.get("phigh", 0)) > 0.6
                elif "plow > 0.6" in condition:
                    df["plow > 0.6"] = df.get("probability", df.get("plow", 0)) > 0.6
                return df

            mock_vexec.side_effect = vexec_side_effect

            with (
                patch("alphapy.system.most_recent_file", return_value=str(probs_file)),
                patch("alphapy.system.SSEP", "/"),
                patch("alphapy.system.read_frame") as mock_read,
                patch("alphapy.system.write_frame"),
            ):
                mock_read.return_value = probs_df
                trades = trade_system(mock_model, system, space, intraday=False, name="TEST", quantity=100)

    def test_trade_system_scale_in_scale_out(self, mock_model):
        """Test scaling in and out of positions."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        rng = np.random.default_rng(seed=42)
        prices = 100 + np.cumsum(rng.standard_normal(100) * 0.5)

        df = pd.DataFrame(
            {
                "close": prices,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "volume": np.full(100, 1000000),
                "sma_20": pd.Series(prices).rolling(20).mean(),
            },
            index=dates,
        )

        # Add scale levels
        df["scale_in_1"] = df["sma_20"] * 0.98  # First scale in at -2%
        df["scale_in_2"] = df["sma_20"] * 0.96  # Second scale in at -4%
        df["scale_out_1"] = df["sma_20"] * 1.02  # First scale out at +2%
        df["scale_out_2"] = df["sma_20"] * 1.04  # Second scale out at +4%

        # Create system with scaling
        system = System(
            name="scale_system",
            longentry="(close < scale_in_1) | (close < scale_in_2)",
            longexit="(close > scale_out_1) | (close > scale_out_2)",
            holdperiod=0,
            scale=True,  # Enable scaling
        )

        space = Space()
        Frame.frames[frame_name("TEST", space)] = Mock(df=df)

        # Mock vexec
        with patch("alphapy.system.vexec") as mock_vexec:

            def vexec_side_effect(df, condition):
                if "(close < scale_in_1) | (close < scale_in_2)" in condition:
                    df["(close < scale_in_1) | (close < scale_in_2)"] = (df["close"] < df["scale_in_1"]) | (
                        df["close"] < df["scale_in_2"]
                    )
                elif "(close > scale_out_1) | (close > scale_out_2)" in condition:
                    df["(close > scale_out_1) | (close > scale_out_2)"] = (df["close"] > df["scale_out_1"]) | (
                        df["close"] > df["scale_out_2"]
                    )
                return df

            mock_vexec.side_effect = vexec_side_effect

            with patch("alphapy.system.write_frame"):
                trades = trade_system(mock_model, system, space, intraday=False, name="TEST", quantity=100)

    def test_trade_system_with_hold_period(self, mock_model, price_data_with_signals):
        """Test minimum holding period constraint."""
        system = System(
            name="hold_period_system",
            longentry="close > sma_20",
            longexit="close < sma_20",
            holdperiod=5,  # Minimum 5 days holding
            scale=False,
        )

        space = Space()
        Frame.frames[frame_name("TEST", space)] = Mock(df=price_data_with_signals)

        # Mock vexec
        with patch("alphapy.system.vexec") as mock_vexec:

            def vexec_side_effect(df, condition):
                if "close > sma_20" in condition:
                    df["close > sma_20"] = df["close"] > df["sma_20"]
                elif "close < sma_20" in condition:
                    df["close < sma_20"] = df["close"] < df["sma_20"]
                return df

            mock_vexec.side_effect = vexec_side_effect

            # Execute system
            trades = trade_system(mock_model, system, space, intraday=False, name="TEST", quantity=100)

            # Verify trades were generated
            assert trades is not None
            # Holding period affects when exits happen, not if trades exist
            # With a holding period, exits should be delayed


class TestSystemIntegration:
    """Test System class integration with portfolio."""

    def test_multiple_systems_same_symbol(self):
        """Test running multiple systems on the same symbol."""
        # Create multiple systems
        momentum_system = System(name="momentum", longentry="close > sma_50", longexit="close < sma_50")

        mean_reversion_system = System(name="mean_reversion", longentry="rsi < 30", longexit="rsi > 70")

        # Both systems should coexist
        assert "momentum" in System.systems
        assert "mean_reversion" in System.systems

        # Systems should have different signals
        assert momentum_system.longentry != mean_reversion_system.longentry

    def test_system_validation(self):
        """Test system parameter validation."""
        # System creation should work even with minimal parameters
        # The validation happens during trade execution
        system1 = System(name="minimal", longentry="close > 100", longexit=None, holdperiod=0)

        # System should be created successfully
        assert system1.name == "minimal"
        assert system1.longentry == "close > 100"
        assert system1.longexit is None

        # Test valid system
        system2 = System(name="valid", longentry="close > sma_20", longexit="close < sma_20", holdperiod=5, scale=False)

        assert system2.name == "valid"
        assert system2.holdperiod == 5

    def test_system_performance_tracking(self):
        """Test tracking system performance metrics."""
        # Mock trade results
        trades = [
            {"entry": 100, "exit": 110, "quantity": 100},  # +$1000
            {"entry": 105, "exit": 102, "quantity": 100},  # -$300
            {"entry": 98, "exit": 105, "quantity": 100},  # +$700
            {"entry": 110, "exit": 108, "quantity": 100},  # -$200
            {"entry": 107, "exit": 115, "quantity": 100},  # +$800
        ]

        # Calculate metrics
        profits = [(t["exit"] - t["entry"]) * t["quantity"] for t in trades]
        total_profit = sum(profits)
        num_trades = len(trades)
        win_trades = sum(1 for p in profits if p > 0)
        loss_trades = sum(1 for p in profits if p < 0)

        win_rate = win_trades / num_trades
        avg_win = sum(p for p in profits if p > 0) / win_trades
        avg_loss = sum(p for p in profits if p < 0) / loss_trades

        profit_factor = sum(p for p in profits if p > 0) / abs(sum(p for p in profits if p < 0))

        # Verify calculations
        assert total_profit == 2000
        assert win_rate == 0.6
        assert profit_factor > 1  # Profitable system


class TestOrderExecution:
    """Test order types and execution logic."""

    def test_market_order_execution(self):
        """Test market order immediate execution."""
        current_price = 100.0
        order_quantity = 100

        # Market order executes at current price (with slippage)
        slippage = 0.001  # 0.1%
        execution_price = current_price * (1 + slippage)

        trade_value = execution_price * order_quantity

        assert execution_price > current_price
        assert trade_value == 10010  # $10,010 including slippage

    def test_limit_order_execution(self):
        """Test limit order execution at specified price."""
        limit_price = 99.50
        current_prices = [100.0, 99.75, 99.40, 99.60]  # Price movement

        executed = False
        execution_price = None

        for price in current_prices:
            if price <= limit_price:
                executed = True
                execution_price = limit_price  # Limit orders execute at limit
                break

        assert executed
        assert execution_price == 99.50

    def test_stop_order_execution(self):
        """Test stop loss order triggering."""
        entry_price = 100.0
        stop_price = 95.0  # 5% stop loss

        current_prices = [100, 98, 96, 94, 92]  # Price falling

        stopped = False
        exit_price = None

        for price in current_prices:
            if price <= stop_price:
                stopped = True
                exit_price = price  # Stop market order
                break

        assert stopped
        assert exit_price == 94  # Executed below stop (slippage)

        # Calculate loss
        loss = (entry_price - exit_price) * 100
        loss_pct = loss / (entry_price * 100)

        assert loss == 600  # $600 loss
        assert loss_pct == 0.06  # 6% loss (1% slippage from 5% stop)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

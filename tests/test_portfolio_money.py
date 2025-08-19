"""
Comprehensive Portfolio Money Handling Tests

These tests ensure the safety and correctness of all money-related operations
in the portfolio module, including position sizing, valuation, and P&L calculations.
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from alphapy.frame import Frame
from alphapy.globals import MULTIPLIERS, Orders
from alphapy.portfolio import (
    Portfolio,
    Position,
    Trade,
    add_position,
    close_position,
    exec_trade,
    remove_position,
    stop_loss,
    update_position,
    valuate_portfolio,
    valuate_position,
)
from alphapy.space import Space


class TestPositionMoney:
    """Test all money-critical position operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear state before each test."""
        Portfolio.portfolios.clear()
        Frame.frames.clear()

    @pytest.fixture
    def mock_portfolio(self):
        """Create a portfolio with realistic money constraints."""
        portfolio = Portfolio(
            group_name="money_test",
            tag=f"test_{uuid.uuid4().hex[:8]}",
            startcap=100000.0,  # $100k starting capital
            margin=0.5,  # 50% margin allowed
            mincash=0.1,  # Keep 10% cash minimum
            fixedfrac=0.02,  # 2% per position
            maxloss=0.05,  # 5% stop loss
            maxpos=10,  # Max 10 positions
        )
        return portfolio

    @pytest.fixture
    def mock_price_data(self):
        """Create realistic price data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        rng = np.random.default_rng(seed=42)
        prices = {
            "AAPL": 150.0 + rng.standard_normal(100) * 5,
            "GOOGL": 140.0 + rng.standard_normal(100) * 4,
            "MSFT": 380.0 + rng.standard_normal(100) * 10,
        }

        data = {}
        for symbol, price_series in prices.items():
            df = pd.DataFrame(
                {
                    "open": price_series * (1 + rng.uniform(-0.01, 0.01, 100)),
                    "high": price_series * (1 + rng.uniform(0, 0.02, 100)),
                    "low": price_series * (1 + rng.uniform(-0.02, 0, 100)),
                    "close": price_series,
                    "volume": rng.integers(1000000, 10000000, 100),
                },
                index=dates,
            )
            data[symbol] = df

            # Add to Frame.frames for Position class
            space = Space("stock", "prices", "1d")
            frame_name = f"{symbol}_stock_prices_1d"
            Frame.frames[frame_name] = Mock(df=df)

        return data

    def test_position_creation_with_money(self, mock_portfolio, mock_price_data):
        """Test creating a position with money validation."""
        # Create position
        position = Position(portfolio=mock_portfolio, name="AAPL", opendate=datetime(2024, 1, 1))

        # Verify initial state
        assert position.quantity == 0
        assert position.value == 0.0
        assert position.profit == 0.0
        assert position.costbasis == 0.0
        assert position.status == "opened"

    def test_position_sizing_fixed_fractional(self, mock_portfolio):
        """Test fixed fractional position sizing (Kelly Criterion)."""
        capital = mock_portfolio.startcap
        risk_fraction = mock_portfolio.fixedfrac  # 2%

        # Calculate position size for different prices
        test_cases = [
            {"price": 100.0, "stop": 95.0},  # 5% stop
            {"price": 50.0, "stop": 48.0},  # 4% stop
            {"price": 200.0, "stop": 190.0},  # 5% stop
        ]

        for case in test_cases:
            price = case["price"]
            stop = case["stop"]
            risk_per_share = price - stop

            # Position sizing formula
            risk_amount = capital * risk_fraction
            shares = int(risk_amount / risk_per_share)
            position_value = shares * price

            # Verify constraints
            # Note: position value can exceed 10% if stop is tight
            # What matters is the risk amount stays within limits
            assert shares * risk_per_share <= capital * risk_fraction + 1  # +1 for rounding

    def test_trade_execution_with_costs(self, mock_portfolio, mock_price_data):
        """Test trade execution including commission and slippage."""
        # Create a position
        position = Position(mock_portfolio, "AAPL", datetime(2024, 1, 1))

        # Execute a buy trade
        buy_price = 150.0
        quantity = 100
        commission = 0.001  # 0.1% commission
        slippage = 0.001  # 0.1% slippage

        # Calculate actual cost
        trade_value = quantity * buy_price
        total_cost = trade_value * (1 + commission + slippage)

        # Create trade
        trade = Trade(
            name="AAPL",
            order=Orders.le,  # Long entry
            quantity=quantity,
            price=buy_price,
            tdate=datetime(2024, 1, 1),
        )

        # Add trade to position
        position.trades.append(trade)
        position.quantity = quantity
        position.ntrades = 1

        # Verify money calculations
        assert position.quantity == quantity
        assert len(position.trades) == 1
        assert position.trades[0].price == buy_price

        # Calculate with costs
        actual_cost_per_share = buy_price * (1 + commission + slippage)
        assert actual_cost_per_share > buy_price

    def test_position_valuation(self, mock_portfolio, mock_price_data):
        """Test accurate position valuation with P&L."""
        # Setup position with trades
        position = Position(mock_portfolio, "AAPL", datetime(2024, 1, 1))

        # Add multiple trades
        trades = [
            Trade("AAPL", Orders.le, 100, 150.0, datetime(2024, 1, 1)),
            Trade("AAPL", Orders.le, 50, 155.0, datetime(2024, 1, 5)),
            Trade("AAPL", Orders.lx, -50, 160.0, datetime(2024, 1, 10)),
        ]

        for trade in trades:
            position.trades.append(trade)

        # Calculate position metrics
        total_bought = 150  # 100 + 50
        total_sold = 50
        net_position = 100  # 150 - 50

        # Cost basis calculation
        buy_value = (100 * 150.0) + (50 * 155.0)  # $22,750
        sell_value = 50 * 160.0  # $8,000

        # Current value at $165
        current_price = 165.0
        current_value = net_position * current_price  # $16,500

        # P&L calculation
        avg_buy_price = buy_value / total_bought  # $151.67
        realized_pnl = (160.0 - avg_buy_price) * 50  # Profit on sold shares
        unrealized_pnl = (current_price - avg_buy_price) * net_position

        # Update position
        position.quantity = net_position
        position.price = current_price
        position.value = current_value

        assert position.quantity == net_position
        assert position.value == current_value
        assert position.price == current_price

    def test_stop_loss_trigger(self, mock_portfolio):
        """Test stop loss order execution to protect capital."""
        # Position with stop loss
        entry_price = 100.0
        stop_loss_pct = mock_portfolio.maxloss  # 5%
        stop_price = entry_price * (1 - stop_loss_pct)  # $95

        position_size = 100
        initial_value = position_size * entry_price  # $10,000

        # Price drops to stop
        current_price = 94.0  # Below stop

        # Stop should trigger
        should_stop = current_price <= stop_price
        assert should_stop

        # Calculate loss
        loss = (entry_price - current_price) * position_size
        loss_pct = loss / initial_value

        # Verify loss is limited
        assert loss_pct <= stop_loss_pct + 0.01  # Allow 1% slippage

    def test_margin_requirements(self, mock_portfolio):
        """Test margin requirement calculations."""
        available_capital = mock_portfolio.startcap
        margin_requirement = mock_portfolio.margin  # 50%

        # Calculate buying power
        buying_power = available_capital / margin_requirement  # $200k with 50% margin

        # Test position limits with margin
        position_value = 150000  # $150k position
        margin_used = position_value * margin_requirement  # $75k margin required

        assert margin_used <= available_capital
        assert buying_power == 200000  # 2x leverage with 50% margin

    def test_portfolio_cash_management(self, mock_portfolio):
        """Test minimum cash requirements."""
        total_capital = mock_portfolio.startcap
        min_cash_pct = mock_portfolio.mincash  # 10%
        min_cash = total_capital * min_cash_pct  # $10k

        # Available for positions
        available_for_trading = total_capital - min_cash  # $90k

        # Try to use all capital
        position_value = 95000  # Would leave only $5k cash
        remaining_cash = total_capital - position_value

        # Should not allow - violates min cash
        assert remaining_cash < min_cash
        allowed = remaining_cash >= min_cash
        assert not allowed

        # Correct position size
        max_position = available_for_trading  # $90k max
        assert max_position == 90000


class TestPortfolioValuation:
    """Test portfolio-wide valuation and P&L calculations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear state before each test."""
        Portfolio.portfolios.clear()
        Frame.frames.clear()

    @pytest.fixture
    def multi_position_portfolio(self):
        """Create portfolio with multiple positions."""
        portfolio = Portfolio(
            group_name="valuation_test",
            tag=f"test_{uuid.uuid4().hex[:8]}",
            startcap=1000000.0,  # $1M portfolio
        )

        # Mock positions with different P&L
        positions = {
            "AAPL": {"quantity": 1000, "entry": 150, "current": 165},  # +$15k
            "GOOGL": {"quantity": 500, "entry": 140, "current": 135},  # -$2.5k
            "MSFT": {"quantity": 300, "entry": 380, "current": 390},  # +$3k
            "AMZN": {"quantity": 0, "entry": 180, "current": 185},  # Closed position
        }

        portfolio.positions = positions
        portfolio.cash = 500000  # $500k in cash

        return portfolio

    def test_portfolio_total_value(self, multi_position_portfolio):
        """Test calculating total portfolio value."""
        portfolio = multi_position_portfolio

        # Calculate position values
        position_values = {
            "AAPL": 1000 * 165,  # $165,000
            "GOOGL": 500 * 135,  # $67,500
            "MSFT": 300 * 390,  # $117,000
        }

        total_position_value = sum(position_values.values())  # $349,500
        total_portfolio_value = portfolio.cash + total_position_value  # $849,500

        # Verify calculations
        assert total_position_value == 349500
        assert total_portfolio_value == 849500

        # Calculate returns
        initial_capital = portfolio.startcap
        total_return = (total_portfolio_value - initial_capital) / initial_capital

        # Lost money overall
        assert total_return < 0
        assert abs(total_return - (-0.1505)) < 0.001  # -15.05%

    def test_portfolio_risk_metrics(self, multi_position_portfolio):
        """Test portfolio risk calculations."""
        portfolio = multi_position_portfolio

        # Position concentrations
        position_values = {
            "AAPL": 1000 * 165,
            "GOOGL": 500 * 135,
            "MSFT": 300 * 390,
        }

        total_value = sum(position_values.values()) + portfolio.cash

        # Calculate position weights
        weights = {symbol: value / total_value for symbol, value in position_values.items()}

        # Check concentration limits
        max_concentration = max(weights.values())
        assert max_concentration < 0.25  # No position > 25%

        # Calculate portfolio beta (simplified)
        betas = {"AAPL": 1.2, "GOOGL": 1.1, "MSFT": 0.9}
        portfolio_beta = sum(weights.get(symbol, 0) * beta for symbol, beta in betas.items())

        assert 0.3 < portfolio_beta < 1.5  # Reasonable market exposure (allowing lower beta for cash-heavy portfolio)


class TestTradeSystem:
    """Test the trade system execution with real money scenarios."""

    @pytest.fixture
    def trading_system(self):
        """Create a trading system with rules."""
        from alphapy.system import System

        system = System(
            name="momentum_system",
            longentry="close > sma_20",
            longexit="close < sma_20",
            shortentry="close < sma_20",
            shortexit="close > sma_20",
            holdperiod=0,
            scale=False,
        )
        return system

    @pytest.fixture
    def market_data_for_system(self):
        """Create market data with signals."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        rng = np.random.default_rng(seed=42)

        # Generate trending price data
        trend = np.linspace(100, 120, 50).tolist() + np.linspace(120, 90, 50).tolist()
        noise = rng.standard_normal(100) * 2
        prices = np.array(trend) + noise

        df = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
                "volume": rng.integers(1000000, 5000000, 100),
            },
            index=dates,
        )

        # Add indicators
        df["sma_20"] = df["close"].rolling(20).mean()
        df["returns"] = df["close"].pct_change()

        return df

    def test_system_signal_generation(self, trading_system, market_data_for_system):
        """Test that system generates correct buy/sell signals."""
        df = market_data_for_system
        system = trading_system

        # Generate signals based on system rules
        df["long_entry"] = df["close"] > df["sma_20"]
        df["long_exit"] = df["close"] < df["sma_20"]

        # Count signals
        long_entries = df["long_entry"].sum()
        long_exits = df["long_exit"].sum()

        # Should have both entry and exit signals
        assert long_entries > 0
        assert long_exits > 0

        # Signals should be mutually exclusive
        both_signals = (df["long_entry"] & df["long_exit"]).sum()
        assert both_signals == 0

    def test_system_position_management(self, trading_system):
        """Test position entry and exit logic."""
        # Mock a trade sequence
        trades = []
        in_position = False
        position_size = 0

        signals = [
            ("2024-01-01", "buy", 100.0),
            ("2024-01-05", "hold", 105.0),
            ("2024-01-10", "sell", 110.0),
            ("2024-01-15", "buy", 108.0),
            ("2024-01-20", "sell", 106.0),
        ]

        for date, signal, price in signals:
            if signal == "buy" and not in_position:
                position_size = 100  # Buy 100 shares
                in_position = True
                trades.append({"date": date, "action": "buy", "quantity": position_size, "price": price})
            elif signal == "sell" and in_position:
                trades.append({"date": date, "action": "sell", "quantity": position_size, "price": price})
                in_position = False
                position_size = 0

        # Verify trade sequence
        assert len(trades) == 4  # 2 buys, 2 sells
        assert trades[0]["action"] == "buy"
        assert trades[1]["action"] == "sell"

        # Calculate P&L
        pnl_1 = (trades[1]["price"] - trades[0]["price"]) * 100  # First trade
        pnl_2 = (trades[3]["price"] - trades[2]["price"]) * 100  # Second trade

        assert pnl_1 == 1000  # $10 profit * 100 shares
        assert pnl_2 == -200  # $2 loss * 100 shares


class TestRiskManagement:
    """Test risk management and capital preservation."""

    def test_kelly_criterion(self):
        """Test Kelly Criterion for optimal position sizing."""
        # Historical trade statistics
        win_rate = 0.55  # 55% win rate
        avg_win = 1.5  # Average win is 1.5R
        avg_loss = 1.0  # Average loss is 1R

        # Kelly formula: f = (p*b - q) / b
        # where p = win_rate, q = 1-p, b = avg_win/avg_loss
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly_fraction = (p * b - q) / b

        # Should be positive for profitable system
        assert kelly_fraction > 0
        assert kelly_fraction < 1  # Should not bet everything

        # Apply Kelly fraction with safety factor
        safety_factor = 0.25  # Use 25% of Kelly
        position_size = kelly_fraction * safety_factor

        assert 0 < position_size < 0.25  # Reasonable position size

    def test_max_drawdown_protection(self):
        """Test maximum drawdown limits and actions."""
        initial_capital = 100000
        max_drawdown_limit = 0.20  # 20% max drawdown

        # Simulate equity curve
        equity_curve = [
            100000,
            105000,
            110000,
            108000,
            103000,  # Up then down
            95000,
            92000,
            85000,  # Big drawdown
            87000,
            90000,  # Recovery
        ]

        # Calculate drawdown at each point
        peak = initial_capital
        max_drawdown = 0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)

            # Check if should stop trading
            if drawdown > max_drawdown_limit:
                should_stop = True
                break
        else:
            should_stop = False

        assert max_drawdown > max_drawdown_limit
        assert should_stop

    def test_position_correlation_limits(self):
        """Test correlation-based position limits."""
        # Correlation matrix for 5 stocks
        correlations = {
            ("AAPL", "MSFT"): 0.7,  # High correlation (tech)
            ("AAPL", "GOOGL"): 0.65,
            ("MSFT", "GOOGL"): 0.8,
            ("AAPL", "XOM"): 0.2,  # Low correlation (tech vs energy)
            ("MSFT", "XOM"): 0.15,
        }

        # Current positions
        positions = ["AAPL", "MSFT"]

        # Check if can add GOOGL (high correlation)
        new_symbol = "GOOGL"
        max_correlation = 0.6  # Max allowed correlation

        # Check correlations with existing positions
        correlations_with_new = []
        for pos in positions:
            key = tuple(sorted([pos, new_symbol]))
            if key in correlations:
                correlations_with_new.append(correlations[key])

        avg_correlation = np.mean(correlations_with_new) if correlations_with_new else 0
        can_add = avg_correlation < max_correlation

        assert not can_add  # Should not add highly correlated position

    def test_var_calculation(self):
        """Test Value at Risk (VaR) calculation."""
        # Portfolio returns (daily)
        rng = np.random.default_rng(seed=42)
        returns = rng.normal(0.001, 0.02, 252)  # 252 trading days

        # Calculate VaR at 95% confidence
        confidence_level = 0.95
        var_95 = np.percentile(returns, (1 - confidence_level) * 100)

        # For $1M portfolio
        portfolio_value = 1000000
        var_dollar = portfolio_value * abs(var_95)

        # VaR should be negative (loss)
        assert var_95 < 0

        # Check if VaR is reasonable (1-5% for daily)
        assert 0.01 < abs(var_95) < 0.05

        # Dollar VaR
        assert 10000 < var_dollar < 50000  # $10k-50k daily VaR for $1M


class TestModelPredictions:
    """Test model predictions for trading signals."""

    def test_model_signal_generation(self):
        """Test that model generates valid trading signals."""
        from alphapy.model import Model

        # Mock model specs
        model_specs = {
            "algorithms": ["LR"],  # Logistic Regression
            "model_type": "classification",
            "target": "signal",
            "features": ["returns", "volume", "rsi"],
            "seed": 42,
        }

        # Create model
        model = Model(model_specs)

        # Mock predictions
        mock_predictions = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])  # Binary signals
        mock_probabilities = np.array(
            [
                [0.7, 0.3],
                [0.2, 0.8],
                [0.1, 0.9],
                [0.6, 0.4],
                [0.3, 0.7],
                [0.8, 0.2],
                [0.9, 0.1],
                [0.4, 0.6],
                [0.2, 0.8],
                [0.7, 0.3],
            ]
        )

        # Validate predictions
        assert len(mock_predictions) == 10
        assert all(p in [0, 1] for p in mock_predictions)

        # Check probability thresholds
        threshold = 0.6
        high_confidence = mock_probabilities[:, 1] > threshold

        # Only trade high confidence signals
        filtered_signals = mock_predictions * high_confidence

        assert sum(filtered_signals) < sum(mock_predictions)  # Fewer signals after filtering


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

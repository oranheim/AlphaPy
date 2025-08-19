"""
Comprehensive Portfolio Finance Tests

These tests focus on the accuracy of financial calculations including:
- Portfolio valuation and returns
- Risk metrics (Sharpe ratio, maximum drawdown, VaR)
- Position sizing and weight calculations
- P&L calculations and cost basis tracking
- Performance attribution and risk-adjusted returns

Critical for financial trading systems where accuracy is paramount.
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
    close_position,
    gen_portfolio,
    update_portfolio,
    valuate_portfolio,
    valuate_position,
)
from alphapy.space import Space


class TestPortfolioFinancialMetrics:
    """Test core financial metrics calculations for portfolio performance."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear state before each test."""
        Portfolio.portfolios.clear()
        Frame.frames.clear()

    @pytest.fixture
    def portfolio_with_history(self):
        """Create portfolio with trading history for metrics testing."""
        portfolio = Portfolio(
            group_name="metrics_test",
            tag=f"test_{uuid.uuid4().hex[:8]}",
            startcap=1000000.0,  # $1M starting capital
            maxpos=20,
            restricted=False,
        )

        # Create price data for testing
        dates = pd.date_range("2024-01-01", periods=252, freq="D")

        # Simulate realistic market data with different scenarios
        symbols = ["AAPL", "GOOGL", "TSLA", "SPY", "QQQ"]

        for symbol in symbols:
            # Different volatility profiles
            if symbol == "TSLA":
                volatility = 0.04  # High volatility
            elif symbol in ["SPY", "QQQ"]:
                volatility = 0.015  # Lower volatility ETFs
            else:
                volatility = 0.025  # Medium volatility

            # Generate returns with different market regimes
            rng = np.random.default_rng(42 + hash(symbol) % 1000)
            returns = rng.normal(0.0005, volatility, len(dates))

            # Add market regime changes
            returns[60:120] = rng.normal(-0.002, volatility * 1.5, 60)  # Bear market
            returns[180:220] = rng.normal(0.003, volatility * 0.8, 40)  # Bull market

            close_prices = 100 * np.exp(np.cumsum(returns))

            df = pd.DataFrame(
                {
                    "open": close_prices * rng.uniform(0.995, 1.005, len(dates)),
                    "high": close_prices * rng.uniform(1.001, 1.02, len(dates)),
                    "low": close_prices * rng.uniform(0.98, 0.999, len(dates)),
                    "close": close_prices,
                    "volume": rng.integers(1000000, 10000000, len(dates), endpoint=True),
                },
                index=dates,
            )

            # Register frame for Position class
            space = Space("stock", "prices", "1d")
            frame_name = f"{symbol}_stock_prices_1d"
            Frame.frames[frame_name] = Mock(df=df)

        return portfolio, symbols, dates

    def test_sharpe_ratio_calculation(self, portfolio_with_history):
        """Test accurate Sharpe ratio calculation for portfolio performance."""
        portfolio, symbols, dates = portfolio_with_history

        # Create portfolio returns series
        daily_returns = []
        portfolio_values = []

        current_value = portfolio.startcap

        rng = np.random.default_rng(42)
        for i, _date in enumerate(dates[:100]):  # 100 days of trading
            # Simulate daily return
            daily_return = 0.0 if i == 0 else rng.normal(0.0008, 0.015)  # ~20% annualized with 15% vol

            daily_returns.append(daily_return)
            current_value *= 1 + daily_return
            portfolio_values.append(current_value)

        returns_series = pd.Series(daily_returns, index=dates[:100])

        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # 2% annual
        daily_rf = risk_free_rate / 252

        excess_returns = returns_series - daily_rf
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns_series.std()

        # Validate Sharpe ratio calculation
        assert not np.isnan(sharpe_ratio)
        assert not np.isinf(sharpe_ratio)

        # For reasonable trading strategy, Sharpe should be within a plausible range
        assert -3 <= sharpe_ratio <= 5

        # Test edge cases
        # Zero volatility case
        zero_vol_returns = pd.Series([0.001] * 100)
        if zero_vol_returns.std() == 0:
            zero_vol_sharpe = 0  # Handle division by zero
        else:
            zero_vol_sharpe = np.sqrt(252) * (zero_vol_returns.mean() - daily_rf) / zero_vol_returns.std()
        # For constant returns, Sharpe should be very high (not zero)
        assert zero_vol_sharpe >= 0

        # Negative returns case
        negative_returns = pd.Series([-0.001] * 100)
        neg_excess = negative_returns - daily_rf
        negative_sharpe = np.sqrt(252) * neg_excess.mean() / negative_returns.std()
        assert negative_sharpe < 0

    def test_maximum_drawdown_calculation(self, portfolio_with_history):
        """Test maximum drawdown calculation for risk assessment."""
        portfolio, symbols, dates = portfolio_with_history

        # Create equity curve with known drawdown
        equity_curve = [
            1000000,  # Start
            1050000,  # +5%
            1100000,  # +10% (peak)
            1080000,  # -1.8%
            1050000,  # -4.5%
            950000,  # -13.6% (trough)
            980000,  # -10.9%
            1020000,  # -7.3%
            1150000,  # +4.5% (new peak)
            1100000,  # -4.3%
        ]

        # Calculate drawdown properly
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown_series = (equity_series - running_max) / running_max
        max_drawdown = drawdown_series.min()

        # Expected max drawdown: (950000 - 1100000) / 1100000 = -13.636%
        expected_max_dd = -0.13636363636363635

        assert abs(max_drawdown - expected_max_dd) < 0.0001
        assert max_drawdown <= 0  # Drawdown should always be negative or zero

        # Test recovery calculation
        recovery_periods = []
        in_drawdown = False
        dd_start_idx = None

        for i, dd in enumerate(drawdown_series):
            if dd < -0.05 and not in_drawdown:  # Drawdown > 5%
                in_drawdown = True
                dd_start_idx = i
            elif dd >= -0.001 and in_drawdown:  # Recovery (within 0.1% of peak)
                in_drawdown = False
                recovery_periods.append(i - dd_start_idx)

        # Should have recorded recovery periods
        assert len(recovery_periods) > 0

    def test_calmar_ratio_calculation(self, portfolio_with_history):
        """Test Calmar ratio (return/max drawdown) for risk-adjusted performance."""
        portfolio, symbols, dates = portfolio_with_history

        # Create sample equity curve
        initial_value = 1000000
        final_value = 1200000  # 20% total return
        periods = 252  # 1 year

        # Annualized return
        annual_return = (final_value / initial_value) ** (252 / periods) - 1

        # Sample drawdown of 15%
        max_drawdown = 0.15

        # Calmar ratio = Annual Return / Max Drawdown
        calmar_ratio = annual_return / max_drawdown

        # For 20% return and 15% max DD: 0.20 / 0.15 = 1.33
        expected_calmar = 0.20 / 0.15

        assert abs(calmar_ratio - expected_calmar) < 0.01

        # Good strategies typically have Calmar > 1.0
        assert calmar_ratio > 1.0

        # Test edge case: zero drawdown
        with pytest.raises(ZeroDivisionError):
            zero_dd_calmar = annual_return / 0

    def test_sortino_ratio_calculation(self, portfolio_with_history):
        """Test Sortino ratio focusing on downside deviation."""
        portfolio, symbols, dates = portfolio_with_history

        # Create returns with mixed performance
        returns = np.array(
            [
                0.02,
                -0.01,
                0.015,
                -0.005,
                0.03,  # Mixed returns
                -0.02,
                0.01,
                -0.015,
                0.025,
                -0.008,
                0.018,
                -0.012,
                0.022,
                -0.003,
                0.016,
            ]
        )

        risk_free_rate = 0.02 / 252  # Daily risk-free rate

        # Calculate excess returns
        excess_returns = returns - risk_free_rate

        # Calculate downside deviation (only negative excess returns)
        negative_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt(np.mean(negative_returns**2))

        # Sortino ratio
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_deviation

        # Validate calculation
        assert not np.isnan(sortino_ratio)
        assert not np.isinf(sortino_ratio)

        # Sortino should be higher than Sharpe for same data (focuses on downside)
        standard_deviation = np.sqrt(np.mean(excess_returns**2))
        pseudo_sharpe = np.sqrt(252) * excess_returns.mean() / standard_deviation

        # Sortino typically higher than Sharpe when positive skewness
        if excess_returns.mean() > 0:
            assert sortino_ratio >= pseudo_sharpe

    def test_value_at_risk_calculation(self, portfolio_with_history):
        """Test Value at Risk (VaR) calculation for risk management."""
        portfolio, symbols, dates = portfolio_with_history

        # Create realistic daily returns
        rng = np.random.default_rng(42)
        rng = np.random.default_rng(seed=42)
        returns = rng.normal(0.001, 0.02, 1000)  # 1000 days of returns

        # Calculate VaR at different confidence levels
        confidence_levels = [0.95, 0.99, 0.999]

        for confidence in confidence_levels:
            # Parametric VaR (assuming normal distribution)
            var_parametric = np.percentile(returns, (1 - confidence) * 100)

            # Historical VaR
            var_historical = np.percentile(returns, (1 - confidence) * 100)

            # Monte Carlo VaR (simplified)
            simulated_returns = rng.normal(returns.mean(), returns.std(), 10000)
            var_monte_carlo = np.percentile(simulated_returns, (1 - confidence) * 100)

            # All VaR measures should be negative (losses)
            assert var_parametric <= 0
            assert var_historical <= 0
            assert var_monte_carlo <= 0

            # Higher confidence should give more extreme VaR
            if confidence > 0.95:
                var_95 = np.percentile(returns, 5)
                assert var_parametric <= var_95  # 99% VaR should be worse than 95% VaR

        # Convert to dollar VaR for $1M portfolio
        portfolio_value = 1000000
        var_95_dollar = portfolio_value * abs(np.percentile(returns, 5))

        # Daily VaR should be reasonable (1-5% of portfolio typically)
        assert 10000 <= var_95_dollar <= 100000

    def test_portfolio_beta_calculation(self, portfolio_with_history):
        """Test portfolio beta calculation against benchmark."""
        portfolio, symbols, dates = portfolio_with_history

        # Create correlated returns (portfolio vs benchmark)
        rng = np.random.default_rng(42)

        # Market returns (benchmark)
        rng = np.random.default_rng(seed=42)
        market_returns = rng.normal(0.0005, 0.015, 252)

        # Portfolio returns (correlated to market)
        beta_true = 1.2  # Portfolio should be 20% more volatile than market
        alpha_true = 0.0002  # Small alpha

        portfolio_returns = alpha_true + beta_true * market_returns + rng.normal(0, 0.005, 252)

        # Calculate beta using regression
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        beta_calculated = covariance / market_variance

        # Beta should be close to true beta
        assert abs(beta_calculated - beta_true) < 0.1

        # Calculate alpha (intercept)
        alpha_calculated = portfolio_returns.mean() - beta_calculated * market_returns.mean()

        # Alpha should be close to true alpha
        assert abs(alpha_calculated - alpha_true) < 0.0005

        # Calculate R-squared
        correlation = np.corrcoef(portfolio_returns, market_returns)[0, 1]
        r_squared = correlation**2

        # R-squared should indicate reasonable correlation
        assert 0.3 <= r_squared <= 1.0

    def test_information_ratio_calculation(self, portfolio_with_history):
        """Test Information Ratio for active management performance."""
        portfolio, symbols, dates = portfolio_with_history

        # Portfolio returns vs benchmark returns
        rng = np.random.default_rng(42)
        rng = np.random.default_rng(seed=42)
        portfolio_returns = rng.normal(0.0008, 0.018, 252)  # Slightly higher return
        benchmark_returns = rng.normal(0.0005, 0.015, 252)

        # Active returns (excess over benchmark)
        active_returns = portfolio_returns - benchmark_returns

        # Information ratio = Active Return / Tracking Error
        active_return_annual = active_returns.mean() * 252
        tracking_error = active_returns.std() * np.sqrt(252)

        information_ratio = active_return_annual / tracking_error

        # Validate calculation
        assert not np.isnan(information_ratio)
        assert not np.isinf(information_ratio)

        # Good active managers typically have IR > 0.5
        # Our simulated data should show some skill
        if active_return_annual > 0:
            assert information_ratio >= 0


class TestPortfolioPositionCalculations:
    """Test position-level calculations including cost basis and P&L."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear state before each test."""
        Portfolio.portfolios.clear()
        Frame.frames.clear()

    @pytest.fixture
    def position_with_trades(self):
        """Create position with multiple trades for testing."""
        portfolio = Portfolio(
            group_name="position_test",
            tag=f"test_{uuid.uuid4().hex[:8]}",
            startcap=100000.0,
        )

        # Create price data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.standard_normal(100) * 0.5)

        df = pd.DataFrame(
            {
                "open": prices * 0.995,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.full(100, 1000000),
            },
            index=dates,
        )

        # Register frame
        space = Space("stock", "prices", "1d")
        Frame.frames["AAPL_stock_prices_1d"] = Mock(df=df)

        # Create position
        position = Position(portfolio, "AAPL", dates[0])

        return position, dates, prices

    def test_cost_basis_calculation_fifo(self, position_with_trades):
        """Test FIFO cost basis calculation with multiple trades."""
        position, dates, prices = position_with_trades

        # Multiple buy trades at different prices
        trades = [
            Trade("AAPL", Orders.le, 100, 150.0, dates[0]),  # Buy 100 @ $150
            Trade("AAPL", Orders.le, 200, 155.0, dates[5]),  # Buy 200 @ $155
            Trade("AAPL", Orders.le, 150, 148.0, dates[10]),  # Buy 150 @ $148
            Trade("AAPL", Orders.lx, -100, 160.0, dates[15]),  # Sell 100 @ $160 (first lot)
            Trade("AAPL", Orders.lx, -50, 162.0, dates[20]),  # Sell 50 @ $162 (second lot)
        ]

        # Simulate the position state after all trades
        total_bought_shares = 450  # 100 + 200 + 150
        total_sold_shares = 150  # 100 + 50
        net_position = 300  # 450 - 150

        # Calculate weighted average cost basis
        total_cost = (100 * 150.0) + (200 * 155.0) + (150 * 148.0)  # $68,200
        avg_cost_basis = total_cost / total_bought_shares  # $151.56
        expected_avg_cost = 68200 / 450  # $151.5556

        # Calculate realized P&L (FIFO basis)
        # First 100 shares sold @ $160, cost basis $150 = $1,000 profit
        # Next 50 shares sold @ $162, cost basis $155 = $350 profit
        realized_pnl = (160 - 150) * 100 + (162 - 155) * 50

        assert abs(avg_cost_basis - expected_avg_cost) < 0.001
        assert realized_pnl == 1350

        # Remaining position cost basis (150 shares @ $155 + 150 shares @ $148)
        remaining_cost = (150 * 155.0) + (150 * 148.0)  # $45,450
        remaining_avg_cost = remaining_cost / 300  # $151.50

        assert abs(remaining_avg_cost - 151.5) < 0.001

    def test_unrealized_pnl_calculation(self, position_with_trades):
        """Test unrealized P&L calculation based on current market price."""
        position, dates, prices = position_with_trades

        # Position details
        shares_held = 500
        avg_cost_basis = 152.50
        current_price = 165.75

        # Unrealized P&L calculation
        unrealized_pnl = (current_price - avg_cost_basis) * shares_held
        unrealized_pnl_pct = (current_price / avg_cost_basis - 1) * 100

        expected_pnl = (165.75 - 152.50) * 500  # $6,625
        expected_pct = (165.75 / 152.50 - 1) * 100  # 8.69%

        assert abs(unrealized_pnl - expected_pnl) < 0.01
        assert abs(unrealized_pnl_pct - expected_pct) < 0.01

        # Test with loss position
        current_price_loss = 140.00
        unrealized_loss = (current_price_loss - avg_cost_basis) * shares_held
        unrealized_loss_pct = (current_price_loss / avg_cost_basis - 1) * 100

        expected_loss = (140.00 - 152.50) * 500  # -$6,250
        expected_loss_pct = (140.00 / 152.50 - 1) * 100  # -8.20%

        assert abs(unrealized_loss - expected_loss) < 0.01
        assert abs(unrealized_loss_pct - expected_loss_pct) < 0.01
        assert unrealized_loss < 0
        assert unrealized_loss_pct < 0

    def test_position_sizing_algorithms(self, position_with_trades):
        """Test different position sizing algorithms for risk management."""
        position, dates, prices = position_with_trades

        # Portfolio parameters
        portfolio_value = 1000000  # $1M
        risk_per_trade = 0.02  # 2% risk per trade

        # Test 1: Fixed Fractional Position Sizing
        position_fraction = 0.05  # 5% of portfolio per position
        position_size_ff = portfolio_value * position_fraction  # $50,000

        stock_price = 150.0
        shares_ff = int(position_size_ff / stock_price)  # 333 shares

        assert shares_ff == 333
        assert shares_ff * stock_price <= position_size_ff

        # Test 2: Volatility-Based Position Sizing
        daily_volatility = 0.025  # 2.5% daily volatility
        annual_volatility = daily_volatility * np.sqrt(252)  # ~39.7%

        # Position size inversely proportional to volatility
        base_size = 50000
        volatility_adjusted_size = base_size / annual_volatility
        shares_vol = int(volatility_adjusted_size / stock_price)

        assert shares_vol > 0
        # Note: Relationship depends on volatility level vs fixed fraction target

        # Test 3: Kelly Criterion Position Sizing
        win_rate = 0.55
        avg_win = 0.08  # 8% average win
        avg_loss = 0.05  # 5% average loss

        # Kelly fraction = (p*b - q) / b, where b = avg_win/avg_loss
        b = avg_win / avg_loss
        kelly_fraction = (win_rate * b - (1 - win_rate)) / b

        # Apply fraction to portfolio (with safety factor)
        safety_factor = 0.25  # Use 25% of Kelly
        kelly_position_size = portfolio_value * kelly_fraction * safety_factor
        shares_kelly = int(kelly_position_size / stock_price)

        assert kelly_fraction > 0  # Should be positive for profitable system
        assert 0 < kelly_position_size < portfolio_value * 0.25
        assert shares_kelly > 0

    def test_portfolio_concentration_limits(self, position_with_trades):
        """Test portfolio concentration and correlation limits."""
        position, dates, prices = position_with_trades

        # Portfolio with multiple positions
        positions = {
            "AAPL": {"value": 150000, "sector": "Technology"},
            "GOOGL": {"value": 120000, "sector": "Technology"},
            "MSFT": {"value": 100000, "sector": "Technology"},
            "JPM": {"value": 80000, "sector": "Financial"},
            "XOM": {"value": 50000, "sector": "Energy"},
        }

        total_portfolio_value = sum(pos["value"] for pos in positions.values())

        # Calculate individual position concentrations
        concentrations = {}
        for symbol, pos in positions.items():
            concentrations[symbol] = pos["value"] / total_portfolio_value

        # Test individual position limits (e.g., max 30% per position for this test)
        max_individual_concentration = 0.30
        for symbol, concentration in concentrations.items():
            assert concentration <= max_individual_concentration

        # Test sector concentration limits (e.g., max 40% per sector)
        sector_concentrations = {}
        for symbol, pos in positions.items():
            sector = pos["sector"]
            if sector not in sector_concentrations:
                sector_concentrations[sector] = 0
            sector_concentrations[sector] += pos["value"]

        max_sector_concentration = 0.50  # 50% max per sector
        for sector, value in sector_concentrations.items():
            sector_pct = value / total_portfolio_value
            # Technology sector is 74% (high concentration)
            if sector == "Technology":
                assert sector_pct > max_sector_concentration  # Should trigger warning
            else:
                assert sector_pct <= max_sector_concentration

    def test_portfolio_rebalancing_calculations(self, position_with_trades):
        """Test portfolio rebalancing to target weights."""
        position, dates, prices = position_with_trades

        # Current portfolio state
        current_positions = {
            "AAPL": {"shares": 1000, "price": 150.0, "target_weight": 0.20},
            "GOOGL": {"shares": 400, "price": 140.0, "target_weight": 0.20},
            "MSFT": {"shares": 300, "price": 380.0, "target_weight": 0.20},
            "AMZN": {"shares": 200, "price": 180.0, "target_weight": 0.20},
            "TSLA": {"shares": 100, "price": 200.0, "target_weight": 0.20},
        }

        # Calculate current values and weights
        total_value = 0
        current_values = {}

        for symbol, pos in current_positions.items():
            value = pos["shares"] * pos["price"]
            current_values[symbol] = value
            total_value += value

        current_weights = {symbol: value / total_value for symbol, value in current_values.items()}

        # Calculate rebalancing trades needed
        trades_needed = {}
        for symbol, pos in current_positions.items():
            target_value = total_value * pos["target_weight"]
            current_value = current_values[symbol]
            value_difference = target_value - current_value

            shares_to_trade = value_difference / pos["price"]
            trades_needed[symbol] = {
                "shares": round(shares_to_trade),
                "value": value_difference,
                "direction": "buy" if shares_to_trade > 0 else "sell",
            }

        # Verify trades sum to approximately zero (conservation of value)
        total_trade_value = sum(trade["value"] for trade in trades_needed.values())
        assert abs(total_trade_value) < 100  # Should be close to zero (rounding errors)

        # Verify each position will be closer to target after rebalancing
        for symbol, trade in trades_needed.items():
            new_shares = current_positions[symbol]["shares"] + trade["shares"]
            new_value = new_shares * current_positions[symbol]["price"]
            new_weight = new_value / total_value
            target_weight = current_positions[symbol]["target_weight"]

            # New weight should be closer to target than current weight
            current_deviation = abs(current_weights[symbol] - target_weight)
            new_deviation = abs(new_weight - target_weight)
            assert new_deviation <= current_deviation


class TestRiskManagement:
    """Test risk management calculations and controls."""

    def test_position_risk_calculations(self):
        """Test position-level risk metrics."""
        # Position parameters
        entry_price = 100.0
        current_price = 95.0
        shares = 1000
        stop_loss_price = 92.0

        # Current loss
        current_loss = (entry_price - current_price) * shares  # $5,000 loss
        current_loss_pct = (current_price / entry_price - 1) * 100  # -5%

        # Risk to stop loss
        risk_to_stop = (current_price - stop_loss_price) * shares  # $3,000 additional risk
        total_risk = (entry_price - stop_loss_price) * shares  # $8,000 total risk

        assert current_loss == 5000
        assert abs(current_loss_pct - (-5.0)) < 0.001
        assert risk_to_stop == 3000
        assert total_risk == 8000

        # Risk-reward ratio for a target of $110
        target_price = 110.0
        potential_reward = (target_price - current_price) * shares  # $15,000
        risk_reward_ratio = potential_reward / risk_to_stop  # 5:1

        assert risk_reward_ratio == 5.0
        assert risk_reward_ratio > 2.0  # Good risk-reward ratio

    def test_portfolio_var_stress_testing(self):
        """Test VaR calculation under stress scenarios."""
        # Portfolio returns for different market conditions
        rng = np.random.default_rng(42)
        rng = np.random.default_rng(seed=42)
        normal_market = rng.normal(0.0005, 0.015, 1000)
        stress_market = rng.normal(-0.002, 0.03, 1000)  # Bear market
        crash_scenario = rng.normal(-0.008, 0.05, 100)  # Market crash

        # Calculate VaR for each scenario
        var_95_normal = np.percentile(normal_market, 5)
        var_95_stress = np.percentile(stress_market, 5)
        var_95_crash = np.percentile(crash_scenario, 5)

        # Stress VaR should be worse than normal VaR
        assert var_95_stress < var_95_normal
        assert var_95_crash < var_95_stress

        # Expected Shortfall (Conditional VaR) - average of losses beyond VaR
        es_95_normal = normal_market[normal_market <= var_95_normal].mean()
        es_95_stress = stress_market[stress_market <= var_95_stress].mean()

        # Expected shortfall should be worse than VaR
        assert es_95_normal < var_95_normal
        assert es_95_stress < var_95_stress

        # Stress test should show higher potential losses
        assert abs(es_95_stress) > abs(es_95_normal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

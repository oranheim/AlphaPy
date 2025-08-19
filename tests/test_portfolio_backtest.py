"""
Portfolio Management and Backtesting Tests for AlphaPy

These tests validate portfolio construction, optimization, backtesting engine,
and performance analytics essential for production trading systems.
"""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from alphapy.analysis import Analysis
from alphapy.frame import Frame
from alphapy.globals import ModelType, Orders, Partition
from alphapy.portfolio import Portfolio, Position, Trade
from alphapy.space import Space


class TestPortfolioConstruction:
    """Test portfolio construction and management."""

    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio for testing."""
        import uuid

        # Use unique tag to avoid conflicts
        unique_tag = f"test_{uuid.uuid4().hex[:8]}"

        # Clear any existing portfolio with same name
        Portfolio.portfolios.clear()

        portfolio = Portfolio(
            group_name="test_group",
            tag=unique_tag,
            maxpos=10,
            startcap=100000,
            margin=0.5,
            mincash=0.1,
            fixedfrac=0.02,
            maxloss=0.05,
        )
        return portfolio

    @pytest.fixture
    def sample_positions(self, sample_portfolio):
        """Create sample positions for testing."""
        positions = {}
        rng = np.random.default_rng(seed=42)

        # Create mock price data frames and add to Frame.frames
        for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT"]):
            # Create price data
            dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
            prices = 100 + i * 50 + rng.standard_normal(100) * 5
            df = pd.DataFrame(
                {"close": prices, "volume": rng.integers(1000000, 10000000, 100, endpoint=True)}, index=dates
            )

            # Add frame to Frame.frames
            space = sample_portfolio.space
            frame_key = f"{symbol}_{space.subject}_{space.schema}_{space.fractal}"
            Frame.frames[frame_key] = Mock(df=df)

            # Create position
            position = Position(
                portfolio=sample_portfolio, name=symbol, opendate=datetime(2023, 1, 1) + timedelta(days=i)
            )

            # Set some initial values
            position.quantity = 100 + i * 10
            position.price = 100 + i * 50
            position.value = position.quantity * position.price

            positions[symbol] = position

        return positions

    def test_portfolio_initialization(self, sample_portfolio):
        """Test portfolio initialization and configuration."""
        assert sample_portfolio.group_name == "test_group"
        assert sample_portfolio.tag.startswith("test_")  # Tag starts with test_
        assert sample_portfolio.maxpos == 10
        assert sample_portfolio.startcap == 100000
        assert sample_portfolio.margin == 0.5
        assert sample_portfolio.mincash == 0.1
        assert sample_portfolio.fixedfrac == 0.02
        assert sample_portfolio.maxloss == 0.05

    def test_position_management(self, sample_portfolio, sample_positions):
        """Test adding and removing positions from portfolio."""
        # Add positions to portfolio
        sample_portfolio.positions = sample_positions

        assert len(sample_portfolio.positions) == 3

        # Check position details
        aapl_position = sample_portfolio.positions["AAPL"]
        assert aapl_position.name == "AAPL"
        assert aapl_position.quantity == 100
        assert aapl_position.price == 100

        # Remove position
        del sample_portfolio.positions["AAPL"]
        assert len(sample_portfolio.positions) == 2

    def test_portfolio_value_calculation(self, sample_portfolio, sample_positions):
        """Test portfolio value and P&L calculations."""
        # Set up portfolio with positions
        sample_portfolio.positions = sample_positions

        # Mock current prices
        current_prices = {"AAPL": 110, "GOOGL": 160, "MSFT": 210}

        # Calculate portfolio value
        total_value = sample_portfolio.cash  # Starting with cash

        for symbol, position in sample_portfolio.positions.items():
            if position.status == "opened":  # Open position
                current_price = current_prices[symbol]
                position_value = position.quantity * current_price
                total_value += position_value

        # Calculate returns
        portfolio_return = (total_value - sample_portfolio.startcap) / sample_portfolio.startcap

        # Validate calculations
        assert total_value > sample_portfolio.startcap
        assert portfolio_return > 0

    def test_position_sizing_constraints(self, sample_portfolio):
        """Test position sizing with various constraints."""
        # Available capital
        available_capital = sample_portfolio.startcap * (1 - sample_portfolio.mincash)

        # Maximum position size (fixed fractional)
        max_position_size = sample_portfolio.startcap * sample_portfolio.fixedfrac

        # Number of positions allowed
        max_positions = min(sample_portfolio.maxpos, int(available_capital / max_position_size))

        # Validate constraints
        assert available_capital == 90000  # 100000 * 0.9
        assert max_position_size == 2000  # 100000 * 0.02
        assert max_positions == 10  # Limited by maxpos, not capital


class TestBacktestingEngine:
    """Test the backtesting engine functionality."""

    @pytest.fixture
    def backtest_data(self):
        """Generate comprehensive backtest data."""
        # Generate 2 years of daily data
        dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="B")  # Business days

        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        data = {}

        for symbol in symbols:
            rng = np.random.default_rng(seed=hash(symbol) % 100)  # Reproducible per symbol

            # Generate realistic price movement
            returns = rng.normal(0.0005, 0.02, len(dates))
            returns[::20] += rng.choice([-0.05, 0.05])  # Add jumps

            prices = 100 * np.exp(np.cumsum(returns))

            df = pd.DataFrame(
                {
                    "open": prices * rng.uniform(0.99, 1.01, len(dates)),
                    "high": prices * rng.uniform(1.01, 1.03, len(dates)),
                    "low": prices * rng.uniform(0.97, 0.99, len(dates)),
                    "close": prices,
                    "volume": rng.integers(1000000, 10000000, len(dates), endpoint=True),
                },
                index=dates,
            )

            data[symbol] = df

        return data

    def test_backtest_execution(self, backtest_data):
        """Test executing a complete backtest."""
        # Backtest parameters
        initial_capital = 100000
        commission = 0.001  # 0.1% per trade
        slippage = 0.0005  # 0.05% slippage

        # Initialize backtest results
        results = {"dates": [], "portfolio_value": [], "positions": [], "trades": [], "cash": []}

        cash = initial_capital
        positions = {}

        # Simple momentum strategy backtest
        for date in backtest_data["AAPL"].index:
            portfolio_value = cash

            for symbol, df in backtest_data.items():
                if date not in df.index:
                    continue

                current_price = df.loc[date, "close"]

                # Calculate 20-day momentum
                if len(df.loc[:date]) >= 20:
                    momentum = df.loc[:date, "close"].iloc[-1] / df.loc[:date, "close"].iloc[-20] - 1

                    # Entry signal
                    if momentum > 0.05 and symbol not in positions:
                        # Buy signal
                        shares = int((cash * 0.1) / current_price)  # 10% position
                        if shares > 0:
                            cost = shares * current_price * (1 + commission + slippage)
                            if cost <= cash:
                                positions[symbol] = {
                                    "shares": shares,
                                    "entry_price": current_price * (1 + slippage),
                                    "entry_date": date,
                                }
                                cash -= cost
                                results["trades"].append(
                                    {
                                        "date": date,
                                        "symbol": symbol,
                                        "action": "buy",
                                        "shares": shares,
                                        "price": current_price * (1 + slippage),
                                    }
                                )

                    # Exit signal
                    elif momentum < -0.05 and symbol in positions:
                        # Sell signal
                        position = positions[symbol]
                        proceeds = position["shares"] * current_price * (1 - commission - slippage)
                        cash += proceeds

                        results["trades"].append(
                            {
                                "date": date,
                                "symbol": symbol,
                                "action": "sell",
                                "shares": position["shares"],
                                "price": current_price * (1 - slippage),
                            }
                        )

                        del positions[symbol]

                # Update portfolio value with current positions
                if symbol in positions:
                    portfolio_value += positions[symbol]["shares"] * current_price

            results["dates"].append(date)
            results["portfolio_value"].append(portfolio_value)
            results["positions"].append(len(positions))
            results["cash"].append(cash)

        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(
            {
                "date": results["dates"],
                "portfolio_value": results["portfolio_value"],
                "num_positions": results["positions"],
                "cash": results["cash"],
            }
        )

        # Validate backtest results
        assert len(results_df) == len(backtest_data["AAPL"])
        assert results_df["portfolio_value"].iloc[0] == initial_capital
        assert (results_df["portfolio_value"] > 0).all()  # No bankruptcy
        assert (results_df["cash"] >= 0).all()  # No negative cash
        assert len(results["trades"]) > 0  # Some trades executed

    def test_transaction_costs(self):
        """Test impact of transaction costs on returns."""
        # Simulate trades with different cost structures
        trade_value = 10000

        # Cost structures
        cost_structures = [
            {"commission": 0, "slippage": 0},  # No costs
            {"commission": 0.001, "slippage": 0},  # Commission only
            {"commission": 0, "slippage": 0.001},  # Slippage only
            {"commission": 0.001, "slippage": 0.001},  # Both
            {"commission": 0.002, "slippage": 0.002},  # High costs
        ]

        results = []
        for costs in cost_structures:
            # Buy transaction
            buy_cost = trade_value * (1 + costs["commission"] + costs["slippage"])

            # Sell transaction (assuming 10% profit)
            sell_value = trade_value * 1.1
            sell_proceeds = sell_value * (1 - costs["commission"] - costs["slippage"])

            # Net profit
            net_profit = sell_proceeds - buy_cost
            net_return = net_profit / buy_cost

            results.append(
                {
                    "commission": costs["commission"],
                    "slippage": costs["slippage"],
                    "total_cost": costs["commission"] + costs["slippage"],
                    "net_return": net_return,
                }
            )

        results_df = pd.DataFrame(results)

        # Validate cost impact
        assert results_df["net_return"].iloc[0] > results_df["net_return"].iloc[-1]
        assert (results_df["net_return"] <= 0.1).all()  # Less than or equal to gross return
        assert results_df["net_return"].is_monotonic_decreasing  # Higher costs = lower returns


class TestPerformanceAnalytics:
    """Test performance measurement and analytics."""

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        # Generate returns
        rng = np.random.default_rng(seed=42)
        rng = np.random.default_rng(seed=42)
        daily_returns = rng.normal(0.0005, 0.02, 252)  # One year of daily returns

        # Calculate Sharpe ratio
        risk_free_rate = 0.02 / 252  # 2% annual, converted to daily
        excess_returns = daily_returns - risk_free_rate

        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

        # Validate Sharpe ratio
        assert -3 < sharpe_ratio < 3  # Reasonable range
        assert not np.isnan(sharpe_ratio)

    def test_maximum_drawdown(self):
        """Test maximum drawdown calculation."""
        # Generate cumulative returns
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        rng = np.random.default_rng(seed=42)
        returns = rng.normal(0.001, 0.02, 252)

        # Add a significant drawdown period
        returns[100:120] = -0.02  # 20 days of -2% returns

        cumulative = pd.Series((1 + returns).cumprod(), index=dates)

        # Calculate drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Find drawdown duration
        drawdown_start = drawdown.idxmin()
        recovery_date = (
            cumulative[drawdown_start:].loc[cumulative[drawdown_start:] >= running_max[drawdown_start]].index
        )

        if len(recovery_date) > 0:
            drawdown_duration = (recovery_date[0] - drawdown_start).days
        else:
            drawdown_duration = (dates[-1] - drawdown_start).days

        # Validate drawdown metrics
        assert -1 <= max_drawdown <= 0
        assert drawdown_duration > 0
        assert (drawdown <= 0).all()

    def test_calmar_ratio(self):
        """Test Calmar ratio (return / max drawdown)."""
        # Generate portfolio values
        initial_value = 100000
        rng = np.random.default_rng(seed=42)
        returns = rng.normal(0.0008, 0.015, 252)
        portfolio_values = initial_value * (1 + returns).cumprod()

        # Calculate annualized return
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1

        # Calculate maximum drawdown
        running_max = pd.Series(portfolio_values).cummax()
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Calculate Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else np.inf

        # Validate Calmar ratio
        assert calmar_ratio > 0 if annualized_return > 0 else calmar_ratio <= 0
        assert not np.isinf(calmar_ratio) if max_drawdown > 0 else True

    def test_win_loss_statistics(self):
        """Test win/loss ratio and related statistics."""
        # Generate trade results
        rng = np.random.default_rng(seed=42)
        n_trades = 100

        # Generate trades with 55% win rate
        win_probability = 0.55
        trades = []

        for _ in range(n_trades):
            # Generate profit based on win probability
            profit = (
                rng.uniform(0.5, 3.0)  # Winning: 0.5% to 3% profit
                if rng.random() < win_probability
                else rng.uniform(-2.0, -0.3)  # Losing: -2% to -0.3% loss
            )

            trades.append(profit)

        trades = np.array(trades)

        # Calculate statistics
        winning_trades = trades[trades > 0]
        losing_trades = trades[trades <= 0]

        win_rate = len(winning_trades) / len(trades)
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0

        # Profit factor
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Validate statistics
        assert 0 <= win_rate <= 1
        assert avg_win > 0
        assert avg_loss > 0
        assert profit_factor > 0
        assert expectancy != 0  # Should have some edge (positive or negative)

    def test_rolling_performance_metrics(self):
        """Test rolling window performance calculations."""
        # Generate time series of returns
        dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
        rng = np.random.default_rng(seed=42)
        returns = pd.Series(rng.normal(0.0005, 0.02, len(dates)), index=dates)

        # Rolling metrics (60-day window)
        window = 60

        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        rolling_sharpe = (rolling_mean * 252) / (rolling_std * np.sqrt(252))

        # Rolling maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window=window).max()
        rolling_dd = (cumulative - rolling_max) / rolling_max

        # Validate rolling metrics
        assert rolling_mean.iloc[window:].notna().all()
        assert rolling_std.iloc[window:].notna().all()
        assert rolling_sharpe.iloc[window:].notna().all()
        assert (rolling_std > 0).iloc[window:].all()


class TestAdvancedPortfolioStrategies:
    """Test advanced portfolio optimization and allocation strategies."""

    def test_markowitz_optimization(self):
        """Test mean-variance (Markowitz) portfolio optimization."""
        # Historical returns for 5 assets
        n_assets = 5
        n_days = 252

        rng = np.random.default_rng(seed=42)
        returns = pd.DataFrame(
            rng.multivariate_normal(
                mean=[0.0005] * n_assets,
                cov=np.eye(n_assets) * 0.0004 + np.ones((n_assets, n_assets)) * 0.0001,
                size=n_days,
            ),
            columns=[f"Asset_{i}" for i in range(n_assets)],
        )

        # Calculate expected returns and covariance
        expected_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized

        # Minimum variance portfolio (simplified)
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(n_assets)

        weights_minvar = inv_cov @ ones / (ones.T @ inv_cov @ ones)

        # Portfolio metrics
        portfolio_return = weights_minvar @ expected_returns
        portfolio_variance = weights_minvar @ cov_matrix @ weights_minvar
        portfolio_std = np.sqrt(portfolio_variance)

        # Validate optimization
        assert np.allclose(weights_minvar.sum(), 1.0)  # Weights sum to 1
        assert (weights_minvar >= -0.1).all()  # Allow small negative weights (short)
        assert portfolio_std > 0
        assert not np.isnan(portfolio_return)

    def test_risk_parity_allocation(self):
        """Test risk parity portfolio allocation."""
        # Asset covariance matrix
        n_assets = 4
        volatilities = np.array([0.15, 0.20, 0.10, 0.25])  # Annual volatilities

        # Create correlation matrix
        correlation = np.array(
            [[1.00, 0.30, 0.20, 0.15], [0.30, 1.00, 0.25, 0.35], [0.20, 0.25, 1.00, 0.10], [0.15, 0.35, 0.10, 1.00]]
        )

        # Covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation

        # Initial equal weights
        weights = np.ones(n_assets) / n_assets

        # Iterative risk parity (simplified)
        for _ in range(10):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            contrib = weights * marginal_contrib

            # Update weights inversely proportional to risk contribution
            weights = (1 / marginal_contrib) / np.sum(1 / marginal_contrib)

        # Final risk contributions
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights / portfolio_vol
        risk_contrib = weights * marginal_contrib

        # Validate risk parity
        assert np.allclose(weights.sum(), 1.0)
        assert (weights > 0).all()  # Long-only
        assert np.std(risk_contrib) < 0.01  # Risk contributions are balanced

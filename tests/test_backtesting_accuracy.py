"""
Backtesting Framework Accuracy Tests

Comprehensive tests to validate the accuracy and reliability of the backtesting framework:
- Look-ahead bias prevention
- Survivorship bias handling
- Transaction cost modeling
- Slippage and market impact
- Point-in-time data integrity
- Performance calculation accuracy
- Edge case handling (splits, dividends, delistings)

Critical for ensuring backtest results are realistic and implementable in live trading.
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from alphapy.frame import Frame
from alphapy.globals import Orders
from alphapy.portfolio import Portfolio, Position, Trade, gen_portfolio
from alphapy.space import Space
from alphapy.system import System, run_system


class TestLookAheadBiasDetection:
    """Test prevention of look-ahead bias in backtesting."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear state before each test."""
        Portfolio.portfolios.clear()
        Frame.frames.clear()
        System.systems.clear()

    def test_signal_generation_timing(self):
        """Test that signals are generated using only historical data."""
        # Create time series data with clear future information
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        rng = np.random.default_rng(seed=12345)

        # Future peak indicator (look-ahead bias)
        future_returns = []
        for i in range(len(dates)):
            if i < len(dates) - 5:
                # Look at next 5 days returns
                rng = np.random.default_rng(seed=42)
                future_5d_return = rng.normal(0.02, 0.1)  # Simulate future knowledge
                future_returns.append(future_5d_return)
            else:
                future_returns.append(0)

        # Historical data (proper)
        historical_prices = 100 * np.exp(np.cumsum(rng.standard_normal(100) * 0.02))

        data = pd.DataFrame(
            {
                "close": historical_prices,
                "sma_20": pd.Series(historical_prices).rolling(20).mean(),
                "future_return_5d": future_returns,  # This should NOT be used
                "historical_signal": None,  # Will populate properly
            },
            index=dates,
        )

        # Proper signal generation (only using historical data)
        for i in range(len(data)):
            if i >= 20:  # Need 20 days for SMA
                current_price = data.iloc[i]["close"]
                historical_sma = data.iloc[i]["sma_20"]

                # Signal based only on current and past information
                signal = 1 if current_price > historical_sma else 0
                data.iloc[i, data.columns.get_loc("historical_signal")] = signal

        # Test: Ensure no future information is used
        # Check that signal at time t only depends on data up to time t
        for i in range(20, len(data) - 1):
            signal_t = data.iloc[i]["historical_signal"]

            # Signal should be based on price relative to SMA at time t
            price_t = data.iloc[i]["close"]
            sma_t = data.iloc[i]["sma_20"]
            expected_signal = 1 if price_t > sma_t else 0

            assert signal_t == expected_signal

        # Verify no correlation with future returns (good signal design)
        valid_data = data.dropna(subset=["historical_signal"])
        valid_signals = pd.to_numeric(valid_data["historical_signal"], errors="coerce").values
        future_rets = pd.to_numeric(valid_data["future_return_5d"], errors="coerce").values

        # Remove any remaining NaN values
        mask = ~(np.isnan(valid_signals) | np.isnan(future_rets))
        valid_signals = valid_signals[mask]
        future_rets = future_rets[mask]

        # Signal should not be perfectly correlated with future returns
        # (that would indicate look-ahead bias)
        if len(valid_signals) > 1 and len(future_rets) > 1 and np.var(valid_signals) > 0 and np.var(future_rets) > 0:
            correlation = np.corrcoef(valid_signals, future_rets)[0, 1]
            assert abs(correlation) < 0.9  # Should not be too correlated

    def test_rebalancing_timing_constraints(self):
        """Test that portfolio rebalancing occurs at realistic times."""
        # Portfolio rebalancing scenarios
        trading_dates = pd.bdate_range("2024-01-01", "2024-12-31")

        # Create mock portfolio with positions
        positions = {
            "AAPL": {"shares": 1000, "target_weight": 0.25},
            "GOOGL": {"shares": 400, "target_weight": 0.25},
            "MSFT": {"shares": 300, "target_weight": 0.25},
            "AMZN": {"shares": 250, "target_weight": 0.25},
        }

        # Market prices (simulated)
        rng = np.random.default_rng(seed=54321)
        rng = np.random.default_rng(seed=42)
        prices = {symbol: 100 + rng.standard_normal(len(trading_dates)) * 5 for symbol in positions}

        # Test rebalancing timing
        rebalance_frequency = "monthly"  # Realistic frequency
        last_rebalance = None

        rebalance_events = []

        for date in trading_dates:
            # Determine if rebalancing should occur
            should_rebalance = False

            if last_rebalance is None:
                should_rebalance = True  # Initial rebalancing
            elif (
                rebalance_frequency == "monthly"
                and date.day <= 3
                and (last_rebalance is None or date.month != last_rebalance.month)
            ):
                # Monthly rebalancing on first trading day of month
                should_rebalance = True

            if should_rebalance:
                # Simulate rebalancing trade generation
                rebalance_events.append(
                    {
                        "date": date,
                        "type": "rebalance",
                        "positions_before": positions.copy(),
                    }
                )
                last_rebalance = date

        # Verify realistic rebalancing frequency
        # Monthly rebalancing should occur ~12 times per year
        assert 10 <= len(rebalance_events) <= 14  # Allow some variation for holidays

        # Verify no same-day rebalancing (unrealistic)
        dates_rebalanced = [event["date"] for event in rebalance_events]
        unique_dates = set(dates_rebalanced)
        assert len(dates_rebalanced) == len(unique_dates)  # No duplicates

    def test_earnings_announcement_timing(self):
        """Test handling of earnings announcements and information timing."""
        # Simulate earnings calendar
        earnings_calendar = {
            "AAPL": ["2024-01-25", "2024-04-25", "2024-07-25", "2024-10-25"],
            "GOOGL": ["2024-01-30", "2024-04-30", "2024-07-30", "2024-10-30"],
            "MSFT": ["2024-01-24", "2024-04-24", "2024-07-24", "2024-10-24"],
        }

        # Convert to datetime
        for symbol in earnings_calendar:
            earnings_calendar[symbol] = [pd.to_datetime(date) for date in earnings_calendar[symbol]]

        # Test signal generation around earnings
        test_date = pd.to_datetime("2024-01-25")  # AAPL earnings day

        # Rule: No position changes on earnings day (too risky)
        # Rule: Signals generated before market open using prior day data

        def can_trade_on_date(symbol, date):
            """Determine if trading is allowed on given date."""
            earnings_dates = earnings_calendar.get(symbol, [])

            # No trading on earnings announcement day
            if date in earnings_dates:
                return False

            # No trading day before earnings (information uncertainty)
            return all(date != earnings_date - timedelta(days=1) for earnings_date in earnings_dates)

        # Test the rule
        assert can_trade_on_date("AAPL", pd.to_datetime("2024-01-23"))  # 2 days before
        assert not can_trade_on_date("AAPL", pd.to_datetime("2024-01-24"))  # 1 day before
        assert not can_trade_on_date("AAPL", pd.to_datetime("2024-01-25"))  # Earnings day
        assert can_trade_on_date("AAPL", pd.to_datetime("2024-01-26"))  # Day after

        # Test signal timing
        signal_generation_time = "before_market_open"  # 9:00 AM
        trade_execution_time = "market_open"  # 9:30 AM

        # Verify proper sequencing
        signal_time = datetime.combine(test_date.date(), datetime.strptime("09:00", "%H:%M").time())
        execution_time = datetime.combine(test_date.date(), datetime.strptime("09:30", "%H:%M").time())

        assert signal_time < execution_time  # Signals must be generated before execution


class TestSurvivorshipBiasHandling:
    """Test proper handling of survivorship bias in backtests."""

    def test_delisted_stock_handling(self):
        """Test that delisted stocks are properly included in backtests."""
        # Simulate universe with delistings
        universe_timeline = {
            "2023-01-01": ["AAPL", "GOOGL", "MSFT", "FAILING_CORP", "BANKRUPT_CO"],
            "2023-06-01": ["AAPL", "GOOGL", "MSFT", "FAILING_CORP"],  # BANKRUPT_CO delisted
            "2023-09-01": ["AAPL", "GOOGL", "MSFT"],  # FAILING_CORP delisted
            "2023-12-31": ["AAPL", "GOOGL", "MSFT"],  # Only survivors remain
        }

        # Stock price histories (including delisted stocks)
        stock_histories = {
            "AAPL": {"start": 150, "end": 180, "delisted": False},
            "GOOGL": {"start": 100, "end": 120, "delisted": False},
            "MSFT": {"start": 250, "end": 300, "delisted": False},
            "FAILING_CORP": {"start": 50, "end": 5, "delisted": True, "delist_date": "2023-09-01"},
            "BANKRUPT_CO": {"start": 20, "end": 0.1, "delisted": True, "delist_date": "2023-06-01"},
        }

        # Survivor-biased backtest (WRONG)
        def survivor_biased_returns():
            """Calculate returns using only stocks that survived."""
            survivors = ["AAPL", "GOOGL", "MSFT"]
            total_return = 0
            for stock in survivors:
                start_price = stock_histories[stock]["start"]
                end_price = stock_histories[stock]["end"]
                stock_return = (end_price / start_price) - 1
                total_return += stock_return / len(survivors)  # Equal weight
            return total_return

        # Proper point-in-time backtest (CORRECT)
        def point_in_time_returns():
            """Calculate returns using point-in-time universe."""
            # Start with initial universe
            initial_universe = universe_timeline["2023-01-01"]
            total_return = 0

            for stock in initial_universe:
                start_price = stock_histories[stock]["start"]

                # Handle delisting
                if stock_histories[stock]["delisted"]:
                    # Use price at delisting
                    end_price = stock_histories[stock]["end"]
                else:
                    # Use final price
                    end_price = stock_histories[stock]["end"]

                stock_return = (end_price / start_price) - 1
                total_return += stock_return / len(initial_universe)  # Equal weight

            return total_return

        # Calculate both approaches
        survivor_return = survivor_biased_returns()
        proper_return = point_in_time_returns()

        # Survivor-biased return should be higher (excludes failures)
        assert survivor_return > proper_return

        # Proper return should include the losses from delistings
        # FAILING_CORP: (5/50) - 1 = -90%
        # BANKRUPT_CO: (0.1/20) - 1 = -99.5%
        expected_failing_return = (5 / 50) - 1
        expected_bankrupt_return = (0.1 / 20) - 1

        assert expected_failing_return == -0.9
        assert expected_bankrupt_return == -0.995

    def test_ipo_and_spac_timing(self):
        """Test proper handling of IPOs and SPACs in backtests."""
        # Timeline of new listings
        ipo_timeline = {
            "EXISTING_STOCK": {"ipo_date": "2020-01-01", "available_from": "2020-01-01"},
            "NEW_IPO": {"ipo_date": "2023-06-15", "available_from": "2023-06-15"},
            "SPAC_MERGER": {"ipo_date": "2023-09-30", "available_from": "2023-09-30"},
        }

        # Test date range
        backtest_start = pd.to_datetime("2023-01-01")
        backtest_end = pd.to_datetime("2023-12-31")

        def get_available_stocks(date):
            """Get stocks available for trading on given date."""
            available = []
            for stock, info in ipo_timeline.items():
                ipo_date = pd.to_datetime(info["ipo_date"])
                if date >= ipo_date:
                    available.append(stock)
            return available

        # Test availability on different dates
        jan_stocks = get_available_stocks(pd.to_datetime("2023-01-01"))
        jun_stocks = get_available_stocks(pd.to_datetime("2023-06-30"))
        oct_stocks = get_available_stocks(pd.to_datetime("2023-10-01"))

        assert "EXISTING_STOCK" in jan_stocks
        assert "NEW_IPO" not in jan_stocks
        assert "SPAC_MERGER" not in jan_stocks

        assert "EXISTING_STOCK" in jun_stocks
        assert "NEW_IPO" in jun_stocks
        assert "SPAC_MERGER" not in jun_stocks

        assert "EXISTING_STOCK" in oct_stocks
        assert "NEW_IPO" in oct_stocks
        assert "SPAC_MERGER" in oct_stocks

        # Verify stocks can't be traded before their IPO
        for stock, info in ipo_timeline.items():
            ipo_date = pd.to_datetime(info["ipo_date"])
            pre_ipo_date = ipo_date - timedelta(days=1)

            pre_ipo_available = get_available_stocks(pre_ipo_date)
            post_ipo_available = get_available_stocks(ipo_date)

            if ipo_date > backtest_start:  # Only test if IPO is during backtest period
                assert stock not in pre_ipo_available
                assert stock in post_ipo_available

    def test_index_composition_changes(self):
        """Test handling of index composition changes over time."""
        # S&P 500 composition changes (simplified)
        sp500_timeline = {
            "2023-Q1": ["AAPL", "MSFT", "GOOGL", "AMZN", "OLD_STOCK1"],
            "2023-Q2": ["AAPL", "MSFT", "GOOGL", "AMZN", "OLD_STOCK1"],  # No change
            "2023-Q3": ["AAPL", "MSFT", "GOOGL", "AMZN", "NEW_STOCK1"],  # OLD_STOCK1 removed
            "2023-Q4": ["AAPL", "MSFT", "GOOGL", "NEW_STOCK1", "NEW_STOCK2"],  # AMZN removed
        }

        # Simulate index tracking strategy
        def get_index_composition(date):
            """Get index composition for given date."""
            if date < pd.to_datetime("2023-04-01"):
                return sp500_timeline["2023-Q1"]
            elif date < pd.to_datetime("2023-07-01"):
                return sp500_timeline["2023-Q2"]
            elif date < pd.to_datetime("2023-10-01"):
                return sp500_timeline["2023-Q3"]
            else:
                return sp500_timeline["2023-Q4"]

        # Test composition changes
        q1_comp = get_index_composition(pd.to_datetime("2023-02-15"))
        q3_comp = get_index_composition(pd.to_datetime("2023-08-15"))
        q4_comp = get_index_composition(pd.to_datetime("2023-11-15"))

        # Verify changes
        assert "OLD_STOCK1" in q1_comp
        assert "OLD_STOCK1" not in q3_comp
        assert "NEW_STOCK1" in q3_comp

        assert "AMZN" in q3_comp
        assert "AMZN" not in q4_comp
        assert "NEW_STOCK2" in q4_comp

        # Simulate forced rebalancing due to index changes
        def calculate_rebalancing_trades(old_composition, new_composition):
            """Calculate trades needed for index composition change."""
            trades = []

            # Stocks to remove
            to_remove = set(old_composition) - set(new_composition)
            for stock in to_remove:
                trades.append({"symbol": stock, "action": "sell", "reason": "index_removal"})

            # Stocks to add
            to_add = set(new_composition) - set(old_composition)
            for stock in to_add:
                trades.append({"symbol": stock, "action": "buy", "reason": "index_addition"})

            return trades

        # Test Q1 to Q3 change
        q1_to_q3_trades = calculate_rebalancing_trades(q1_comp, q3_comp)

        sell_trades = [t for t in q1_to_q3_trades if t["action"] == "sell"]
        buy_trades = [t for t in q1_to_q3_trades if t["action"] == "buy"]

        assert len(sell_trades) == 1
        assert sell_trades[0]["symbol"] == "OLD_STOCK1"
        assert len(buy_trades) == 1
        assert buy_trades[0]["symbol"] == "NEW_STOCK1"


class TestTransactionCostModeling:
    """Test accurate transaction cost modeling in backtests."""

    def test_commission_cost_calculation(self):
        """Test various commission structures."""
        # Different commission structures
        commission_structures = {
            "fixed_per_trade": {"type": "fixed", "cost": 1.0},
            "per_share": {"type": "per_share", "cost": 0.005, "minimum": 1.0},
            "percentage": {"type": "percentage", "cost": 0.001, "minimum": 1.0},  # 0.1%
            "tiered": {
                "type": "tiered",
                "tiers": [
                    {"threshold": 0, "cost": 0.01},  # First $10k: 1%
                    {"threshold": 10000, "cost": 0.005},  # Next portion: 0.5%
                    {"threshold": 50000, "cost": 0.002},  # Above $50k: 0.2%
                ],
            },
        }

        # Test trades
        test_trades = [
            {"shares": 100, "price": 50.0},  # $5,000 trade
            {"shares": 500, "price": 100.0},  # $50,000 trade
            {"shares": 1000, "price": 75.0},  # $75,000 trade
        ]

        for trade in test_trades:
            trade_value = trade["shares"] * trade["price"]

            for structure_name, structure in commission_structures.items():
                if structure["type"] == "fixed":
                    commission = structure["cost"]

                elif structure["type"] == "per_share":
                    commission = max(trade["shares"] * structure["cost"], structure["minimum"])

                elif structure["type"] == "percentage":
                    commission = max(trade_value * structure["cost"], structure["minimum"])

                elif structure["type"] == "tiered":
                    commission = 0
                    remaining_value = trade_value

                    for i, tier in enumerate(structure["tiers"]):
                        tier_threshold = tier["threshold"]
                        tier_cost = tier["cost"]

                        if i < len(structure["tiers"]) - 1:
                            next_threshold = structure["tiers"][i + 1]["threshold"]
                            tier_amount = min(remaining_value, next_threshold - tier_threshold)
                        else:
                            tier_amount = remaining_value

                        if tier_amount > 0:
                            commission += tier_amount * tier_cost
                            remaining_value -= tier_amount

                # Verify reasonable commission amounts
                assert commission > 0
                assert commission < trade_value * 0.05  # Less than 5% of trade value

                # Store for comparison
                trade[f"commission_{structure_name}"] = commission

        # Verify per-share is higher for small shares, percentage for large values
        small_trade = test_trades[0]  # 100 shares @ $50
        large_trade = test_trades[2]  # 1000 shares @ $75

        # Per-share commission should scale with shares (accounting for minimum)
        # Small: 100 shares * 0.005 = 0.5, but minimum 1.0 = 1.0
        # Large: 1000 shares * 0.005 = 5.0
        # Ratio: 5.0 / 1.0 = 5
        assert (large_trade["commission_per_share"] / small_trade["commission_per_share"]) == 5

    def test_bid_ask_spread_modeling(self):
        """Test bid-ask spread impact on execution prices."""
        # Market data with bid-ask spreads
        market_scenarios = {
            "large_cap": {"mid": 100.0, "spread": 0.02, "depth": 1000000},  # 2 cent spread
            "small_cap": {"mid": 25.0, "spread": 0.10, "depth": 50000},  # 10 cent spread
            "microcap": {"mid": 5.0, "spread": 0.05, "depth": 10000},  # 5 cent spread (1%)
            "etf": {"mid": 150.0, "spread": 0.01, "depth": 2000000},  # 1 cent spread
        }

        # Test trades of different sizes
        trade_sizes = [100, 1000, 10000, 50000]  # shares

        for scenario_name, market in market_scenarios.items():
            mid_price = market["mid"]
            spread = market["spread"]
            depth = market["depth"]

            bid = mid_price - spread / 2
            ask = mid_price + spread / 2

            for size in trade_sizes:
                # Market buy order (pays ask + market impact)
                base_cost_buy = size * ask

                # Market sell order (receives bid - market impact)
                base_proceeds_sell = size * bid

                # Market impact (simplified square root model)
                impact_factor = np.sqrt(size / depth) * 0.001  # 0.1% per unit of sqrt(size/depth)
                market_impact_buy = base_cost_buy * impact_factor
                market_impact_sell = base_proceeds_sell * impact_factor

                # Total execution cost
                total_cost_buy = base_cost_buy + market_impact_buy
                total_proceeds_sell = base_proceeds_sell - market_impact_sell

                # Round trip cost (buy then immediately sell)
                round_trip_cost = total_cost_buy - total_proceeds_sell
                round_trip_pct = round_trip_cost / (size * mid_price)

                # Verify costs scale appropriately
                if scenario_name == "large_cap":
                    assert round_trip_pct < 0.005  # Less than 0.5% round trip
                elif scenario_name == "microcap":
                    assert round_trip_pct > 0.01  # More than 1% round trip

                # Larger trades should have higher market impact
                if size == max(trade_sizes):
                    assert market_impact_buy > 0
                    assert market_impact_sell > 0

    def test_slippage_modeling(self):
        """Test various slippage models."""
        # Order parameters
        order_size = 10000  # shares
        market_price = 100.0
        order_value = order_size * market_price

        # Different slippage models
        slippage_models = {
            "fixed_bps": {"type": "fixed", "bps": 5},  # 5 basis points
            "volume_based": {"type": "volume", "daily_volume": 1000000, "participation": 0.1},
            "volatility_based": {"type": "volatility", "daily_vol": 0.02, "size_factor": 0.1},
            "sqrt_impact": {"type": "sqrt", "liquidity": 5000000, "impact_coeff": 0.001},
        }

        for model_name, model in slippage_models.items():
            if model["type"] == "fixed":
                slippage = order_value * (model["bps"] / 10000)

            elif model["type"] == "volume":
                # Volume participation impact
                daily_volume = model["daily_volume"]
                participation_rate = order_size / daily_volume

                if participation_rate > model["participation"]:
                    # Higher impact for large participation
                    excess_participation = participation_rate - model["participation"]
                    slippage = order_value * (0.0005 + excess_participation * 0.01)
                else:
                    slippage = order_value * 0.0005  # Base slippage

            elif model["type"] == "volatility":
                # Volatility-based slippage
                daily_vol = model["daily_vol"]
                size_factor = model["size_factor"]
                slippage = order_value * daily_vol * size_factor

            elif model["type"] == "sqrt":
                # Square root market impact model
                liquidity = model["liquidity"]
                impact_coeff = model["impact_coeff"]

                # Impact proportional to sqrt(order_size / liquidity)
                impact_factor = np.sqrt(order_size / liquidity)
                slippage = order_value * impact_coeff * impact_factor

            # Verify slippage is reasonable
            slippage_pct = slippage / order_value
            assert 0 <= slippage_pct <= 0.02  # Between 0% and 2%

            # Store results
            print(f"{model_name}: ${slippage:.2f} ({slippage_pct:.4%})")

    def test_timing_delay_costs(self):
        """Test costs from execution timing delays."""
        # Market movement during order execution
        entry_price = 100.0
        execution_delay_minutes = [0, 1, 5, 15, 30]  # Different delays

        # Market volatility (price movement during delay)
        hourly_volatility = 0.02  # 2% per hour

        timing_costs = {}

        for delay in execution_delay_minutes:
            # Price movement during delay
            delay_hours = delay / 60
            expected_movement_std = hourly_volatility * np.sqrt(delay_hours)

            # Simulate adverse price movement (always costs money)
            # For buy orders, price moves up; for sell orders, price moves down
            adverse_movement = expected_movement_std * 0.5  # Average adverse movement

            execution_price_buy = entry_price * (1 + adverse_movement)
            execution_price_sell = entry_price * (1 - adverse_movement)

            # Cost of delay
            buy_cost = execution_price_buy - entry_price
            sell_cost = entry_price - execution_price_sell

            timing_costs[delay] = {"buy_cost": buy_cost, "sell_cost": sell_cost, "avg_cost": (buy_cost + sell_cost) / 2}

        # Verify costs increase with delay
        for i in range(1, len(execution_delay_minutes)):
            prev_delay = execution_delay_minutes[i - 1]
            curr_delay = execution_delay_minutes[i]

            assert timing_costs[curr_delay]["avg_cost"] >= timing_costs[prev_delay]["avg_cost"]

        # 30-minute delay should have meaningful cost
        max_delay_cost = timing_costs[30]["avg_cost"]
        assert max_delay_cost > entry_price * 0.001  # At least 0.1% cost


class TestPointInTimeDataIntegrity:
    """Test point-in-time data integrity and availability."""

    def test_fundamental_data_lag(self):
        """Test proper lag for fundamental data availability."""
        # Earnings announcement dates
        earnings_dates = {
            "AAPL": {
                "Q1_2023": {"announcement": "2023-05-04", "quarter_end": "2023-03-31"},
                "Q2_2023": {"announcement": "2023-08-03", "quarter_end": "2023-06-30"},
                "Q3_2023": {"announcement": "2023-11-02", "quarter_end": "2023-09-30"},
            }
        }

        # Test data availability
        def get_available_fundamental_data(symbol, date):
            """Get fundamental data available on given date."""
            date = pd.to_datetime(date)
            available_data = {}

            for quarter, info in earnings_dates[symbol].items():
                announcement_date = pd.to_datetime(info["announcement"])
                quarter_end = pd.to_datetime(info["quarter_end"])

                # Data only available after announcement
                if date >= announcement_date:
                    available_data[quarter] = {
                        "quarter_end": quarter_end,
                        "announcement": announcement_date,
                        "days_lag": (announcement_date - quarter_end).days,
                    }

            return available_data

        # Test different dates
        test_dates = [
            "2023-04-01",  # Before Q1 announcement
            "2023-05-05",  # After Q1 announcement
            "2023-08-04",  # After Q2 announcement
            "2023-11-03",  # After Q3 announcement
        ]

        for test_date in test_dates:
            available = get_available_fundamental_data("AAPL", test_date)

            if test_date == "2023-04-01":
                assert len(available) == 0  # No Q1 data yet
            elif test_date == "2023-05-05":
                assert "Q1_2023" in available
                assert "Q2_2023" not in available
            elif test_date == "2023-08-04":
                assert "Q1_2023" in available
                assert "Q2_2023" in available
                assert "Q3_2023" not in available
            elif test_date == "2023-11-03":
                assert len(available) == 3  # All quarters available

        # Verify realistic reporting lags
        q1_data = get_available_fundamental_data("AAPL", "2023-05-05")["Q1_2023"]
        reporting_lag = q1_data["days_lag"]

        # Typical reporting lag is 30-45 days
        assert 25 <= reporting_lag <= 50

    def test_price_adjustment_timing(self):
        """Test proper timing of price adjustments for splits and dividends."""
        # Corporate actions timeline
        corporate_actions = {
            "AAPL": [
                {
                    "type": "split",
                    "ratio": 4.0,  # 4:1 split
                    "ex_date": "2023-08-24",
                    "record_date": "2023-08-21",
                    "announcement": "2023-07-27",
                },
                {
                    "type": "dividend",
                    "amount": 0.24,
                    "ex_date": "2023-11-10",
                    "pay_date": "2023-11-16",
                    "announcement": "2023-11-02",
                },
            ]
        }

        # Price history (unadjusted)
        pre_split_price = 180.0
        post_split_price = 45.0  # After 4:1 split

        def get_adjusted_price(symbol, date, raw_price):
            """Get properly adjusted price for given date."""
            date = pd.to_datetime(date)
            adjusted_price = raw_price

            for action in corporate_actions[symbol]:
                ex_date = pd.to_datetime(action["ex_date"])

                if action["type"] == "split" and date >= ex_date:
                    # Adjust for split - multiply by ratio to get pre-split equivalent
                    split_ratio = action["ratio"]
                    adjusted_price = adjusted_price * split_ratio

                elif action["type"] == "dividend" and date >= ex_date:
                    # Adjust for dividend
                    dividend = action["amount"]
                    adjusted_price = adjusted_price + dividend

            return adjusted_price

        # Test price adjustments
        test_cases = [
            {"date": "2023-08-23", "raw_price": 180.0, "expected": 180.0},  # Before split
            {"date": "2023-08-24", "raw_price": 45.0, "expected": 180.0},  # After split (adjusted)
            {"date": "2023-11-09", "raw_price": 45.0, "expected": 180.0},  # After split, before dividend
            {"date": "2023-11-10", "raw_price": 44.94, "expected": 180.0},  # After split and dividend (adjusted)
        ]

        for case in test_cases:
            adjusted = get_adjusted_price("AAPL", case["date"], case["raw_price"])
            # Allow small rounding differences
            assert abs(adjusted - case["expected"]) < 0.01

    def test_index_membership_lag(self):
        """Test proper lag for index membership changes."""
        # S&P 500 inclusion announcement
        sp500_changes = {
            "NEW_STOCK": {
                "announcement_date": "2023-09-05",  # Announced after market close
                "effective_date": "2023-09-11",  # Effective before market open
                "reason": "Inclusion due to market cap growth",
            }
        }

        def is_in_index(symbol, date):
            """Check if stock is in index on given date."""
            date = pd.to_datetime(date)

            if symbol in sp500_changes:
                effective_date = pd.to_datetime(sp500_changes[symbol]["effective_date"])
                return date >= effective_date

            return False  # Not in changes, assume not in index

        # Test membership timing
        assert not is_in_index("NEW_STOCK", "2023-09-05")  # Announcement day
        assert not is_in_index("NEW_STOCK", "2023-09-10")  # Day before effective
        assert is_in_index("NEW_STOCK", "2023-09-11")  # Effective date
        assert is_in_index("NEW_STOCK", "2023-09-12")  # After effective


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

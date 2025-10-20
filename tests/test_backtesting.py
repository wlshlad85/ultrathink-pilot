#!/usr/bin/env python3
"""
Comprehensive tests for backtesting framework.
Tests data fetcher, portfolio, metrics, and backtest engine.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtesting.data_fetcher import DataFetcher
from backtesting.portfolio import Portfolio, Trade, Position
from backtesting.metrics import PerformanceMetrics
from backtesting.backtest_engine import BacktestEngine


class TestDataFetcher:
    """Test DataFetcher class."""

    def test_initialization(self):
        """Test DataFetcher can be initialized."""
        fetcher = DataFetcher("BTC-USD")
        assert fetcher.symbol == "BTC-USD"
        assert fetcher.data is None

    def test_fetch_data(self):
        """Test fetching historical data."""
        fetcher = DataFetcher("BTC-USD")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        df = fetcher.fetch(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        assert not df.empty
        assert 'close' in df.columns
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'volume' in df.columns
        assert len(df) > 0

    def test_add_technical_indicators(self):
        """Test adding technical indicators."""
        fetcher = DataFetcher("BTC-USD")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        fetcher.fetch(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        df = fetcher.add_technical_indicators()

        # Check indicators exist
        indicators = ['sma_20', 'sma_50', 'rsi_14', 'atr_14', 'macd',
                     'bb_upper', 'bb_lower', 'returns_1d']
        for ind in indicators:
            assert ind in df.columns, f"Missing indicator: {ind}"

        # Check indicators are calculated (not all NaN)
        assert df['rsi_14'].notna().sum() > 0
        assert df['sma_20'].notna().sum() > 0

    def test_get_market_context(self):
        """Test getting market context."""
        fetcher = DataFetcher("BTC-USD")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        fetcher.fetch(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        fetcher.add_technical_indicators()

        context = fetcher.get_market_context(250)

        assert 'date' in context
        assert 'price' in context
        assert 'indicators' in context
        assert 'returns' in context
        assert 'sentiment' in context
        assert context['price'] > 0


class TestPortfolio:
    """Test Portfolio class."""

    def test_initialization(self):
        """Test Portfolio initialization."""
        portfolio = Portfolio(initial_capital=100000.0)
        assert portfolio.initial_capital == 100000.0
        assert portfolio.cash == 100000.0
        assert portfolio.get_total_value() == 100000.0

    def test_buy_execution(self):
        """Test buy order execution."""
        portfolio = Portfolio(initial_capital=100000.0)

        trade = portfolio.execute_trade(
            action="BUY",
            price=50000.0,
            timestamp="2024-01-01",
            risk_percent=1.0
        )

        assert trade is not None
        assert trade.action == "BUY"
        assert portfolio.position.quantity > 0
        assert portfolio.cash < 100000.0
        assert portfolio.total_trades == 1

    def test_sell_execution(self):
        """Test sell order execution."""
        portfolio = Portfolio(initial_capital=100000.0)

        # Buy first
        portfolio.execute_trade("BUY", 50000.0, "2024-01-01", 1.0)
        initial_quantity = portfolio.position.quantity

        # Then sell
        trade = portfolio.execute_trade("SELL", 55000.0, "2024-01-02")

        assert trade is not None
        assert trade.action == "SELL"
        assert portfolio.position.quantity == 0
        assert portfolio.get_total_value() > 100000.0  # Made profit

    def test_hold_execution(self):
        """Test hold action."""
        portfolio = Portfolio(initial_capital=100000.0)

        trade = portfolio.execute_trade("HOLD", 50000.0, "2024-01-01")

        assert trade is None
        assert portfolio.position.quantity == 0
        assert portfolio.cash == 100000.0

    def test_insufficient_capital(self):
        """Test trade with insufficient capital."""
        portfolio = Portfolio(initial_capital=1000.0)

        trade = portfolio.execute_trade("BUY", 100000.0, "2024-01-01", 100.0)

        assert trade is None  # Should fail

    def test_sell_without_position(self):
        """Test sell without position."""
        portfolio = Portfolio(initial_capital=100000.0)

        trade = portfolio.execute_trade("SELL", 50000.0, "2024-01-01")

        assert trade is None  # Should fail

    def test_commission_applied(self):
        """Test that commissions are applied."""
        portfolio = Portfolio(initial_capital=100000.0, commission_rate=0.001)

        portfolio.execute_trade("BUY", 50000.0, "2024-01-01", 1.0)

        assert portfolio.total_commission_paid > 0

    def test_equity_recording(self):
        """Test equity curve recording."""
        portfolio = Portfolio(initial_capital=100000.0)

        portfolio.record_equity("2024-01-01")
        portfolio.record_equity("2024-01-02")

        equity_df = portfolio.get_equity_dataframe()
        assert len(equity_df) == 2
        assert 'total_value' in equity_df.columns


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""

    @pytest.fixture
    def sample_equity_data(self):
        """Create sample equity curve."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        values = 100000 * (1 + np.random.randn(100).cumsum() * 0.01)

        df = pd.DataFrame({
            'timestamp': dates,
            'total_value': values,
            'cash': values * 0.5,
            'position_value': values * 0.5,
            'position_quantity': 0.1,
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'returns': values / 100000 - 1
        })
        return df

    def test_sharpe_ratio_calculation(self, sample_equity_data):
        """Test Sharpe ratio calculation."""
        metrics = PerformanceMetrics(sample_equity_data)
        sharpe = metrics.calculate_sharpe_ratio()

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_max_drawdown_calculation(self, sample_equity_data):
        """Test max drawdown calculation."""
        metrics = PerformanceMetrics(sample_equity_data)
        max_dd, dd_info = metrics.calculate_max_drawdown()

        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero
        assert 'peak_date' in dd_info
        assert 'trough_date' in dd_info

    def test_volatility_calculation(self, sample_equity_data):
        """Test volatility calculation."""
        metrics = PerformanceMetrics(sample_equity_data)
        vol = metrics.calculate_volatility()

        assert isinstance(vol, float)
        assert vol >= 0

    def test_var_calculation(self, sample_equity_data):
        """Test VaR calculation."""
        metrics = PerformanceMetrics(sample_equity_data)
        var = metrics.calculate_var(0.95)

        assert isinstance(var, float)

    def test_get_all_metrics(self, sample_equity_data):
        """Test getting all metrics."""
        metrics = PerformanceMetrics(sample_equity_data)
        all_metrics = metrics.get_all_metrics()

        required_metrics = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct',
                          'volatility_pct', 'sortino_ratio']
        for metric in required_metrics:
            assert metric in all_metrics


class TestBacktestEngine:
    """Test BacktestEngine class."""

    def test_initialization(self):
        """Test BacktestEngine initialization."""
        engine = BacktestEngine(
            symbol="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-02-01",
            initial_capital=100000.0
        )

        assert engine.symbol == "BTC-USD"
        assert engine.initial_capital == 100000.0

    def test_load_data(self):
        """Test loading market data."""
        engine = BacktestEngine(
            symbol="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-02-01"
        )

        engine.load_data()

        assert engine.market_data is not None
        assert len(engine.market_data) > 0

    def test_create_fixture_from_context(self):
        """Test creating fixture from market context."""
        engine = BacktestEngine(
            symbol="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-02-01"
        )

        engine.load_data()
        # Use a valid index based on actual data length
        # Need at least 50 data points for technical indicators
        data_len = len(engine.data_fetcher.data)
        valid_index = min(data_len - 1, max(50, data_len // 2))

        context = engine.data_fetcher.get_market_context(valid_index)
        fixture = engine.create_fixture_from_context(context, "2024-01-01")

        assert 'task' in fixture
        assert 'input_mr_sr' in fixture
        assert 'expected' in fixture


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

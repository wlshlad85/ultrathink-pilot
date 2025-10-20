#!/usr/bin/env python3
"""
Performance metrics calculation for backtesting.
Includes Sharpe ratio, max drawdown, and other key metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Calculate various performance metrics from backtest results."""

    def __init__(self, equity_df: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator.

        Args:
            equity_df: DataFrame with equity curve (must have 'total_value' and 'timestamp')
            risk_free_rate: Annual risk-free rate for Sharpe calculation (default 2%)
        """
        self.equity_df = equity_df.copy()
        self.risk_free_rate = risk_free_rate

        # Handle empty DataFrame gracefully
        if equity_df.empty:
            logger.warning("Empty equity DataFrame provided - metrics will return default values")
            # Create a minimal DataFrame with a single row for compatibility
            self.equity_df = pd.DataFrame([{
                'total_value': 0.0,
                'timestamp': pd.Timestamp.now()
            }])
        elif 'total_value' not in equity_df.columns:
            raise ValueError("equity_df must have 'total_value' column")

    def calculate_returns(self) -> pd.Series:
        """Calculate daily returns from equity curve."""
        returns = self.equity_df['total_value'].pct_change().fillna(0)
        return returns

    def calculate_sharpe_ratio(self, periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            periods_per_year: Trading periods per year (252 for daily, 12 for monthly)

        Returns:
            Sharpe ratio
        """
        returns = self.calculate_returns()

        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        # Annualize returns and volatility
        mean_return = returns.mean() * periods_per_year
        std_return = returns.std() * np.sqrt(periods_per_year)

        # Calculate Sharpe ratio
        sharpe = (mean_return - self.risk_free_rate) / std_return
        return sharpe

    def calculate_sortino_ratio(self, periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sortino ratio (uses downside deviation).

        Args:
            periods_per_year: Trading periods per year

        Returns:
            Sortino ratio
        """
        returns = self.calculate_returns()

        if len(returns) < 2:
            return 0.0

        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')

        downside_std = downside_returns.std() * np.sqrt(periods_per_year)
        if downside_std == 0:
            return float('inf')

        mean_return = returns.mean() * periods_per_year
        sortino = (mean_return - self.risk_free_rate) / downside_std

        return sortino

    def calculate_max_drawdown(self) -> Tuple[float, Dict]:
        """
        Calculate maximum drawdown and related metrics.

        Returns:
            (max_drawdown_pct, drawdown_info)
        """
        equity = self.equity_df['total_value']

        # Handle insufficient data
        if len(equity) < 2:
            return 0.0, {
                "max_drawdown_pct": 0.0,
                "peak_date": "N/A",
                "trough_date": "N/A",
                "recovery_date": "N/A",
                "peak_value": 0.0,
                "trough_value": 0.0
            }

        # Calculate running maximum
        running_max = equity.expanding().max()

        # Calculate drawdown
        drawdown = (equity - running_max) / running_max

        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()

        # Find peak before drawdown (handle edge case of empty slice)
        equity_before = equity[:max_dd_idx]
        if len(equity_before) == 0:
            peak_idx = 0
        else:
            peak_idx = equity_before.idxmax()

        # Find recovery point (if any)
        recovery_idx = None
        if max_dd_idx < len(equity) - 1:
            future_equity = equity[max_dd_idx:]
            peak_value = equity[peak_idx]
            recovery_points = future_equity[future_equity >= peak_value]
            if len(recovery_points) > 0:
                recovery_idx = recovery_points.index[0]

        drawdown_info = {
            "max_drawdown_pct": max_dd * 100,
            "peak_date": str(self.equity_df.iloc[peak_idx]['timestamp']),
            "trough_date": str(self.equity_df.iloc[max_dd_idx]['timestamp']),
            "recovery_date": str(self.equity_df.iloc[recovery_idx]['timestamp']) if recovery_idx is not None else "Not recovered",
            "peak_value": float(equity[peak_idx]),
            "trough_value": float(equity[max_dd_idx])
        }

        return max_dd * 100, drawdown_info

    def calculate_calmar_ratio(self, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).

        Args:
            periods_per_year: Trading periods per year

        Returns:
            Calmar ratio
        """
        returns = self.calculate_returns()
        annual_return = returns.mean() * periods_per_year

        max_dd, _ = self.calculate_max_drawdown()
        max_dd = abs(max_dd) / 100  # Convert to decimal

        if max_dd == 0:
            return float('inf')

        calmar = annual_return / max_dd
        return calmar

    def calculate_volatility(self, periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility.

        Args:
            periods_per_year: Trading periods per year

        Returns:
            Annualized volatility (%)
        """
        returns = self.calculate_returns()
        volatility = returns.std() * np.sqrt(periods_per_year) * 100
        return volatility

    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk at given confidence level.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR as percentage
        """
        returns = self.calculate_returns()
        var = np.percentile(returns, (1 - confidence_level) * 100) * 100
        return var

    def calculate_cvar(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            confidence_level: Confidence level

        Returns:
            CVaR as percentage
        """
        returns = self.calculate_returns()
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = returns[returns <= var_threshold].mean() * 100
        return cvar

    def calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """
        Calculate profit factor from trades.

        Args:
            trades_df: DataFrame with trade history

        Returns:
            Profit factor (gross profits / gross losses)
        """
        if trades_df.empty or 'action' not in trades_df.columns:
            return 0.0

        # Calculate P&L for each trade pair (BUY followed by SELL)
        gross_profit = 0.0
        gross_loss = 0.0

        buy_price = None
        buy_quantity = None

        for _, trade in trades_df.iterrows():
            if trade['action'] == 'BUY':
                buy_price = trade['price']
                buy_quantity = trade['quantity']
            elif trade['action'] == 'SELL' and buy_price is not None:
                pnl = (trade['price'] - buy_price) * buy_quantity - trade['commission']
                if pnl > 0:
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)
                buy_price = None
                buy_quantity = None

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def get_all_metrics(self, trades_df: pd.DataFrame = None) -> Dict:
        """
        Calculate all performance metrics.

        Args:
            trades_df: Optional DataFrame with trade history for trade-based metrics

        Returns:
            Dictionary with all metrics
        """
        max_dd_pct, dd_info = self.calculate_max_drawdown()

        metrics = {
            # Return metrics
            "total_return_pct": (
                (self.equity_df['total_value'].iloc[-1] / self.equity_df['total_value'].iloc[0] - 1) * 100
                if len(self.equity_df) > 0 else 0
            ),
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "sortino_ratio": self.calculate_sortino_ratio(),
            "calmar_ratio": self.calculate_calmar_ratio(),

            # Risk metrics
            "max_drawdown_pct": max_dd_pct,
            "volatility_pct": self.calculate_volatility(),
            "var_95_pct": self.calculate_var(0.95),
            "cvar_95_pct": self.calculate_cvar(0.95),

            # Drawdown details
            "drawdown_info": dd_info,

            # Number of periods
            "num_periods": len(self.equity_df),
        }

        # Add trade-based metrics if trades provided
        if trades_df is not None and not trades_df.empty:
            metrics["profit_factor"] = self.calculate_profit_factor(trades_df)

        return metrics

    def print_metrics(self, trades_df: pd.DataFrame = None):
        """Print formatted metrics report."""
        metrics = self.get_all_metrics(trades_df)

        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)

        print("\n--- Return Metrics ---")
        print(f"Total Return:        {metrics['total_return_pct']:>10.2f}%")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        print(f"Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")

        print("\n--- Risk Metrics ---")
        print(f"Max Drawdown:        {metrics['max_drawdown_pct']:>10.2f}%")
        print(f"Volatility:          {metrics['volatility_pct']:>10.2f}%")
        print(f"VaR (95%):           {metrics['var_95_pct']:>10.2f}%")
        print(f"CVaR (95%):          {metrics['cvar_95_pct']:>10.2f}%")

        print("\n--- Drawdown Details ---")
        dd = metrics['drawdown_info']
        print(f"Peak Date:           {dd['peak_date']}")
        print(f"Trough Date:         {dd['trough_date']}")
        print(f"Recovery Date:       {dd['recovery_date']}")
        print(f"Peak Value:          ${dd['peak_value']:,.2f}")
        print(f"Trough Value:        ${dd['trough_value']:,.2f}")

        if 'profit_factor' in metrics:
            print("\n--- Trade Metrics ---")
            pf = metrics['profit_factor']
            print(f"Profit Factor:       {pf if pf != float('inf') else 'Inf':>10}")

        print("\n" + "="*60)

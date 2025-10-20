#!/usr/bin/env python3
"""
Portfolio simulator for backtesting.
Tracks positions, executes trades, calculates P&L.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: str
    action: str  # BUY, SELL, HOLD
    price: float
    quantity: float
    cost: float
    commission: float = 0.0
    reason: str = ""


@dataclass
class Position:
    """Current position in an asset."""
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_price(self, price: float):
        """Update current price and unrealized P&L."""
        self.current_price = price
        if self.quantity > 0:
            self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity

    def get_value(self) -> float:
        """Get current market value of position."""
        return self.quantity * self.current_price


class Portfolio:
    """
    Portfolio manager for backtesting.
    Handles cash, positions, trade execution, and P&L tracking.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1%
        max_position_size: float = 0.95,  # Max 95% of capital in positions
        symbol: str = "BTC-USD"
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.max_position_size = max_position_size
        self.symbol = symbol

        self.position = Position(symbol=symbol)
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission_paid = 0.0

    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        return self.cash + self.position.get_value()

    def get_available_capital(self) -> float:
        """Get capital available for new trades."""
        total_value = self.get_total_value()
        max_trade_value = total_value * self.max_position_size
        current_position_value = self.position.get_value()
        return max(0, max_trade_value - current_position_value)

    def execute_trade(
        self,
        action: str,
        price: float,
        timestamp: str,
        risk_percent: Optional[float] = None,
        reason: str = ""
    ) -> Optional[Trade]:
        """
        Execute a trade based on agent recommendation.

        Args:
            action: BUY, SELL, or HOLD
            price: Current market price
            timestamp: Trade timestamp
            risk_percent: Optional risk percentage for position sizing
            reason: Reason for the trade

        Returns:
            Trade object if executed, None if HOLD or failed
        """
        action = action.upper()

        # Update current price
        self.position.update_price(price)

        if action == "HOLD":
            return None

        elif action == "BUY":
            return self._execute_buy(price, timestamp, risk_percent, reason)

        elif action == "SELL":
            return self._execute_sell(price, timestamp, reason)

        else:
            logger.warning(f"Unknown action: {action}")
            return None

    def _execute_buy(
        self,
        price: float,
        timestamp: str,
        risk_percent: Optional[float],
        reason: str
    ) -> Optional[Trade]:
        """Execute a buy order."""
        # Calculate position size
        available = self.get_available_capital()

        if available < price:
            logger.warning(f"Insufficient capital for BUY at {price}. Available: {available}")
            return None

        # Use risk_percent if provided, otherwise use max available
        if risk_percent is not None:
            # Risk percent should be small (e.g., 1-2%)
            trade_value = self.get_total_value() * (risk_percent / 100.0)
            trade_value = min(trade_value, available)
        else:
            # Use a reasonable default (e.g., 20% of available)
            trade_value = available * 0.2

        quantity = trade_value / price
        commission = trade_value * self.commission_rate
        total_cost = trade_value + commission

        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for BUY. Need: {total_cost}, Have: {self.cash}")
            return None

        # Execute trade
        old_quantity = self.position.quantity
        old_avg_price = self.position.avg_entry_price

        # Update position
        total_quantity = old_quantity + quantity
        total_cost_basis = (old_quantity * old_avg_price) + trade_value
        new_avg_price = total_cost_basis / total_quantity if total_quantity > 0 else 0

        self.position.quantity = total_quantity
        self.position.avg_entry_price = new_avg_price
        self.position.update_price(price)

        # Update cash
        self.cash -= total_cost
        self.total_commission_paid += commission

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            action="BUY",
            price=price,
            quantity=quantity,
            cost=trade_value,
            commission=commission,
            reason=reason
        )
        self.trades.append(trade)
        self.total_trades += 1

        logger.info(f"BUY: {quantity:.6f} @ {price:.2f}, Cost: {total_cost:.2f}, Cash: {self.cash:.2f}")
        return trade

    def _execute_sell(
        self,
        price: float,
        timestamp: str,
        reason: str
    ) -> Optional[Trade]:
        """Execute a sell order (full position)."""
        if self.position.quantity <= 0:
            logger.warning("No position to SELL")
            return None

        quantity = self.position.quantity
        revenue = quantity * price
        commission = revenue * self.commission_rate
        net_revenue = revenue - commission

        # Calculate realized P&L
        cost_basis = self.position.quantity * self.position.avg_entry_price
        realized_pnl = revenue - cost_basis - commission

        # Update cash
        self.cash += net_revenue
        self.total_commission_paid += commission

        # Track win/loss
        if realized_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            action="SELL",
            price=price,
            quantity=quantity,
            cost=revenue,
            commission=commission,
            reason=reason
        )
        self.trades.append(trade)
        self.total_trades += 1

        # Update position
        self.position.realized_pnl += realized_pnl
        self.position.quantity = 0
        self.position.avg_entry_price = 0
        self.position.unrealized_pnl = 0
        self.position.update_price(price)

        logger.info(f"SELL: {quantity:.6f} @ {price:.2f}, Revenue: {net_revenue:.2f}, P&L: {realized_pnl:.2f}")
        return trade

    def record_equity(self, timestamp: str):
        """Record current portfolio equity for equity curve."""
        total_value = self.get_total_value()
        returns = (total_value - self.initial_capital) / self.initial_capital

        equity_point = {
            "timestamp": timestamp,
            "total_value": total_value,
            "cash": self.cash,
            "position_value": self.position.get_value(),
            "position_quantity": self.position.quantity,
            "unrealized_pnl": self.position.unrealized_pnl,
            "realized_pnl": self.position.realized_pnl,
            "returns": returns
        }
        self.equity_curve.append(equity_point)

    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        return pd.DataFrame(self.equity_curve)

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        trades_data = [
            {
                "timestamp": t.timestamp,
                "action": t.action,
                "price": t.price,
                "quantity": t.quantity,
                "cost": t.cost,
                "commission": t.commission,
                "reason": t.reason
            }
            for t in self.trades
        ]
        return pd.DataFrame(trades_data)

    def get_summary_stats(self) -> Dict:
        """Get summary statistics for the portfolio."""
        total_value = self.get_total_value()
        total_return = (total_value - self.initial_capital) / self.initial_capital
        total_pnl = total_value - self.initial_capital

        win_rate = (
            self.winning_trades / (self.winning_trades + self.losing_trades)
            if (self.winning_trades + self.losing_trades) > 0
            else 0
        )

        stats = {
            "initial_capital": self.initial_capital,
            "final_value": total_value,
            "total_pnl": total_pnl,
            "total_return_pct": total_return * 100,
            "cash": self.cash,
            "position_value": self.position.get_value(),
            "realized_pnl": self.position.realized_pnl,
            "unrealized_pnl": self.position.unrealized_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": win_rate * 100,
            "total_commission_paid": self.total_commission_paid,
        }

        return stats

    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.position = Position(symbol=self.symbol)
        self.trades = []
        self.equity_curve = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission_paid = 0.0

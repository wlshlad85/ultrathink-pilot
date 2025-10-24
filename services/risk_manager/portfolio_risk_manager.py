"""
Risk Manager Service - Portfolio Risk Management
Implements portfolio-level risk constraints and real-time validation
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class RiskCheckCode(Enum):
    """Risk check rejection codes"""
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"
    SECTOR_EXPOSURE_LIMIT = "SECTOR_EXPOSURE_LIMIT"
    LEVERAGE_LIMIT = "LEVERAGE_LIMIT"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    VAR_LIMIT = "VAR_LIMIT"
    POSITION_SIZE_LIMIT = "POSITION_SIZE_LIMIT"


@dataclass
class Position:
    """Individual position state"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    sector: str = "unknown"
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return abs(self.quantity) * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        return (self.current_price - self.avg_cost) * self.quantity

    @property
    def return_pct(self) -> float:
        """Return percentage"""
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


@dataclass
class RejectionReason:
    """Structured rejection reason"""
    code: RiskCheckCode
    message: str
    limit: float
    proposed: float


@dataclass
class RiskCheckResult:
    """Risk check validation result"""
    approved: bool
    rejection_reasons: List[RejectionReason] = field(default_factory=list)
    allowed_quantity: Optional[int] = None
    risk_assessment: Optional[Dict] = None
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PortfolioRiskManager:
    """
    Portfolio-level risk constraint enforcement

    Responsibilities:
    - Position limit enforcement (25% max per asset)
    - Real-time VaR calculation (95% confidence, 1-day horizon)
    - Correlation tracking
    - Hierarchical risk parity
    """

    def __init__(
        self,
        total_capital: float = 1_000_000.0,
        max_position_pct: float = 0.25,  # 25% max per asset
        max_sector_pct: float = 0.50,    # 50% max per sector
        max_leverage: float = 1.5,       # 1.5x leverage
        max_daily_loss_pct: float = 0.02,  # 2% daily loss limit
        var_confidence: float = 0.95,    # 95% VaR confidence
        var_horizon_days: int = 1,       # 1-day VaR
    ):
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_leverage = max_leverage
        self.max_daily_loss_pct = max_daily_loss_pct
        self.var_confidence = var_confidence
        self.var_horizon_days = var_horizon_days

        # Portfolio state (in-memory)
        self.positions: Dict[str, Position] = {}
        self.cash: float = total_capital
        self.daily_pnl: float = 0.0
        self.daily_start_value: float = total_capital

        # Historical returns for VaR calculation (30 days)
        self.returns_history: Dict[str, List[float]] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        self.correlation_symbols: List[str] = []

        # Performance metrics
        self.last_updated = datetime.utcnow()
        logger.info(
            f"Risk Manager initialized: capital=${total_capital:,.2f}, "
            f"max_position={max_position_pct*100}%, max_sector={max_sector_pct*100}%"
        )

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value (cash + positions)"""
        return self.cash + sum(pos.market_value for pos in self.positions.values())

    @property
    def total_exposure(self) -> float:
        """Total position exposure (sum of absolute values)"""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def leverage_ratio(self) -> float:
        """Current leverage ratio"""
        if self.portfolio_value == 0:
            return 0.0
        return self.total_exposure / self.portfolio_value

    def get_position_pct(self, symbol: str) -> float:
        """Get position as percentage of portfolio"""
        if symbol not in self.positions:
            return 0.0
        return self.positions[symbol].market_value / self.portfolio_value

    def get_sector_exposure(self, sector: str) -> float:
        """Get total sector exposure as percentage"""
        sector_value = sum(
            pos.market_value
            for pos in self.positions.values()
            if pos.sector == sector
        )
        return sector_value / self.portfolio_value

    def update_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        sector: str = "unknown"
    ) -> None:
        """Update or create position"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            # Update average cost (VWAP)
            total_cost = pos.avg_cost * pos.quantity + price * quantity
            total_quantity = pos.quantity + quantity

            if total_quantity != 0:
                pos.avg_cost = total_cost / total_quantity
                pos.quantity = total_quantity
                pos.current_price = price
                pos.last_updated = datetime.utcnow()
            else:
                # Position closed
                del self.positions[symbol]
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
                sector=sector
            )

        # Update cash
        self.cash -= quantity * price
        self.last_updated = datetime.utcnow()

        logger.debug(
            f"Position updated: {symbol} qty={quantity} @ ${price:.2f}, "
            f"portfolio_value=${self.portfolio_value:,.2f}"
        )

    def update_price(self, symbol: str, price: float) -> None:
        """Update current price for position"""
        if symbol in self.positions:
            self.positions[symbol].current_price = price
            self.positions[symbol].last_updated = datetime.utcnow()

    def calculate_var(self, confidence: float = None) -> float:
        """
        Calculate portfolio Value at Risk (VaR)

        Uses historical simulation method with 30 days of returns
        Returns: Maximum expected loss at given confidence level
        """
        if confidence is None:
            confidence = self.var_confidence

        if not self.positions or not self.returns_history:
            return 0.0

        # Get portfolio weights
        symbols = list(self.positions.keys())
        weights = np.array([
            self.positions[sym].market_value / self.portfolio_value
            for sym in symbols
        ])

        # Get historical returns (ensure all symbols have data)
        returns_matrix = []
        valid_symbols = []

        for sym in symbols:
            if sym in self.returns_history and len(self.returns_history[sym]) >= 20:
                returns_matrix.append(self.returns_history[sym][-30:])  # Last 30 days
                valid_symbols.append(sym)

        if len(returns_matrix) < 1:
            logger.warning("Insufficient historical data for VaR calculation")
            return 0.0

        # Ensure equal length (pad if needed)
        max_len = max(len(r) for r in returns_matrix)
        returns_matrix = [
            r + [0.0] * (max_len - len(r))
            for r in returns_matrix
        ]

        returns_array = np.array(returns_matrix).T  # Shape: (days, assets)

        # Update weights for valid symbols only
        valid_weights = np.array([
            self.positions[sym].market_value / self.portfolio_value
            for sym in valid_symbols
        ])
        valid_weights = valid_weights / valid_weights.sum()  # Normalize

        # Calculate portfolio returns
        portfolio_returns = returns_array @ valid_weights

        # VaR is the (1-confidence) percentile of losses
        var_percentile = (1 - confidence) * 100
        var = np.percentile(portfolio_returns, var_percentile)

        # Convert to dollar value (negative = loss)
        var_dollar = abs(var * self.portfolio_value)

        logger.debug(f"VaR calculated: ${var_dollar:,.2f} at {confidence*100}% confidence")
        return var_dollar

    def update_correlation_matrix(self) -> None:
        """Update correlation matrix from historical returns"""
        if not self.returns_history or len(self.positions) < 2:
            return

        symbols = [
            sym for sym in self.positions.keys()
            if sym in self.returns_history and len(self.returns_history[sym]) >= 20
        ]

        if len(symbols) < 2:
            return

        # Build returns matrix
        returns_matrix = []
        for sym in symbols:
            returns_matrix.append(self.returns_history[sym][-30:])

        # Ensure equal length
        max_len = max(len(r) for r in returns_matrix)
        returns_matrix = [
            r + [0.0] * (max_len - len(r))
            for r in returns_matrix
        ]

        returns_array = np.array(returns_matrix)

        # Calculate correlation matrix
        self.correlation_matrix = np.corrcoef(returns_array)
        self.correlation_symbols = symbols

        logger.debug(f"Correlation matrix updated for {len(symbols)} symbols")

    def add_return_observation(self, symbol: str, return_pct: float) -> None:
        """Add return observation for VaR calculation"""
        if symbol not in self.returns_history:
            self.returns_history[symbol] = []

        self.returns_history[symbol].append(return_pct)

        # Keep only last 30 days
        if len(self.returns_history[symbol]) > 30:
            self.returns_history[symbol] = self.returns_history[symbol][-30:]

    async def check_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        estimated_price: float,
        sector: str = "unknown"
    ) -> RiskCheckResult:
        """
        Validate proposed trade against risk constraints

        Returns: RiskCheckResult with approval status and details
        Target latency: <10ms P95
        """
        start_time = datetime.utcnow()
        rejection_reasons = []
        warnings = []

        # Simulate position after trade
        simulated_positions = self.positions.copy()
        simulated_cash = self.cash

        trade_quantity = quantity if action == "BUY" else -quantity
        trade_value = trade_quantity * estimated_price

        if symbol in simulated_positions:
            new_quantity = simulated_positions[symbol].quantity + trade_quantity
            if new_quantity != 0:
                # Update existing position
                old_pos = simulated_positions[symbol]
                total_cost = old_pos.avg_cost * old_pos.quantity + trade_value
                avg_cost = total_cost / new_quantity if new_quantity != 0 else estimated_price

                simulated_positions[symbol] = Position(
                    symbol=symbol,
                    quantity=new_quantity,
                    avg_cost=avg_cost,
                    current_price=estimated_price,
                    sector=sector
                )
            else:
                # Position closed
                del simulated_positions[symbol]
        else:
            # New position
            simulated_positions[symbol] = Position(
                symbol=symbol,
                quantity=trade_quantity,
                avg_cost=estimated_price,
                current_price=estimated_price,
                sector=sector
            )

        simulated_cash -= trade_value

        # Calculate simulated portfolio metrics
        simulated_portfolio_value = simulated_cash + sum(
            pos.market_value for pos in simulated_positions.values()
        )
        simulated_exposure = sum(
            pos.market_value for pos in simulated_positions.values()
        )

        # Check 1: Position concentration limit (25%)
        if symbol in simulated_positions:
            position_pct = simulated_positions[symbol].market_value / simulated_portfolio_value
            if position_pct > self.max_position_pct:
                rejection_reasons.append(RejectionReason(
                    code=RiskCheckCode.CONCENTRATION_LIMIT,
                    message=f"Trade would exceed {self.max_position_pct*100}% single-position limit (proposed: {position_pct*100:.1f}%)",
                    limit=self.max_position_pct,
                    proposed=position_pct
                ))

        # Check 2: Sector exposure limit (50%)
        sector_exposure = sum(
            pos.market_value
            for pos in simulated_positions.values()
            if pos.sector == sector
        ) / simulated_portfolio_value

        if sector_exposure > self.max_sector_pct:
            rejection_reasons.append(RejectionReason(
                code=RiskCheckCode.SECTOR_EXPOSURE_LIMIT,
                message=f"Trade would exceed {self.max_sector_pct*100}% sector exposure limit (proposed: {sector_exposure*100:.1f}%)",
                limit=self.max_sector_pct,
                proposed=sector_exposure
            ))

        # Check 3: Leverage limit (1.5x)
        simulated_leverage = simulated_exposure / simulated_portfolio_value if simulated_portfolio_value > 0 else 0
        if simulated_leverage > self.max_leverage:
            rejection_reasons.append(RejectionReason(
                code=RiskCheckCode.LEVERAGE_LIMIT,
                message=f"Trade would exceed {self.max_leverage}x leverage limit (proposed: {simulated_leverage:.2f}x)",
                limit=self.max_leverage,
                proposed=simulated_leverage
            ))

        # Check 4: Daily loss limit (2%)
        daily_loss_pct = -self.daily_pnl / self.daily_start_value if self.daily_start_value > 0 else 0
        if daily_loss_pct > self.max_daily_loss_pct:
            rejection_reasons.append(RejectionReason(
                code=RiskCheckCode.DAILY_LOSS_LIMIT,
                message=f"Daily loss limit reached: {daily_loss_pct*100:.2f}% (limit: {self.max_daily_loss_pct*100}%)",
                limit=self.max_daily_loss_pct,
                proposed=daily_loss_pct
            ))

        # Check 5: Insufficient cash
        if simulated_cash < 0:
            warnings.append(f"Insufficient cash: ${simulated_cash:,.2f} (need ${-simulated_cash:,.2f} more)")

        # Calculate allowed quantity if rejected
        allowed_quantity = None
        if rejection_reasons and symbol in simulated_positions:
            # Binary search for maximum allowed quantity
            max_allowed_pct = self.max_position_pct * 0.95  # 95% of limit for safety
            current_value = self.positions.get(symbol, Position(symbol, 0, 0, 0)).market_value
            max_value = max_allowed_pct * self.portfolio_value
            additional_value = max_value - current_value
            allowed_quantity = int(additional_value / estimated_price) if additional_value > 0 else 0

        # Build risk assessment
        risk_assessment = {
            "position_after_trade": {
                "quantity": simulated_positions.get(symbol, Position(symbol, 0, 0, 0)).quantity,
                "market_value": simulated_positions.get(symbol, Position(symbol, 0, 0, 0)).market_value,
                "pct_portfolio": simulated_positions.get(symbol, Position(symbol, 0, 0, 0)).market_value / simulated_portfolio_value if simulated_portfolio_value > 0 else 0
            },
            "portfolio_impact": {
                "concentration_increase": abs(
                    simulated_positions.get(symbol, Position(symbol, 0, 0, 0)).market_value / simulated_portfolio_value -
                    self.get_position_pct(symbol)
                ) if simulated_portfolio_value > 0 else 0,
                "leverage_change": simulated_leverage - self.leverage_ratio,
            },
            "limit_utilization": {
                "position_size": (simulated_positions.get(symbol, Position(symbol, 0, 0, 0)).market_value / simulated_portfolio_value) / self.max_position_pct if simulated_portfolio_value > 0 else 0,
                "sector_exposure": sector_exposure / self.max_sector_pct,
                "leverage": simulated_leverage / self.max_leverage
            }
        }

        # Execution time tracking
        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.debug(f"Risk check completed in {elapsed:.2f}ms")

        return RiskCheckResult(
            approved=len(rejection_reasons) == 0,
            rejection_reasons=rejection_reasons,
            allowed_quantity=allowed_quantity,
            risk_assessment=risk_assessment,
            warnings=warnings
        )

    def get_portfolio_state(self) -> Dict:
        """Get current portfolio state"""
        positions_dict = {
            symbol: {
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "pct_portfolio": pos.market_value / self.portfolio_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "sector": pos.sector
            }
            for symbol, pos in self.positions.items()
        }

        return {
            "portfolio": {
                "total_value": self.portfolio_value,
                "cash": self.cash,
                "positions": positions_dict
            },
            "risk_metrics": {
                "var_95_1d": self.calculate_var(),
                "portfolio_beta": 1.0,  # Placeholder
                "leverage_ratio": self.leverage_ratio,
                "daily_pnl": self.daily_pnl,
                "daily_pnl_pct": self.daily_pnl / self.daily_start_value if self.daily_start_value > 0 else 0
            },
            "limit_utilization": {
                "max_position_size_pct": max((pos.market_value / self.portfolio_value for pos in self.positions.values()), default=0),
                "leverage_ratio": self.leverage_ratio,
                "daily_loss_pct": -self.daily_pnl / self.daily_start_value if self.daily_pnl < 0 and self.daily_start_value > 0 else 0
            },
            "last_updated": self.last_updated.isoformat()
        }

    def reset_daily_metrics(self) -> None:
        """Reset daily P&L tracking"""
        self.daily_start_value = self.portfolio_value
        self.daily_pnl = 0.0
        logger.info(f"Daily metrics reset: starting value=${self.daily_start_value:,.2f}")

    def calculate_daily_pnl(self) -> float:
        """Calculate current daily P&L"""
        self.daily_pnl = self.portfolio_value - self.daily_start_value
        return self.daily_pnl

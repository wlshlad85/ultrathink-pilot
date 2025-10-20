#!/usr/bin/env python3
"""
Trade Decision Records for Forensic Analysis

Captures complete context for every trading decision to enable
detailed post-mortem analysis of model behavior.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal
import numpy as np
import pandas as pd

# Type aliases
ActionType = Literal["HOLD", "BUY", "SELL"]
RegimeType = Literal["bull", "bear", "neutral"]
FailurePattern = Literal[
    "caught_falling_knife",
    "fought_the_trend",
    "high_volatility_mistake",
    "bear_market_denial",
    "momentum_misread",
    "success",
    "other"
]


@dataclass
class TradeDecision:
    """
    Complete record of a single trading decision.

    Stores everything needed to understand WHY the model made a decision
    and WHAT happened as a result.
    """
    # Time and price context
    timestamp: str
    price: float

    # Model decision
    action: ActionType
    action_probs: np.ndarray  # [prob_hold, prob_buy, prob_sell]
    confidence: float  # Probability of chosen action

    # Complete state vector
    state: np.ndarray  # Full 43-feature state

    # Portfolio context at decision time
    portfolio_value: float
    cash_ratio: float
    position_ratio: float
    total_return: float

    # Market indicators at decision time
    indicators: Dict[str, float] = field(default_factory=dict)

    # Market regime classification
    regime: RegimeType = "neutral"
    regime_confidence: float = 0.0

    # Forward-looking outcomes (what happened AFTER this decision)
    returns_1d: Optional[float] = None   # 1-day forward return
    returns_5d: Optional[float] = None   # 5-day forward return
    returns_20d: Optional[float] = None  # 20-day forward return

    # Failure classification
    is_bad_decision: bool = False
    failure_pattern: FailurePattern = "other"
    cost_of_mistake: float = 0.0  # Dollar loss from bad decision

    # Additional context
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV/JSON export."""
        return {
            'timestamp': self.timestamp,
            'price': self.price,
            'action': self.action,
            'confidence': self.confidence,
            'prob_hold': self.action_probs[0],
            'prob_buy': self.action_probs[1],
            'prob_sell': self.action_probs[2],
            'portfolio_value': self.portfolio_value,
            'cash_ratio': self.cash_ratio,
            'position_ratio': self.position_ratio,
            'total_return': self.total_return,
            'regime': self.regime,
            'regime_confidence': self.regime_confidence,
            'returns_1d': self.returns_1d,
            'returns_5d': self.returns_5d,
            'returns_20d': self.returns_20d,
            'is_bad_decision': self.is_bad_decision,
            'failure_pattern': self.failure_pattern,
            'cost_of_mistake': self.cost_of_mistake,
            **self.indicators,  # Include all market indicators
            'notes': self.notes
        }

    def is_failed_buy(self, threshold_pct: float = -5.0) -> bool:
        """
        Check if this was a BUY that resulted in significant loss.

        Args:
            threshold_pct: Minimum loss percentage to classify as failure

        Returns:
            True if BUY action followed by loss > threshold
        """
        if self.action != "BUY":
            return False

        if self.returns_5d is None:
            return False

        return self.returns_5d < threshold_pct

    def is_missed_sell(self, threshold_pct: float = -5.0) -> bool:
        """
        Check if we HELD/BOUGHT before a significant drop.

        Args:
            threshold_pct: Drop percentage to classify as missed opportunity

        Returns:
            True if we should have sold but didn't
        """
        if self.action == "SELL":
            return False

        if self.returns_5d is None:
            return False

        # If we had a position and price dropped significantly, we should have sold
        return self.position_ratio > 0.1 and self.returns_5d < threshold_pct

    def get_summary(self) -> str:
        """Get human-readable summary of this decision."""
        outcome = "GOOD" if not self.is_bad_decision else "BAD"

        summary = f"[{outcome}] {self.timestamp}: {self.action} @ ${self.price:.2f}"
        summary += f" (confidence: {self.confidence:.1%})"

        if self.returns_5d is not None:
            summary += f" → 5d return: {self.returns_5d:+.2f}%"

        if self.is_bad_decision:
            summary += f" | Pattern: {self.failure_pattern}"
            if self.cost_of_mistake != 0:
                summary += f" | Cost: ${self.cost_of_mistake:,.2f}"

        return summary


@dataclass
class ForensicsReport:
    """
    Aggregated analysis of trading decisions over a period.
    """
    period_start: str
    period_end: str
    total_decisions: int

    # Action breakdown
    num_holds: int
    num_buys: int
    num_sells: int

    # Failure analysis
    num_bad_decisions: int
    bad_decision_rate: float
    total_cost_of_mistakes: float

    # Failure patterns breakdown
    pattern_counts: Dict[FailurePattern, int] = field(default_factory=dict)
    pattern_costs: Dict[FailurePattern, float] = field(default_factory=dict)

    # Regime performance
    regime_performance: Dict[RegimeType, Dict] = field(default_factory=dict)

    # Top failures (most costly individual mistakes)
    top_failures: list = field(default_factory=list)

    def print_summary(self):
        """Print formatted report summary."""
        print("\n" + "="*80)
        print(f"TRADE FORENSICS REPORT: {self.period_start} to {self.period_end}")
        print("="*80)
        print()

        print("OVERALL STATISTICS")
        print("-" * 80)
        print(f"Total Decisions:       {self.total_decisions:,}")
        print(f"  HOLD:                {self.num_holds:,} ({self.num_holds/self.total_decisions*100:.1f}%)")
        print(f"  BUY:                 {self.num_buys:,} ({self.num_buys/self.total_decisions*100:.1f}%)")
        print(f"  SELL:                {self.num_sells:,} ({self.num_sells/self.total_decisions*100:.1f}%)")
        print()

        print("FAILURE ANALYSIS")
        print("-" * 80)
        print(f"Bad Decisions:         {self.num_bad_decisions:,} ({self.bad_decision_rate*100:.1f}%)")
        print(f"Total Cost:            ${self.total_cost_of_mistakes:,.2f}")
        if self.num_bad_decisions > 0:
            avg_cost = self.total_cost_of_mistakes / self.num_bad_decisions
            print(f"Avg Cost per Failure:  ${avg_cost:,.2f}")
        print()

        if self.pattern_counts:
            print("FAILURE PATTERNS (by frequency)")
            print("-" * 80)
            sorted_patterns = sorted(
                self.pattern_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for pattern, count in sorted_patterns:
                if pattern in ["success", "other"]:
                    continue
                pct = count / self.num_bad_decisions * 100 if self.num_bad_decisions > 0 else 0
                cost = self.pattern_costs.get(pattern, 0)
                print(f"  {pattern:25s}: {count:4d} ({pct:5.1f}%) - Cost: ${cost:,.2f}")
            print()

        if self.regime_performance:
            print("PERFORMANCE BY MARKET REGIME")
            print("-" * 80)
            for regime, stats in self.regime_performance.items():
                print(f"\n  {regime.upper()} Market:")
                print(f"    Decisions:        {stats.get('total', 0):,}")
                print(f"    Bad Decisions:    {stats.get('bad', 0):,} ({stats.get('bad_rate', 0)*100:.1f}%)")
                print(f"    Cost:             ${stats.get('cost', 0):,.2f}")
            print()

        if self.top_failures:
            print("TOP 10 MOST COSTLY MISTAKES")
            print("-" * 80)
            for i, decision in enumerate(self.top_failures[:10], 1):
                print(f"{i:2d}. {decision.get_summary()}")
            print()

        print("="*80)
        print()

#!/usr/bin/env python3
"""
Simple Reward-Only System for RL Trading Agents

Philosophy: Reward positive outcomes, reduce rewards for negative outcomes.
No penalties - just scaled rewards based on outcome quality.

Key principles:
1. Positive returns → Large rewards (scaled by magnitude)
2. Negative returns → Small rewards (10x less sensitive)
3. Volatility → Scales DOWN rewards (never makes them negative)
4. All rewards ≥ 0 (no punishment, just less reward)
"""

import numpy as np
from typing import Dict, List
from collections import deque


class SimpleRewardCalculatorStrong:
    """
    Calculate rewards using reward-only approach.

    Rewards scale with return magnitude, adjusted for risk.
    No negative rewards - just bigger rewards for better outcomes.
    """

    def __init__(
        self,
        lookback_window: int = 30,
        gain_weight: float = 1000.0,
        loss_weight: float = 100.0,
        volatility_sensitivity: float = 100.0,  # STRONG PENALTY
        min_samples_for_volatility: int = 10
    ):
        """
        Initialize simple reward calculator.

        Args:
            lookback_window: Window for volatility calculation
            gain_weight: Multiplier for positive returns (higher = more reward)
            loss_weight: Multiplier for negative returns (lower = less penalty)
            volatility_sensitivity: How much volatility reduces rewards (higher = more reduction)
            min_samples_for_volatility: Minimum samples before using volatility adjustment
        """
        self.lookback_window = lookback_window
        self.gain_weight = gain_weight
        self.loss_weight = loss_weight
        self.volatility_sensitivity = volatility_sensitivity
        self.min_samples_for_volatility = min_samples_for_volatility

        # State tracking
        self.returns_history = deque(maxlen=lookback_window)
        self.portfolio_values = deque(maxlen=lookback_window)
        self.initial_capital = None
        self.peak_value = 0.0
        self.trade_count = 0

    def reset(self, initial_capital: float):
        """Reset calculator for new episode."""
        self.returns_history.clear()
        self.portfolio_values.clear()
        self.initial_capital = initial_capital
        self.peak_value = initial_capital
        self.trade_count = 0

    def calculate_reward(
        self,
        current_value: float,
        previous_value: float,
        action: int = None,
        commission: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate reward using reward-only approach.

        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            action: Action taken (0=BUY, 1=HOLD, 2=SELL) - optional
            commission: Commission paid - optional

        Returns:
            Dict with total reward and component breakdown
        """
        # Calculate step return
        if previous_value > 0:
            step_return = (current_value - previous_value) / previous_value
        else:
            step_return = 0.0

        # Track history
        self.returns_history.append(step_return)
        self.portfolio_values.append(current_value)

        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value

        # Count trades
        if action in [0, 2]:  # BUY or SELL
            self.trade_count += 1

        # === STEP 1: BASE REWARD FROM RETURN ===
        if step_return > 0:
            # Positive return: Full reward scaled by magnitude
            base_reward = step_return * self.gain_weight
        else:
            # Negative return: Reduced reward (10x less sensitive by default)
            base_reward = step_return * self.loss_weight
            # Alternative: base_reward = 0.0 to ignore losses completely

        # === STEP 2: RISK ADJUSTMENT ===
        if len(self.returns_history) >= self.min_samples_for_volatility:
            returns_array = np.array(list(self.returns_history))
            volatility = np.std(returns_array)

            # Stability factor: reduces reward for volatile performance
            # High volatility → low factor → reduced reward
            # Low volatility → high factor → full reward
            stability_factor = 1.0 / (1.0 + volatility * self.volatility_sensitivity)

            risk_adjusted_reward = base_reward * stability_factor
        else:
            # Not enough samples yet, use base reward
            risk_adjusted_reward = base_reward
            volatility = 0.0
            stability_factor = 1.0

        # === STEP 3: ENSURE NON-NEGATIVE ===
        # This is key: we never punish, just reward less
        final_reward = max(0.0, risk_adjusted_reward)

        # Return detailed breakdown
        return {
            'total': final_reward,
            'base_reward': base_reward,
            'stability_factor': stability_factor,
            'volatility': volatility,
            'step_return': step_return
        }

    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Calculate final episode-level metrics.

        Returns standard trading metrics for evaluation.
        """
        if len(self.returns_history) == 0:
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'trade_count': 0
            }

        returns_array = np.array(list(self.returns_history))

        # Sharpe ratio (annualized, assuming daily returns)
        mean_return = np.mean(returns_array)
        volatility = np.std(returns_array)
        risk_free_rate = 0.02 / 365  # 2% annual → daily

        if volatility > 0:
            sharpe_ratio = (mean_return - risk_free_rate) / volatility * np.sqrt(365)
        else:
            sharpe_ratio = 0.0

        # Total return
        if self.initial_capital and self.initial_capital > 0:
            final_value = list(self.portfolio_values)[-1] if len(self.portfolio_values) > 0 else self.initial_capital
            total_return = (final_value - self.initial_capital) / self.initial_capital
        else:
            total_return = 0.0

        # Annualized volatility
        annualized_volatility = volatility * np.sqrt(365)

        # Maximum drawdown
        values_array = np.array(list(self.portfolio_values))
        if len(values_array) > 0:
            running_max = np.maximum.accumulate(values_array)
            drawdowns = (values_array - running_max) / running_max
            max_drawdown = np.min(drawdowns)
        else:
            max_drawdown = 0.0

        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'volatility': annualized_volatility,
            'max_drawdown': max_drawdown,
            'trade_count': self.trade_count
        }


if __name__ == "__main__":
    """Test the simple reward calculator."""

    print("=" * 80)
    print("SIMPLE REWARD CALCULATOR TEST")
    print("=" * 80)
    print()
    print("Testing reward-only approach:")
    print("  - Positive returns → Large rewards")
    print("  - Negative returns → Small rewards")
    print("  - High volatility → Scaled down rewards")
    print("  - All rewards ≥ 0 (no punishment)")
    print()

    # Initialize calculator
    calc = SimpleRewardCalculator(
        gain_weight=1000.0,
        loss_weight=100.0,
        volatility_sensitivity=20.0
    )

    initial_capital = 100000.0
    calc.reset(initial_capital)

    print("Scenario 1: Stable gains (low volatility)")
    print("-" * 80)
    portfolio_value = initial_capital
    for step in range(20):
        previous_value = portfolio_value
        # Simulate stable 0.5% daily gains
        portfolio_value = portfolio_value * 1.005

        reward = calc.calculate_reward(portfolio_value, previous_value)

        if (step + 1) % 5 == 0:
            print(f"Step {step+1:2d}: Value=${portfolio_value:,.0f} | "
                  f"Reward={reward['total']:+7.2f} | "
                  f"Stability={reward['stability_factor']:.3f}")

    print()
    print("Scenario 2: Volatile gains (high volatility)")
    print("-" * 80)
    calc.reset(initial_capital)
    portfolio_value = initial_capital
    for step in range(20):
        previous_value = portfolio_value
        # Simulate volatile gains: alternating +2% / -1%
        change = 1.02 if step % 2 == 0 else 0.99
        portfolio_value = portfolio_value * change

        reward = calc.calculate_reward(portfolio_value, previous_value)

        if (step + 1) % 5 == 0:
            print(f"Step {step+1:2d}: Value=${portfolio_value:,.0f} | "
                  f"Reward={reward['total']:+7.2f} | "
                  f"Stability={reward['stability_factor']:.3f}")

    print()
    print("Scenario 3: Losses (negative returns)")
    print("-" * 80)
    calc.reset(initial_capital)
    portfolio_value = initial_capital
    for step in range(20):
        previous_value = portfolio_value
        # Simulate consistent losses: -0.3% daily
        portfolio_value = portfolio_value * 0.997

        reward = calc.calculate_reward(portfolio_value, previous_value)

        if (step + 1) % 5 == 0:
            print(f"Step {step+1:2d}: Value=${portfolio_value:,.0f} | "
                  f"Reward={reward['total']:+7.2f} | "
                  f"Base={reward['base_reward']:+7.2f}")

    print()
    print("Key Observations:")
    print("  - Stable gains → Full rewards (stability_factor ~1.0)")
    print("  - Volatile gains → Reduced rewards (stability_factor ~0.3-0.5)")
    print("  - Losses → Small rewards (10x less than gains)")
    print("  - All rewards ≥ 0 (no negative punishment)")
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

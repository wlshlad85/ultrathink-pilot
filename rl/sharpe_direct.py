#!/usr/bin/env python3
"""
Direct Sharpe Reward System for RL Trading Agents

Philosophy: Use Sharpe ratio directly as reward signal, but with non-negative floor.
This is the most principled approach - directly optimize what we want to evaluate on.

Key principles:
1. Calculate rolling Sharpe ratio from recent returns
2. Scale appropriately for RL (multiply by constant)
3. Apply max(0, sharpe) floor to prevent adversarial gradients
4. No other penalties or adjustments
"""

import numpy as np
from typing import Dict
from collections import deque


class SharpeDirectRewardCalculator:
    """
    Calculate rewards using direct Sharpe ratio with non-negative floor.

    This is the most theoretically sound approach - we optimize directly
    for the metric we care about (Sharpe ratio) while avoiding the
    adversarial gradient problem through the non-negative floor.
    """

    def __init__(
        self,
        lookback_window: int = 30,
        risk_free_rate: float = 0.02,  # 2% annual
        sharpe_scale: float = 100.0,   # Scale factor for RL
        min_samples: int = 10
    ):
        """
        Initialize Sharpe-direct reward calculator.

        Args:
            lookback_window: Window for Sharpe calculation
            risk_free_rate: Annual risk-free rate
            sharpe_scale: Scaling factor to make rewards appropriate for RL
            min_samples: Minimum samples before calculating Sharpe
        """
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free = risk_free_rate / 365  # Daily rate
        self.sharpe_scale = sharpe_scale
        self.min_samples = min_samples

        # State tracking
        self.returns_history = deque(maxlen=lookback_window)
        self.portfolio_values = deque(maxlen=lookback_window)
        self.initial_capital = None
        self.trade_count = 0

    def reset(self, initial_capital: float):
        """Reset calculator for new episode."""
        self.returns_history.clear()
        self.portfolio_values.clear()
        self.initial_capital = initial_capital
        self.trade_count = 0

    def calculate_reward(
        self,
        current_value: float,
        previous_value: float,
        action: int = None,
        commission: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate reward using direct Sharpe ratio.

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

        # Count trades
        if action in [0, 2]:  # BUY or SELL
            self.trade_count += 1

        # === CALCULATE ROLLING SHARPE RATIO ===
        if len(self.returns_history) >= self.min_samples:
            returns_array = np.array(list(self.returns_history))

            mean_return = np.mean(returns_array)
            volatility = np.std(returns_array)

            # Calculate Sharpe ratio
            if volatility > 0:
                sharpe_ratio = (mean_return - self.daily_risk_free) / volatility
            else:
                # Zero volatility means constant returns
                # If positive, give high Sharpe; if zero/negative, give zero
                sharpe_ratio = 100.0 if mean_return > self.daily_risk_free else 0.0

            # Scale Sharpe ratio for RL reward magnitude
            scaled_sharpe = sharpe_ratio * self.sharpe_scale

            # Apply non-negative floor (KEY INNOVATION)
            final_reward = max(0.0, scaled_sharpe)
        else:
            # Not enough samples yet - use simple return-based reward
            # Positive returns → small reward, negative → zero
            final_reward = max(0.0, step_return * 1000.0)
            sharpe_ratio = 0.0
            mean_return = step_return
            volatility = 0.0

        # Return detailed breakdown
        return {
            'total': final_reward,
            'sharpe_ratio': sharpe_ratio,
            'scaled_sharpe': scaled_sharpe if len(self.returns_history) >= self.min_samples else 0.0,
            'mean_return': mean_return,
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

        if volatility > 0:
            sharpe_ratio = (mean_return - self.daily_risk_free) / volatility * np.sqrt(365)
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
    """Test the Sharpe-direct reward calculator."""

    print("=" * 80)
    print("SHARPE-DIRECT REWARD CALCULATOR TEST")
    print("=" * 80)
    print()
    print("Testing direct Sharpe ratio approach with non-negative floor:")
    print("  - Rolling Sharpe ratio calculated from recent returns")
    print("  - Scaled by constant factor for RL magnitude")
    print("  - Floor at 0 prevents negative rewards")
    print()

    # Initialize calculator
    calc = SharpeDirectRewardCalculator(
        lookback_window=30,
        sharpe_scale=100.0
    )

    initial_capital = 100000.0
    calc.reset(initial_capital)

    print("Scenario 1: Stable positive returns (ideal for Sharpe)")
    print("-" * 80)
    portfolio_value = initial_capital
    for step in range(30):
        previous_value = portfolio_value
        # Simulate stable 0.5% daily gains
        portfolio_value = portfolio_value * 1.005

        reward = calc.calculate_reward(portfolio_value, previous_value)

        if (step + 1) % 10 == 0:
            print(f"Step {step+1:2d}: Value=${portfolio_value:,.0f} | "
                  f"Reward={reward['total']:+7.2f} | "
                  f"Sharpe={reward['sharpe_ratio']:+.3f}")

    print()
    print("Scenario 2: Volatile returns (bad for Sharpe)")
    print("-" * 80)
    calc.reset(initial_capital)
    portfolio_value = initial_capital
    for step in range(30):
        previous_value = portfolio_value
        # Simulate volatile gains: alternating +3% / -2%
        change = 1.03 if step % 2 == 0 else 0.98
        portfolio_value = portfolio_value * change

        reward = calc.calculate_reward(portfolio_value, previous_value)

        if (step + 1) % 10 == 0:
            print(f"Step {step+1:2d}: Value=${portfolio_value:,.0f} | "
                  f"Reward={reward['total']:+7.2f} | "
                  f"Sharpe={reward['sharpe_ratio']:+.3f}")

    print()
    print("Scenario 3: Negative returns (floored at zero)")
    print("-" * 80)
    calc.reset(initial_capital)
    portfolio_value = initial_capital
    for step in range(30):
        previous_value = portfolio_value
        # Simulate losses: -0.5% daily
        portfolio_value = portfolio_value * 0.995

        reward = calc.calculate_reward(portfolio_value, previous_value)

        if (step + 1) % 10 == 0:
            print(f"Step {step+1:2d}: Value=${portfolio_value:,.0f} | "
                  f"Reward={reward['total']:+7.2f} | "
                  f"Sharpe={reward['sharpe_ratio']:+.3f}")

    print()
    print("Key Observations:")
    print("  - Stable gains → High Sharpe → High rewards")
    print("  - Volatile gains → Low Sharpe → Low rewards")
    print("  - Losses → Negative Sharpe → Floored at 0")
    print("  - Agent learns to maximize Sharpe ratio directly!")
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

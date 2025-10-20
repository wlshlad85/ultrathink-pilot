#!/usr/bin/env python3
"""
Hybrid Reward System for Progressive Training

Combines simple P&L rewards (early training) with Sharpe-optimized rewards (later training).
This approach allows the agent to first learn basic profitability, then refine to risk-adjusted returns.

Key innovation: Smooth transition prevents jarring reward signal changes during training.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict
from collections import deque

# Add parent directory to path for standalone testing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.simple_reward import SimpleRewardCalculator
from rl.sharpe_reward import SharpeRewardCalculator


class HybridRewardCalculator:
    """
    Hybrid reward calculator with progressive weighting.

    Training progression:
    - Episodes 1-30: 100% Simple P&L → Learn profitability
    - Episodes 31-70: Gradual transition → Balance profit & risk
    - Episodes 71+: 100% Sharpe-optimized → Maximize risk-adjusted returns
    """

    def __init__(
        self,
        transition_start: int = 30,
        transition_end: int = 70,
        lookback_window: int = 50,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize hybrid reward calculator.

        Args:
            transition_start: Episode when transition begins
            transition_end: Episode when transition completes
            lookback_window: Window for rolling calculations
            risk_free_rate: Annual risk-free rate
        """
        self.transition_start = transition_start
        self.transition_end = transition_end

        # Initialize both calculators
        self.simple_calc = SimpleRewardCalculator(
            lookback_window=lookback_window,
            gain_weight=1000.0,
            loss_weight=100.0,
            volatility_sensitivity=20.0
        )

        self.sharpe_calc = SharpeRewardCalculator(
            lookback_window=lookback_window,
            risk_free_rate=risk_free_rate,
            sharpe_weight=1.0,
            drawdown_penalty_weight=0.5,
            trading_cost_weight=0.1,
            exploration_bonus_weight=0.0  # Disabled
        )

        # Track current episode
        self.current_episode = 0
        self.initial_capital = None

    def reset(self, initial_capital: float, episode: int = None):
        """
        Reset for new episode.

        Args:
            initial_capital: Starting capital
            episode: Current episode number (optional, auto-increments if not provided)
        """
        if episode is not None:
            self.current_episode = episode
        else:
            self.current_episode += 1

        self.initial_capital = initial_capital
        self.simple_calc.reset(initial_capital)
        self.sharpe_calc.reset(initial_capital)

    def calculate_reward(
        self,
        current_value: float,
        previous_value: float,
        action: int,
        commission: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate hybrid reward with progressive weighting.

        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            action: Action taken (0=BUY, 1=HOLD, 2=SELL)
            commission: Commission paid

        Returns:
            Dict with total reward and component breakdown
        """
        # Calculate both reward types
        simple_breakdown = self.simple_calc.calculate_reward(
            current_value, previous_value, action, commission
        )

        sharpe_breakdown = self.sharpe_calc.calculate_reward(
            current_value, previous_value, action, commission
        )

        # Calculate weighting based on current episode
        sharpe_weight = self._get_sharpe_weight()
        simple_weight = 1.0 - sharpe_weight

        # Blend rewards
        hybrid_reward = (
            simple_weight * simple_breakdown['total'] +
            sharpe_weight * sharpe_breakdown['total']
        )

        return {
            'total': hybrid_reward,
            'simple_reward': simple_breakdown['total'],
            'sharpe_reward': sharpe_breakdown['total'],
            'simple_weight': simple_weight,
            'sharpe_weight': sharpe_weight,
            'episode': self.current_episode,
            'step_return': simple_breakdown['step_return']
        }

    def _get_sharpe_weight(self) -> float:
        """
        Calculate Sharpe weight based on current episode.

        Returns linear interpolation between transition_start and transition_end.
        """
        if self.current_episode < self.transition_start:
            # Pure simple rewards
            return 0.0
        elif self.current_episode > self.transition_end:
            # Pure Sharpe rewards
            return 1.0
        else:
            # Linear transition
            progress = (self.current_episode - self.transition_start) / (
                self.transition_end - self.transition_start
            )
            return progress

    def get_episode_metrics(self) -> Dict[str, float]:
        """Get episode-level metrics from Sharpe calculator."""
        return self.sharpe_calc.get_episode_metrics()


if __name__ == "__main__":
    """Test hybrid reward calculator."""

    print("=" * 80)
    print("HYBRID REWARD CALCULATOR TEST")
    print("=" * 80)
    print()

    # Initialize calculator
    calc = HybridRewardCalculator(
        transition_start=30,
        transition_end=70
    )

    initial_capital = 100000.0

    # Simulate episodes at different stages
    test_episodes = [1, 30, 50, 70, 100]

    for ep in test_episodes:
        calc.reset(initial_capital, episode=ep)

        print(f"Episode {ep}: Sharpe weight = {calc._get_sharpe_weight():.2f}")
        print("-" * 80)

        # Simulate 10 steps
        portfolio_value = initial_capital
        for step in range(10):
            previous_value = portfolio_value
            # Simulate 0.5% gain
            portfolio_value *= 1.005
            action = 1  # HOLD

            reward_breakdown = calc.calculate_reward(
                current_value=portfolio_value,
                previous_value=previous_value,
                action=action
            )

        print(f"  Final reward breakdown:")
        print(f"    Simple reward:  {reward_breakdown['simple_reward']:+7.2f} (weight: {reward_breakdown['simple_weight']:.2f})")
        print(f"    Sharpe reward:  {reward_breakdown['sharpe_reward']:+7.2f} (weight: {reward_breakdown['sharpe_weight']:.2f})")
        print(f"    Hybrid reward:  {reward_breakdown['total']:+7.2f}")
        print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

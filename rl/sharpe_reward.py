#!/usr/bin/env python3
"""
Sharpe-Optimized Reward Function for RL Trading Agents

Replaces simple P&L reward with risk-adjusted performance metrics.
Designed to train agents that optimize for Sharpe ratio, not just raw returns.

Key innovations:
1. Rolling Sharpe ratio calculation (primary reward)
2. Drawdown penalty (risk management)
3. Trading frequency penalty (avoid overtrading)
4. Exploration bonus (encourage diverse strategies)
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque


class SharpeRewardCalculator:
    """
    Calculate Sharpe-optimized rewards for RL trading agents.

    The reward combines multiple components:
    - Sharpe ratio (primary): Risk-adjusted returns
    - Drawdown penalty: Penalize large losses
    - Trading cost: Discourage excessive trading
    - Exploration bonus: Reward strategy diversity
    """

    def __init__(
        self,
        lookback_window: int = 50,
        risk_free_rate: float = 0.02,  # 2% annual
        sharpe_weight: float = 1.0,
        drawdown_penalty_weight: float = 0.5,
        trading_cost_weight: float = 0.1,
        exploration_bonus_weight: float = 0.05,
        min_trades_for_sharpe: int = 10
    ):
        """
        Initialize Sharpe reward calculator.

        Args:
            lookback_window: Number of steps for rolling calculations
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            sharpe_weight: Weight for Sharpe ratio component
            drawdown_penalty_weight: Weight for drawdown penalty
            trading_cost_weight: Weight for trading cost penalty
            exploration_bonus_weight: Weight for exploration bonus
            min_trades_for_sharpe: Minimum trades before calculating Sharpe
        """
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free = risk_free_rate / 365  # Convert to daily

        # Reward component weights
        self.sharpe_weight = sharpe_weight
        self.drawdown_penalty_weight = drawdown_penalty_weight
        self.trading_cost_weight = trading_cost_weight
        self.exploration_bonus_weight = exploration_bonus_weight
        self.min_trades_for_sharpe = min_trades_for_sharpe

        # State tracking
        self.returns_history = deque(maxlen=lookback_window)
        self.portfolio_values = deque(maxlen=lookback_window)
        self.actions_history = deque(maxlen=lookback_window)
        self.initial_capital = None
        self.peak_value = 0.0
        self.trade_count = 0

    def reset(self, initial_capital: float):
        """Reset calculator for new episode."""
        self.returns_history.clear()
        self.portfolio_values.clear()
        self.actions_history.clear()
        self.initial_capital = initial_capital
        self.peak_value = initial_capital
        self.trade_count = 0

    def calculate_reward(
        self,
        current_value: float,
        previous_value: float,
        action: int,
        commission: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate comprehensive Sharpe-optimized reward.

        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            action: Action taken (0=BUY, 1=HOLD, 2=SELL)
            commission: Commission paid this step

        Returns:
            Dict with total reward and component breakdown
        """
        # Calculate return for this step
        if previous_value > 0:
            step_return = (current_value - previous_value) / previous_value
        else:
            step_return = 0.0

        # Track history
        self.returns_history.append(step_return)
        self.portfolio_values.append(current_value)
        self.actions_history.append(action)

        # Update peak for drawdown calculation
        if current_value > self.peak_value:
            self.peak_value = current_value

        # Count trades
        if action in [0, 2]:  # BUY or SELL
            self.trade_count += 1

        # === COMPONENT 1: SHARPE RATIO ===
        sharpe_reward = self._calculate_sharpe_component()

        # === COMPONENT 2: DRAWDOWN PENALTY ===
        drawdown_penalty = self._calculate_drawdown_penalty(current_value)

        # === COMPONENT 3: TRADING COST PENALTY ===
        trading_cost_penalty = self._calculate_trading_cost_penalty(commission, current_value)

        # === COMPONENT 4: EXPLORATION BONUS ===
        # FIXED: Removed exploration bonus during training - creates noisy signal
        # exploration_bonus = self._calculate_exploration_bonus()

        # === COMBINE COMPONENTS ===
        total_reward = (
            self.sharpe_weight * sharpe_reward +
            self.drawdown_penalty_weight * drawdown_penalty +
            self.trading_cost_weight * trading_cost_penalty
            # Removed exploration_bonus_weight component
        )

        # Return detailed breakdown
        return {
            'total': total_reward,
            'sharpe': sharpe_reward,
            'drawdown_penalty': drawdown_penalty,
            'trading_cost': trading_cost_penalty,
            'exploration': 0.0,  # Disabled during training
            'step_return': step_return
        }

    def _calculate_sharpe_component(self) -> float:
        """
        Calculate Sharpe ratio-based reward component.

        Returns risk-adjusted performance over recent window.
        """
        if len(self.returns_history) < self.min_trades_for_sharpe:
            # Not enough data yet, return step return as proxy
            if len(self.returns_history) > 0:
                return self.returns_history[-1]
            return 0.0

        returns_array = np.array(list(self.returns_history))

        # Calculate mean excess return
        mean_return = np.mean(returns_array)
        excess_return = mean_return - self.daily_risk_free

        # Calculate volatility
        volatility = np.std(returns_array)

        # Sharpe ratio (daily)
        if volatility > 0:
            sharpe = excess_return / volatility
        else:
            # No volatility = perfect (if positive returns) or terrible (if negative)
            sharpe = 10.0 if mean_return > 0 else -10.0

        # Scale Sharpe to reasonable reward range
        # Daily Sharpe of 0.1 = annualized ~1.9 (good)
        # Daily Sharpe of 0.2 = annualized ~3.8 (excellent)
        # We want rewards in [-1, +1] range mostly
        # FIXED: Increased scaling factor from 5.0 to 10.0 for stronger signal
        scaled_sharpe = np.tanh(sharpe * 10.0)  # Sigmoid-like scaling

        return scaled_sharpe

    def _calculate_drawdown_penalty(self, current_value: float) -> float:
        """
        Penalize large drawdowns from peak equity.

        Encourages risk management and capital preservation.
        """
        if self.peak_value <= 0:
            return 0.0

        # Calculate current drawdown
        drawdown = (self.peak_value - current_value) / self.peak_value

        if drawdown <= 0:
            # At new peak, no penalty
            return 0.0

        # Exponential penalty for larger drawdowns
        # FIXED: Reduced penalty coefficient from 50.0 to 10.0 (5x less harsh)
        # 5% drawdown: -0.025 penalty (was -0.125)
        # 10% drawdown: -0.10 penalty (was -0.5)
        # 20% drawdown: -0.40 penalty (was -2.0)
        penalty = -(drawdown ** 2) * 10.0

        # Clip to avoid extreme penalties
        penalty = max(penalty, -2.0)  # Also reduced clip threshold

        return penalty

    def _calculate_trading_cost_penalty(self, commission: float, current_value: float) -> float:
        """
        Penalize excessive trading via commission costs.

        Encourages selective trading, not high-frequency noise.
        """
        if current_value <= 0 or commission <= 0:
            return 0.0

        # Commission as fraction of portfolio
        cost_fraction = commission / current_value

        # Linear penalty proportional to cost
        # FIXED: Reduced coefficient from 100.0 to 10.0 (10x less harsh)
        penalty = -cost_fraction * 10.0  # Scale to reasonable range

        return penalty

    def _calculate_exploration_bonus(self) -> float:
        """
        Reward diverse action selection to encourage exploration.

        Prevents premature convergence to single strategy.
        """
        if len(self.actions_history) < 20:
            # Not enough history
            return 0.0

        actions_array = np.array(list(self.actions_history))

        # Calculate action entropy (Shannon entropy)
        action_counts = np.bincount(actions_array, minlength=3)
        action_probs = action_counts / len(actions_array)

        # Remove zero probabilities
        action_probs = action_probs[action_probs > 0]

        # Entropy: H = -sum(p * log(p))
        entropy = -np.sum(action_probs * np.log(action_probs))

        # Max entropy for 3 actions = log(3) = 1.099
        max_entropy = np.log(3)

        # Normalize to [0, 1]
        normalized_entropy = entropy / max_entropy

        # Bonus for high entropy (diverse actions)
        # But not too high - we want some convergence eventually
        # Optimal entropy ~0.8 (some preference, but not dogmatic)
        bonus = -((normalized_entropy - 0.8) ** 2)  # Parabola peaked at 0.8

        return bonus

    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Calculate final episode-level metrics.

        Useful for logging and evaluation.
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

        # Sharpe ratio (annualized)
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


class RegimeAwareSharpeReward(SharpeRewardCalculator):
    """
    Regime-aware extension of Sharpe reward calculator.

    Adjusts reward targets based on market regime:
    - Bull market: Target high Sharpe (>1.5), reward aggressive gains
    - Bear market: Target positive Sharpe (>0.5), reward capital preservation
    - Sideways: Target moderate Sharpe (>1.0), reward consistency
    """

    def __init__(
        self,
        regime: str = "BULL",  # "BULL", "BEAR", or "SIDEWAYS"
        **kwargs
    ):
        """
        Initialize regime-aware Sharpe reward calculator.

        Args:
            regime: Current market regime
            **kwargs: Arguments for base SharpeRewardCalculator
        """
        super().__init__(**kwargs)

        self.regime = regime

        # Regime-specific targets
        self.regime_targets = {
            'BULL': {
                'target_sharpe': 1.5,
                'sharpe_weight': 1.0,
                'drawdown_penalty': 0.3,  # Less strict in bull markets
                'exploration_bonus': 0.05
            },
            'BEAR': {
                'target_sharpe': 0.5,
                'sharpe_weight': 0.7,  # Less emphasis on returns
                'drawdown_penalty': 1.0,  # Very strict in bear markets
                'exploration_bonus': 0.02  # Less exploration, more survival
            },
            'SIDEWAYS': {
                'target_sharpe': 1.0,
                'sharpe_weight': 1.0,
                'drawdown_penalty': 0.5,
                'exploration_bonus': 0.1  # More exploration for range-trading
            }
        }

        # Apply regime-specific weights
        self._apply_regime_settings()

    def _apply_regime_settings(self):
        """Apply regime-specific reward weights."""
        if self.regime in self.regime_targets:
            settings = self.regime_targets[self.regime]
            self.sharpe_weight = settings['sharpe_weight']
            self.drawdown_penalty_weight = settings['drawdown_penalty']
            self.exploration_bonus_weight = settings['exploration_bonus']

    def set_regime(self, regime: str):
        """Update regime and adjust reward weights accordingly."""
        self.regime = regime
        self._apply_regime_settings()


if __name__ == "__main__":
    """Test Sharpe reward calculator."""

    print("=" * 80)
    print("SHARPE REWARD CALCULATOR TEST")
    print("=" * 80)
    print()

    # Initialize calculator
    calc = SharpeRewardCalculator(
        lookback_window=50,
        risk_free_rate=0.02
    )

    # Simulate episode
    initial_capital = 100000.0
    calc.reset(initial_capital)

    print("Simulating trading episode with positive trend...")
    print()

    portfolio_value = initial_capital
    for step in range(100):
        # Simulate upward trend with noise
        step_return = 0.005 + np.random.normal(0, 0.02)
        previous_value = portfolio_value
        portfolio_value = portfolio_value * (1 + step_return)

        # Simulate occasional trades
        action = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])  # Mostly HOLD
        commission = 10.0 if action in [0, 2] else 0.0

        # Calculate reward
        reward_breakdown = calc.calculate_reward(
            current_value=portfolio_value,
            previous_value=previous_value,
            action=action,
            commission=commission
        )

        if (step + 1) % 20 == 0:
            print(f"Step {step+1:3d}: "
                  f"Value=${portfolio_value:,.0f} | "
                  f"Reward={reward_breakdown['total']:+.3f} | "
                  f"Sharpe={reward_breakdown['sharpe']:+.3f}")

    print()
    print("Episode Summary:")
    print("-" * 80)
    metrics = calc.get_episode_metrics()
    for key, value in metrics.items():
        if 'ratio' in key or 'return' in key or 'drawdown' in key:
            print(f"  {key:20s}: {value:+.3f}")
        elif 'count' in key:
            print(f"  {key:20s}: {int(value)}")
        else:
            print(f"  {key:20s}: {value:.3f}")

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

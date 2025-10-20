#!/usr/bin/env python3
"""
RL Trading Environment for UltraThink.
Implements OpenAI Gym interface for training trading agents.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Any
import gymnasium as gym
from gymnasium import spaces

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtesting.data_fetcher import DataFetcher
from backtesting.portfolio import Portfolio
from rl.sharpe_reward import SharpeRewardCalculator
from rl.simple_reward import SimpleRewardCalculator


class TradingEnv(gym.Env):
    """
    Trading environment for RL agent training.

    State Space:
        - Portfolio state (cash, position, value)
        - Market indicators (RSI, MACD, ATR, etc.)
        - Price history (recent returns)

    Action Space:
        - 0: HOLD
        - 1: BUY (20% of available capital)
        - 2: SELL (full position)

    Reward:
        - Portfolio value change + Sharpe-adjusted risk penalty
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        symbol: str = "BTC-USD",
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        window_size: int = 30,
        reward_scaling: float = 1e-4,
        use_sharpe_penalty: bool = True,
        use_sharpe_reward: bool = False
    ):
        """
        Initialize trading environment.

        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            initial_capital: Starting capital
            commission_rate: Trading commission rate
            window_size: Number of historical bars to include in state
            reward_scaling: Scale rewards for better training
            use_sharpe_penalty: Whether to include risk-adjusted penalty (legacy)
            use_sharpe_reward: Whether to use SharpeRewardCalculator (new)
        """
        super().__init__()

        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.use_sharpe_penalty = use_sharpe_penalty
        self.use_sharpe_reward = use_sharpe_reward

        # Initialize Sharpe reward calculator if enabled
        self.sharpe_calculator = None
        if self.use_sharpe_reward:
            self.sharpe_calculator = SharpeRewardCalculator(
                lookback_window=50,
                risk_free_rate=0.02,
                sharpe_weight=1.0,
                drawdown_penalty_weight=0.5,
                trading_cost_weight=0.1,
                exploration_bonus_weight=0.05
            )

        # Track last commission for Sharpe reward
        self.last_commission = 0.0

        # Load market data
        self.data_fetcher = DataFetcher(symbol)
        self.data_fetcher.fetch(start_date, end_date)
        self.data_fetcher.add_technical_indicators()
        self.market_data = self.data_fetcher.data

        # Skip initial days for indicator warmup (use min to handle short datasets)
        self.start_idx = min(200, max(50, len(self.market_data) // 4))
        self.max_idx = len(self.market_data) - 1

        # Portfolio
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            symbol=symbol
        )

        # Current state
        self.current_idx = self.start_idx
        self.prev_portfolio_value = initial_capital
        self.returns_history = []

        # Define action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Define observation space
        # Features: portfolio state (3) + market indicators (10) + price history (window_size)
        num_features = 3 + 10 + window_size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features,),
            dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset portfolio
        self.portfolio.reset()

        # Reset position
        self.current_idx = self.start_idx
        self.prev_portfolio_value = self.initial_capital
        self.returns_history = []
        self.last_commission = 0.0

        # Reset Sharpe calculator if enabled
        if self.sharpe_calculator is not None:
            self.sharpe_calculator.reset(self.initial_capital)

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current market data
        row = self.market_data.iloc[self.current_idx]
        price = float(row['close'])
        date = str(row.name.date()) if hasattr(row.name, 'date') else str(row.name)

        # Update portfolio with current price
        self.portfolio.position.update_price(price)

        # Execute action
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_str = action_map[action]

        trade = self.portfolio.execute_trade(
            action=action_str,
            price=price,
            timestamp=date,
            risk_percent=1.0,  # Use 20% of capital for BUY
            reason="RL_agent"
        )

        # Track commission for Sharpe reward
        if trade and hasattr(trade, 'commission'):
            self.last_commission = trade.commission
        else:
            # Estimate commission if trade occurred
            self.last_commission = 0.0
            if action_str in ["BUY", "SELL"]:
                trade_value = price * 0.20 * self.portfolio.cash / price  # Rough estimate
                self.last_commission = trade_value * self.commission_rate

        # Record equity
        self.portfolio.record_equity(date)

        # Calculate reward
        current_value = self.portfolio.get_total_value()
        reward = self._calculate_reward(current_value, action)

        # Update history
        self.prev_portfolio_value = current_value
        period_return = (current_value / self.initial_capital) - 1
        self.returns_history.append(period_return)

        # Move to next step
        self.current_idx += 1

        # Check if episode is done
        terminated = (self.current_idx >= self.max_idx)
        truncated = False

        # Get new observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state).

        Returns:
            State vector with portfolio state, indicators, and price history
        """
        row = self.market_data.iloc[self.current_idx]

        # Portfolio state (normalized)
        portfolio_value = self.portfolio.get_total_value()
        cash_ratio = self.portfolio.cash / portfolio_value if portfolio_value > 0 else 1.0
        position_ratio = self.portfolio.position.get_value() / portfolio_value if portfolio_value > 0 else 0.0
        total_return = (portfolio_value / self.initial_capital) - 1

        portfolio_features = np.array([
            cash_ratio,
            position_ratio,
            total_return
        ], dtype=np.float32)

        # Market indicators (normalized)
        price = row['close']
        indicators = np.array([
            row.get('rsi_14', 50) / 100.0,  # Normalize RSI to [0, 1]
            row.get('macd', 0) / price if price > 0 else 0,
            row.get('macd_signal', 0) / price if price > 0 else 0,
            row.get('atr_14', 0) / price if price > 0 else 0,
            (row.get('close', 0) - row.get('bb_lower', row['close'])) /
                (row.get('bb_upper', row['close']) - row.get('bb_lower', row['close']) + 1e-8),
            row.get('volume_ratio', 1.0) / 3.0,  # Normalize volume ratio
            row.get('returns_1d', 0),
            row.get('returns_5d', 0),
            row.get('returns_20d', 0),
            1.0 if row.get('sma_20', 0) > row.get('sma_50', 0) else 0.0,
        ], dtype=np.float32)

        # Recent price history (returns)
        start_idx = max(0, self.current_idx - self.window_size)
        price_history = self.market_data['close'].iloc[start_idx:self.current_idx + 1].pct_change().fillna(0).values

        # Pad if necessary
        if len(price_history) < self.window_size:
            padding = np.zeros(self.window_size - len(price_history))
            price_history = np.concatenate([padding, price_history])
        else:
            price_history = price_history[-self.window_size:]

        price_history = price_history.astype(np.float32)

        # Combine all features
        observation = np.concatenate([portfolio_features, indicators, price_history])

        # Handle any NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        return observation

    def _calculate_reward(self, current_value: float, action: int) -> float:
        """
        Calculate reward for the current step.

        If use_sharpe_reward=True: Use SharpeRewardCalculator (comprehensive)
        Else: Simple reward with optional Sharpe penalty (legacy)

        Args:
            current_value: Current portfolio value
            action: Action taken this step

        Returns:
            Reward value
        """
        # NEW: Use Sharpe reward calculator if enabled
        if self.use_sharpe_reward and self.sharpe_calculator is not None:
            reward_breakdown = self.sharpe_calculator.calculate_reward(
                current_value=current_value,
                previous_value=self.prev_portfolio_value,
                action=action,
                commission=self.last_commission
            )
            raw_reward = reward_breakdown['total']
            # FIXED: Add reward clipping to prevent extreme values
            clipped_reward = np.clip(raw_reward, -10.0, 10.0)
            return clipped_reward

        # LEGACY: Basic reward with optional Sharpe penalty
        value_change = current_value - self.prev_portfolio_value
        reward = value_change * self.reward_scaling

        # Optional: Add Sharpe-based risk penalty (legacy method)
        if self.use_sharpe_penalty and len(self.returns_history) > 10:
            returns = np.array(self.returns_history[-30:])  # Last 30 periods
            volatility = np.std(returns)

            # Penalize high volatility
            if volatility > 0.02:  # If volatility > 2%
                risk_penalty = -0.1 * (volatility - 0.02)
                reward += risk_penalty

        # FIXED: Clip legacy rewards too
        return np.clip(reward, -10.0, 10.0)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        row = self.market_data.iloc[self.current_idx]

        return {
            "date": str(row.name.date()) if hasattr(row.name, 'date') else str(row.name),
            "price": float(row['close']),
            "portfolio_value": self.portfolio.get_total_value(),
            "cash": self.portfolio.cash,
            "position_value": self.portfolio.position.get_value(),
            "position_quantity": self.portfolio.position.quantity,
            "total_trades": self.portfolio.total_trades,
            "total_return": (self.portfolio.get_total_value() / self.initial_capital - 1) * 100
        }

    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            info = self._get_info()
            print(f"Date: {info['date']}, Price: ${info['price']:.2f}, "
                  f"Portfolio: ${info['portfolio_value']:.2f}, "
                  f"Return: {info['total_return']:.2f}%")

    def get_portfolio_stats(self) -> Dict:
        """Get final portfolio statistics."""
        return self.portfolio.get_summary_stats()


if __name__ == "__main__":
    # Test environment
    print("Testing TradingEnv...")

    env = TradingEnv(
        symbol="BTC-USD",
        start_date="2024-01-01",
        end_date="2024-06-01",
        initial_capital=100000.0
    )

    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    # Run a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}: Action={action}, Reward={reward:.4f}, Value=${info['portfolio_value']:.2f}")

        if terminated or truncated:
            break

    print("\n" + "="*60)
    print("Final Portfolio Stats:")
    stats = env.get_portfolio_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

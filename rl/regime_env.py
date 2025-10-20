#!/usr/bin/env python3
"""
Regime-specific trading environment for training specialist agents.

Each environment filters episodes to specific market regimes:
- Bull market specialist: Trains only on bull market periods
- Bear market specialist: Trains only on bear market periods
- Sideways specialist: Trains only on sideways/consolidation periods
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.data_fetcher import DataFetcher
from backtesting.portfolio import Portfolio
from rl.sharpe_direct import SharpeDirectRewardCalculator
from rl.regime_classifier import RegimeClassifier, RegimeType


class RegimeSpecificTradingEnv(gym.Env):
    """Trading environment that only samples from specific market regime."""

    def __init__(
        self,
        regime: RegimeType,
        symbol: str = "BTC-USD",
        start_date: str = "2017-01-01",
        end_date: str = "2021-12-31",
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        window_size: int = 30,
        episode_length: int = 200  # Max steps per episode
    ):
        super().__init__()

        self.regime = regime
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.window_size = window_size
        self.episode_length = episode_length

        # Load and classify data
        self.data_fetcher = DataFetcher(symbol)
        self.data_fetcher.fetch(start_date, end_date)
        self.data_fetcher.add_technical_indicators()

        # Classify regimes
        self.regime_classifier = RegimeClassifier(sma_short=50, sma_long=200)
        classified_data = self.regime_classifier.classify(self.data_fetcher.data)

        # Filter to specific regime
        self.market_data = classified_data[classified_data['regime'] == regime].copy()

        if len(self.market_data) < 250:  # Need enough data
            raise ValueError(
                f"Insufficient {regime} market data: only {len(self.market_data)} points. "
                f"Need at least 250 points for training."
            )

        # Valid starting indices (need window_size before + episode_length after)
        min_start = self.window_size
        max_start = len(self.market_data) - episode_length - 1
        self.valid_start_indices = list(range(min_start, max_start))

        # Portfolio
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            symbol=symbol
        )

        # Reward calculator
        self.reward_calculator = SharpeDirectRewardCalculator(
            lookback_window=30,
            sharpe_scale=100.0
        )

        # State
        self.current_idx = 0
        self.episode_start_idx = 0
        self.steps_in_episode = 0
        self.prev_portfolio_value = initial_capital
        self.last_commission = 0.0

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation space
        num_features = 3 + 10 + window_size  # portfolio + indicators + price history
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features,),
            dtype=np.float32
        )

        print(f"Regime-Specific Environment Created:")
        print(f"  Regime: {regime}")
        print(f"  Total {regime} data points: {len(self.market_data)}")
        print(f"  Valid episode starts: {len(self.valid_start_indices)}")
        print(f"  Episode length: {episode_length} steps")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly select a regime-specific starting point
        self.episode_start_idx = np.random.choice(self.valid_start_indices)
        self.current_idx = self.episode_start_idx
        self.steps_in_episode = 0

        # Reset portfolio and reward calculator
        self.portfolio.reset()
        self.reward_calculator.reset(self.initial_capital)
        self.prev_portfolio_value = self.initial_capital
        self.last_commission = 0.0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        # Get current data
        row = self.market_data.iloc[self.current_idx]
        price = float(row['close'])
        date = str(row.name.date()) if hasattr(row.name, 'date') else str(row.name)

        # Update portfolio price
        self.portfolio.position.update_price(price)

        # Execute action
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_str = action_map[action]

        trade = self.portfolio.execute_trade(
            action=action_str,
            price=price,
            timestamp=date,
            risk_percent=1.0,
            reason="RL_regime_specialist"
        )

        # Track commission
        if trade and hasattr(trade, 'commission'):
            self.last_commission = trade.commission
        else:
            self.last_commission = 0.0

        # Record equity
        self.portfolio.record_equity(date)

        # Calculate reward
        current_value = self.portfolio.get_total_value()
        reward_breakdown = self.reward_calculator.calculate_reward(
            current_value=current_value,
            previous_value=self.prev_portfolio_value,
            action=action,
            commission=self.last_commission
        )
        reward = reward_breakdown['total']

        # Update state
        self.prev_portfolio_value = current_value
        self.current_idx += 1
        self.steps_in_episode += 1

        # Check if done
        terminated = (
            self.steps_in_episode >= self.episode_length or
            self.current_idx >= len(self.market_data) - 1
        )
        truncated = False

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        row = self.market_data.iloc[self.current_idx]

        # Portfolio state
        portfolio_value = self.portfolio.get_total_value()
        cash_ratio = self.portfolio.cash / portfolio_value if portfolio_value > 0 else 1.0
        position_ratio = self.portfolio.position.get_value() / portfolio_value if portfolio_value > 0 else 0.0
        total_return = (portfolio_value / self.initial_capital) - 1

        portfolio_features = np.array([cash_ratio, position_ratio, total_return], dtype=np.float32)

        # Market indicators
        price = row['close']
        indicators = np.array([
            row.get('rsi_14', 50) / 100.0,
            row.get('macd', 0) / price if price > 0 else 0,
            row.get('macd_signal', 0) / price if price > 0 else 0,
            row.get('atr_14', 0) / price if price > 0 else 0,
            (row.get('close', 0) - row.get('bb_lower', row['close'])) /
                (row.get('bb_upper', row['close']) - row.get('bb_lower', row['close']) + 1e-8),
            row.get('volume_ratio', 1.0) / 3.0,
            row.get('returns_1d', 0),
            row.get('returns_5d', 0),
            row.get('returns_20d', 0),
            1.0 if row.get('sma_20', 0) > row.get('sma_50', 0) else 0.0,
        ], dtype=np.float32)

        # Price history (from the filtered regime data)
        start_idx = max(0, self.current_idx - self.window_size)
        price_history = self.market_data['close'].iloc[start_idx:self.current_idx + 1].pct_change().fillna(0).values

        if len(price_history) < self.window_size:
            padding = np.zeros(self.window_size - len(price_history))
            price_history = np.concatenate([padding, price_history])
        else:
            price_history = price_history[-self.window_size:]

        price_history = price_history.astype(np.float32)

        observation = np.concatenate([portfolio_features, indicators, price_history])
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        return observation

    def _get_info(self):
        row = self.market_data.iloc[self.current_idx]
        return {
            "date": str(row.name.date()) if hasattr(row.name, 'date') else str(row.name),
            "price": float(row['close']),
            "portfolio_value": self.portfolio.get_total_value(),
            "total_return": (self.portfolio.get_total_value() / self.initial_capital - 1) * 100,
            "regime": self.regime,
            "steps_in_episode": self.steps_in_episode
        }

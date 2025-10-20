#!/usr/bin/env python3
"""
RL Trading Environment V2 with Regime Awareness

Enhanced version that addresses forensics findings:
1. Explicit regime detection features
2. Trend confirmation signals
3. Regime-aware reward shaping
4. Better sell signal support

Based on forensics analysis showing model struggles with:
- Neutral market transitions (11.5% error rate)
- Fighting the trend in bear markets
- Not selling near market tops
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
from rl.regime_detector import RegimeDetector, RegimeType

class TradingEnvV2(gym.Env):
    """
    Regime-aware trading environment for RL agent training.

    NEW State Space (53 features):
        - Portfolio state (3): cash_ratio, position_ratio, total_return
        - Market indicators (10): RSI, MACD, ATR, BB, volume, returns, SMA cross
        - Regime features (5): regime_type, confidence, duration, transition_risk, stability
        - Trend features (5): trend_50d, trend_200d, sma_alignment, price_momentum, volatility_percentile
        - Price history (30): recent returns

    Action Space:
        - 0: HOLD
        - 1: BUY (20% of available capital)
        - 2: SELL (full position)

    NEW Reward:
        - Base: Portfolio value change
        - Regime penalties: Penalize buying in bear/neutral markets
        - Trend bonuses: Reward trend-following behavior
        - Sell bonuses: Reward taking profits near peaks
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
        use_regime_rewards: bool = True,
        regime_penalty_weight: float = 0.5,
        trend_bonus_weight: float = 0.3,
        sell_bonus_weight: float = 0.2
    ):
        """
        Initialize regime-aware trading environment.

        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            initial_capital: Starting capital
            commission_rate: Trading commission rate
            window_size: Number of historical bars to include in state
            reward_scaling: Scale rewards for better training
            use_regime_rewards: Enable regime-aware reward shaping
            regime_penalty_weight: Weight for regime-based penalties
            trend_bonus_weight: Weight for trend-following bonuses
            sell_bonus_weight: Weight for profit-taking bonuses
        """
        super().__init__()

        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.use_regime_rewards = use_regime_rewards
        self.regime_penalty_weight = regime_penalty_weight
        self.trend_bonus_weight = trend_bonus_weight
        self.sell_bonus_weight = sell_bonus_weight

        # Load market data
        self.data_fetcher = DataFetcher(symbol)
        self.data_fetcher.fetch(start_date, end_date)
        self.data_fetcher.add_technical_indicators()
        self.market_data = self.data_fetcher.data

        # Initialize regime detector
        self.regime_detector = RegimeDetector(
            bull_threshold=0.10,
            bear_threshold=-0.10,
            lookback_days=60
        )

        # Pre-compute regimes for all timestamps
        self._precompute_regimes()

        # Skip initial days for indicator warmup
        self.start_idx = min(200, max(50, len(self.market_data) // 4))
        self.max_idx = len(self.market_data) - 1

        # Portfolio
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            symbol=symbol
        )

        # Tracking
        self.current_idx = self.start_idx
        self.prev_portfolio_value = initial_capital
        self.returns_history = []
        self.portfolio_peak = initial_capital
        self.current_regime = "neutral"
        self.regime_start_idx = self.start_idx

        # Define action space
        self.action_space = spaces.Discrete(3)

        # Define observation space
        # Portfolio (3) + Indicators (10) + Regime (5) + Trend (5) + History (window_size)
        num_features = 3 + 10 + 5 + 5 + window_size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features,),
            dtype=np.float32
        )

    def _precompute_regimes(self):
        """
        Pre-compute regime classifications for all timestamps.
        This avoids redundant computation during training.
        """
        self.regime_cache = {}

        for idx in range(len(self.market_data)):
            if idx < self.regime_detector.lookback_days:
                self.regime_cache[idx] = "neutral"
            else:
                regime = self.regime_detector.detect_regime(
                    self.market_data,
                    idx
                )
                self.regime_cache[idx] = regime

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset portfolio
        self.portfolio.reset()

        # Reset tracking
        self.current_idx = self.start_idx
        self.prev_portfolio_value = self.initial_capital
        self.returns_history = []
        self.portfolio_peak = self.initial_capital
        self.current_regime = self.regime_cache.get(self.start_idx, "neutral")
        self.regime_start_idx = self.start_idx

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
            risk_percent=1.0,  # 20% position sizing
            reason="RL_agent_v2"
        )

        # Record equity
        self.portfolio.record_equity(date)

        # Calculate reward
        current_value = self.portfolio.get_total_value()
        reward = self._calculate_reward(current_value, action, price)

        # Update tracking
        self.prev_portfolio_value = current_value
        period_return = (current_value / self.initial_capital) - 1
        self.returns_history.append(period_return)

        # Track portfolio peak for sell bonus calculation
        if current_value > self.portfolio_peak:
            self.portfolio_peak = current_value

        # Track regime changes
        new_regime = self.regime_cache.get(self.current_idx, "neutral")
        if new_regime != self.current_regime:
            self.regime_start_idx = self.current_idx
            self.current_regime = new_regime

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
        Get current observation with regime-aware features.

        Returns:
            State vector (53 features):
                - Portfolio state (3)
                - Market indicators (10)
                - Regime features (5)
                - Trend features (5)
                - Price history (30)
        """
        row = self.market_data.iloc[self.current_idx]
        price = row['close']

        # 1. Portfolio state (3 features)
        portfolio_value = self.portfolio.get_total_value()
        cash_ratio = self.portfolio.cash / portfolio_value if portfolio_value > 0 else 1.0
        position_ratio = self.portfolio.position.get_value() / portfolio_value if portfolio_value > 0 else 0.0
        total_return = (portfolio_value / self.initial_capital) - 1

        portfolio_features = np.array([
            cash_ratio,
            position_ratio,
            total_return
        ], dtype=np.float32)

        # 2. Market indicators (10 features) - same as before
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

        # 3. NEW: Regime features (5 features)
        regime = self.regime_cache.get(self.current_idx, "neutral")
        regime_encoded = {"bull": 1.0, "neutral": 0.5, "bear": 0.0}.get(regime, 0.5)

        # Calculate regime confidence based on how long we've been in this regime
        regime_duration = self.current_idx - self.regime_start_idx
        regime_confidence = min(regime_duration / 20, 1.0)  # Max confidence after 20 days

        # Regime transition risk: high if recent volatility increased or regime unstable
        lookback_regimes = [
            self.regime_cache.get(i, "neutral")
            for i in range(max(0, self.current_idx - 10), self.current_idx + 1)
        ]
        unique_regimes = len(set(lookback_regimes))
        regime_transition_risk = (unique_regimes - 1) / 2  # 0 if stable, 1 if changing every period

        # Regime stability: opposite of transition risk
        regime_stability = 1.0 - regime_transition_risk

        regime_features = np.array([
            regime_encoded,           # 0=bear, 0.5=neutral, 1=bull
            regime_confidence,        # 0-1, how confident we are
            regime_duration / 100,    # Normalized days in regime
            regime_transition_risk,   # 0-1, higher = more unstable
            regime_stability          # 0-1, higher = more stable
        ], dtype=np.float32)

        # 4. NEW: Trend confirmation features (5 features)
        sma_20 = row.get('sma_20', price)
        sma_50 = row.get('sma_50', price)

        # Calculate 200-day SMA if we have enough data
        start_200 = max(0, self.current_idx - 200)
        if self.current_idx - start_200 >= 100:  # Need at least 100 days
            sma_200 = self.market_data['close'].iloc[start_200:self.current_idx + 1].mean()
        else:
            sma_200 = sma_50

        # Trend strength: how far is price from moving average
        trend_50d = (price - sma_50) / (sma_50 + 1e-8)  # Normalized distance
        trend_200d = (price - sma_200) / (sma_200 + 1e-8)

        # SMA alignment score: +1 if all aligned bullish, -1 if bearish, 0 if mixed
        sma_alignment = 0.0
        if price > sma_20 > sma_50 > sma_200:
            sma_alignment = 1.0  # Strong bullish alignment
        elif price < sma_20 < sma_50 < sma_200:
            sma_alignment = -1.0  # Strong bearish alignment
        elif price > sma_20 and sma_20 > sma_50:
            sma_alignment = 0.5  # Partial bullish
        elif price < sma_20 and sma_20 < sma_50:
            sma_alignment = -0.5  # Partial bearish

        # Price momentum: recent trend direction
        lookback_20 = max(0, self.current_idx - 20)
        past_price = self.market_data['close'].iloc[lookback_20]
        price_momentum = (price - past_price) / (past_price + 1e-8)

        # Volatility percentile: where is current volatility relative to history
        returns_1d = self.market_data['returns_1d'].iloc[max(0, self.current_idx - 90):self.current_idx + 1]
        current_vol = abs(row.get('returns_1d', 0))
        historical_vols = returns_1d.abs()
        if len(historical_vols) > 0:
            volatility_percentile = (historical_vols < current_vol).sum() / len(historical_vols)
        else:
            volatility_percentile = 0.5

        trend_features = np.array([
            np.clip(trend_50d, -1, 1),           # Price vs SMA50 trend
            np.clip(trend_200d, -1, 1),          # Price vs SMA200 trend
            sma_alignment,                        # SMA stack alignment
            np.clip(price_momentum, -0.5, 0.5),  # Recent momentum
            volatility_percentile                 # Current volatility level
        ], dtype=np.float32)

        # 5. Price history (30 features) - same as before
        start_idx = max(0, self.current_idx - self.window_size)
        price_history = self.market_data['close'].iloc[start_idx:self.current_idx + 1].pct_change().fillna(0).values

        if len(price_history) < self.window_size:
            padding = np.zeros(self.window_size - len(price_history))
            price_history = np.concatenate([padding, price_history])
        else:
            price_history = price_history[-self.window_size:]

        price_history = price_history.astype(np.float32)

        # Combine all features
        observation = np.concatenate([
            portfolio_features,  # 3
            indicators,          # 10
            regime_features,     # 5
            trend_features,      # 5
            price_history        # 30
        ])  # Total: 53 features

        # Handle any NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        return observation

    def _calculate_reward(self, current_value: float, action: int, price: float) -> float:
        """
        Calculate regime-aware reward.

        Reward Components:
        1. Base: Portfolio value change (scaled)
        2. Regime penalties: Penalize buying in unfavorable regimes
        3. Trend bonuses: Reward trend-following behavior
        4. Sell bonuses: Reward profit-taking near peaks

        Args:
            current_value: Current portfolio value
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            price: Current price

        Returns:
            Total reward
        """
        # Base reward: change in portfolio value
        value_change = current_value - self.prev_portfolio_value
        base_reward = value_change * self.reward_scaling

        if not self.use_regime_rewards:
            return base_reward

        # Get current regime and trend info
        regime = self.regime_cache.get(self.current_idx, "neutral")
        row = self.market_data.iloc[self.current_idx]
        sma_50 = row.get('sma_50', price)

        # Initialize penalty/bonus
        regime_adjustment = 0.0
        trend_adjustment = 0.0
        sell_adjustment = 0.0

        # 1. Regime-based penalties/bonuses
        if action == 1:  # BUY
            if regime == "bear":
                # Strong penalty for buying in bear market
                regime_adjustment = -self.regime_penalty_weight
            elif regime == "neutral":
                # Moderate penalty for buying in uncertain market
                regime_adjustment = -self.regime_penalty_weight * 0.4
            elif regime == "bull":
                # Small bonus for buying in bull market
                regime_adjustment = self.regime_penalty_weight * 0.2

        elif action == 0:  # HOLD
            if regime in ["bear", "neutral"]:
                # Reward capital preservation in uncertain times
                regime_adjustment = self.regime_penalty_weight * 0.1

        elif action == 2:  # SELL
            if regime == "bear" and self.portfolio.position.quantity > 0:
                # Reward selling in bear market
                regime_adjustment = self.regime_penalty_weight * 0.3

        # 2. Trend confirmation bonuses
        if action == 1:  # BUY
            if price > sma_50:
                # Bonus for buying with the trend
                trend_adjustment = self.trend_bonus_weight * 0.2
            else:
                # Penalty for buying against the trend
                trend_adjustment = -self.trend_bonus_weight * 0.3

        # 3. Sell bonuses (profit-taking)
        if action == 2 and self.portfolio.position.quantity > 0:  # SELL
            # Calculate unrealized P&L
            position_value = self.portfolio.position.get_value()
            cost_basis = self.portfolio.position.quantity * self.portfolio.position.avg_entry_price

            if position_value > cost_basis:
                # Reward taking profits
                profit_pct = (position_value - cost_basis) / cost_basis
                sell_adjustment = self.sell_bonus_weight * min(profit_pct, 0.5)

            # Extra bonus for selling near peak
            if current_value > self.portfolio_peak * 0.95:  # Within 5% of peak
                sell_adjustment += self.sell_bonus_weight * 0.2

        # Combine all components
        total_reward = base_reward + regime_adjustment + trend_adjustment + sell_adjustment

        return total_reward

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
            "total_return": (self.portfolio.get_total_value() / self.initial_capital - 1) * 100,
            "regime": self.current_regime,
            "regime_duration": self.current_idx - self.regime_start_idx
        }

    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            info = self._get_info()
            print(f"[{info['regime'].upper()}] Date: {info['date']}, Price: ${info['price']:.2f}, "
                  f"Portfolio: ${info['portfolio_value']:.2f}, Return: {info['total_return']:.2f}%")

    def get_portfolio_stats(self) -> Dict:
        """Get final portfolio statistics."""
        return self.portfolio.get_summary_stats()


if __name__ == "__main__":
    # Test regime-aware environment
    print("Testing TradingEnvV2 (Regime-Aware)...")

    env = TradingEnvV2(
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2022-06-30",
        initial_capital=100000.0,
        use_regime_rewards=True
    )

    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape} (expected: 53)")
    print(f"Initial info: {info}")
    print(f"Regime: {info['regime']}")

    # Run a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}: Action={action}, Reward={reward:.4f}, "
              f"Value=${info['portfolio_value']:.2f}, Regime={info['regime']}")

        if terminated or truncated:
            break

    print("\n" + "="*60)
    print("✓ TradingEnvV2 test complete!")
    print("State space: 53 features (was 43)")
    print("  - Portfolio: 3")
    print("  - Indicators: 10")
    print("  - Regime: 5 (NEW)")
    print("  - Trend: 5 (NEW)")
    print("  - History: 30")

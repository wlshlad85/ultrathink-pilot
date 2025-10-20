#!/usr/bin/env python3
"""
Train Universal Agent with Simple Reward-Only System

Key differences from train_sharpe_universal.py:
- Uses SimpleRewardCalculator (reward-only, no penalties)
- Positive returns → Large rewards
- Negative returns → Small rewards (10x less)
- Volatility scales DOWN rewards (never negative)
- Expected: Agent learns to trade actively for positive returns
"""

import sys
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl.ppo_agent import PPOAgent
from rl.simple_reward_exp import SimpleRewardCalculatorExp
from backtesting.data_fetcher import DataFetcher
from backtesting.portfolio import Portfolio
import gymnasium as gym
from gymnasium import spaces

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTradingEnv(gym.Env):
    """Simplified trading environment with reward-only system."""

    def __init__(
        self,
        symbol: str = "BTC-USD",
        start_date: str = "2017-01-01",
        end_date: str = "2021-12-31",
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        window_size: int = 30
    ):
        super().__init__()

        self.symbol = symbol
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.window_size = window_size

        # Load data
        self.data_fetcher = DataFetcher(symbol)
        self.data_fetcher.fetch(start_date, end_date)
        self.data_fetcher.add_technical_indicators()
        self.market_data = self.data_fetcher.data

        self.start_idx = min(200, max(50, len(self.market_data) // 4))
        self.max_idx = len(self.market_data) - 1

        # Portfolio
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            symbol=symbol
        )

        # Reward calculator
        self.reward_calculator = SimpleRewardCalculatorExp(
            lookback_window=30,
            gain_weight=1000.0,
            loss_weight=100.0,
            volatility_sensitivity=20.0
        )

        # State
        self.current_idx = self.start_idx
        self.prev_portfolio_value = initial_capital
        self.last_commission = 0.0

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation space
        num_features = 3 + 10 + window_size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.portfolio.reset()
        self.reward_calculator.reset(self.initial_capital)
        self.current_idx = self.start_idx
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
            reason="RL_agent"
        )

        # Track commission
        if trade and hasattr(trade, 'commission'):
            self.last_commission = trade.commission
        else:
            self.last_commission = 0.0

        # Record equity
        self.portfolio.record_equity(date)

        # Calculate reward using simple calculator
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

        # Check if done
        terminated = (self.current_idx >= self.max_idx)
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

        # Price history
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
            "total_return": (self.portfolio.get_total_value() / self.initial_capital - 1) * 100
        }


def evaluate_on_validation(agent, device, initial_capital):
    """Evaluate agent on 2022 validation set."""
    val_env = SimpleTradingEnv(
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2022-12-31",
        initial_capital=initial_capital
    )

    agent.policy.eval()
    state, info = val_env.reset()
    val_reward = 0.0

    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs, _ = agent.policy(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()

        state, reward, terminated, truncated, info = val_env.step(action)
        val_reward += reward

        if terminated or truncated:
            break

    # Get metrics
    metrics = val_env.reward_calculator.get_episode_metrics()
    metrics['final_value'] = val_env.portfolio.get_total_value()
    agent.policy.train()

    return metrics


def train_simple_reward():
    """Train universal agent with simple reward-only system."""

    print("=" * 80)
    print("EXPERIMENT 2: EXPONENTIAL VOLATILITY PENALTY")
    print("=" * 80)
    print()
    print("Training Configuration:")
    print("  Reward System:     SIMPLE (reward-only, no penalties)")
    print("  Training Data:     2017-2021 (5 years)")
    print("  Validation Data:   2022 (1 year)")
    print()
    print("  Episodes:          200 (with early stopping)")
    print("  Early Stopping:    50 episode patience")
    print("  Validation Freq:   Every 10 episodes")
    print()
    print("Reward Philosophy:")
    print("  ✅ Positive returns → Large rewards (1000x multiplier)")
    print("  ⚠️ Negative returns → Small rewards (100x multiplier)")
    print("  📊 High volatility → Scaled down rewards (stability factor)")
    print("  🛡️ All rewards ≥ 0 (no punishment, just less reward)")
    print("=" * 80)
    print()

    root_dir = Path(__file__).parent
    model_dir = root_dir / "rl" / "models" / "exp2_exp"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    initial_capital = 100000.0
    train_env = SimpleTradingEnv(
        symbol="BTC-USD",
        start_date="2017-01-01",
        end_date="2021-12-31",
        initial_capital=initial_capital
    )

    print(f"Environment created:")
    print(f"  State dimensions:  {train_env.observation_space.shape[0]}")
    print(f"  Action space:      {train_env.action_space.n} (HOLD/BUY/SELL)")
    print(f"  Training period:   2017-2021 (5 years)")
    print(f"  Initial capital:   ${initial_capital:,.2f}")
    print()

    # Initialize PPO agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PPOAgent(
        state_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.n,
        device=device,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4
    )

    print(f"PPO Agent initialized on device: {device}")
    print()

    # Training loop
    num_episodes = 200
    update_freq = 2048
    val_freq = 10
    patience = 50

    episode_rewards = []
    episode_sharpes = []
    episode_returns = []
    validation_sharpes = []
    validation_episodes = []

    best_val_sharpe = -np.inf
    patience_counter = 0

    print(f"Training for up to {num_episodes} episodes...")
    print()

    for episode in range(1, num_episodes + 1):
        state, info = train_env.reset()
        episode_reward = 0.0
        step = 0

        while True:
            # Select action
            action = agent.select_action(state)

            # Take step
            next_state, reward, terminated, truncated, info = train_env.step(action)

            # Store
            agent.store_reward_and_terminal(reward, terminated or truncated)

            episode_reward += reward
            state = next_state
            step += 1

            # Update policy
            if step % update_freq == 0:
                agent.update()

            if terminated or truncated:
                break

        # Record metrics
        final_value = train_env.portfolio.get_total_value()
        episode_return = ((final_value - initial_capital) / initial_capital) * 100

        train_metrics = train_env.reward_calculator.get_episode_metrics()
        episode_sharpe = train_metrics['sharpe_ratio']

        episode_rewards.append(episode_reward)
        episode_returns.append(episode_return)
        episode_sharpes.append(episode_sharpe)

        # Print progress
        if episode % 10 == 0:
            avg_sharpe = np.mean(episode_sharpes[-10:])
            print(f"Episode {episode}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Return: {episode_return:+.2f}% | "
                  f"Sharpe: {episode_sharpe:+.3f} | "
                  f"Avg(10): {avg_sharpe:+.3f}")

            # Validation
            print(f"  [VALIDATION] Testing on 2022 bear market...")
            val_metrics = evaluate_on_validation(agent, device, initial_capital)
            val_sharpe = val_metrics['sharpe_ratio']
            val_return = val_metrics['total_return'] * 100

            validation_sharpes.append(val_sharpe)
            validation_episodes.append(episode)

            print(f"  [VALIDATION] Sharpe: {val_sharpe:+.3f} | "
                  f"Return: {val_return:+.2f}%")

            # Check for improvement
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                patience_counter = 0

                # Save best model
                best_model_path = model_dir / "best_model.pth"
                torch.save(agent.policy.state_dict(), best_model_path)
                print(f"  [NEW BEST] Validation Sharpe: {val_sharpe:+.3f} (saved)")
            else:
                patience_counter += 1
                print(f"  [NO IMPROVEMENT] Patience: {patience_counter}/{patience}")

                # Early stopping
                if patience_counter >= patience:
                    print()
                    print(f"  [EARLY STOPPING] No improvement for {patience} validations")
                    print(f"  [EARLY STOPPING] Best validation Sharpe: {best_val_sharpe:+.3f}")
                    print(f"  [EARLY STOPPING] Stopping at episode {episode}")
                    break

            print()

    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print(f"Episodes Completed: {episode}/{num_episodes}")
    print(f"Best Validation Sharpe: {best_val_sharpe:+.3f}")
    print(f"Final Training Sharpe (avg last 10): {np.mean(episode_sharpes[-10:]):+.3f}")
    print()

    # Save metrics
    training_metrics = {
        "episode_rewards": episode_rewards,
        "episode_returns": episode_returns,
        "episode_sharpes": episode_sharpes,
        "validation_sharpes": validation_sharpes,
        "validation_episodes": validation_episodes,
        "best_val_sharpe": best_val_sharpe,
        "total_episodes": episode,
        "early_stopped": (episode < num_episodes),
        "reward_type": "simple_reward_exp"
    }

    metrics_path = model_dir / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    print()

    # Compare to baseline
    print("=" * 80)
    print("COMPARISON TO FAILED SHARPE-OPTIMIZED VERSION")
    print("=" * 80)
    print()
    print(f"  Sharpe-Optimized:   Validation Sharpe = 0.000 (agent stopped trading)")
    print(f"  Simple Reward-Only: Validation Sharpe = {best_val_sharpe:+.3f}")
    print()

    if best_val_sharpe > 0.0:
        print("  ✅ SUCCESS: Agent achieved positive Sharpe ratio!")
        print("     Simple reward-only system works!")
    elif best_val_sharpe > -1.172:
        print("  ⚠️  PROGRESS: Better than original baseline (-1.172)")
    else:
        print("  ❌ NEEDS WORK: Still below baseline")
    print()

    return training_metrics


if __name__ == "__main__":
    try:
        metrics = train_simple_reward()
        print("=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Train Bear Market Specialist Agent

Trains an agent specifically on bear market conditions.
Uses Sharpe-direct reward system for risk-adjusted performance.
"""

import sys
import torch
import logging
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl.ppo_agent import PPOAgent
from rl.regime_env import RegimeSpecificTradingEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_on_validation(agent, device, initial_capital):
    """Evaluate agent on 2022 validation set (bear periods only)."""
    val_env = RegimeSpecificTradingEnv(
        regime="BEAR",
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2022-12-31",
        initial_capital=initial_capital,
        episode_length=100
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

    metrics = val_env.reward_calculator.get_episode_metrics()
    metrics['final_value'] = val_env.portfolio.get_total_value()
    agent.policy.train()

    return metrics


def train_bear_specialist():
    """Train bear market specialist."""

    print("=" * 80)
    print("BEAR MARKET SPECIALIST TRAINING")
    print("=" * 80)
    print()
    print("Training Configuration:")
    print("  Regime:            BEAR MARKET ONLY")
    print("  Reward System:     Sharpe-direct (risk-adjusted returns)")
    print("  Training Data:     2017-2021 bear periods")
    print("  Validation Data:   2022 bear periods")
    print()
    print("  Episodes:          200 (with early stopping)")
    print("  Early Stopping:    50 episode patience")
    print("  Validation Freq:   Every 10 episodes")
    print("=" * 80)
    print()

    root_dir = Path(__file__).parent
    model_dir = root_dir / "rl" / "models" / "bear_specialist"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    initial_capital = 100000.0
    train_env = RegimeSpecificTradingEnv(
        regime="BEAR",
        symbol="BTC-USD",
        start_date="2017-01-01",
        end_date="2021-12-31",
        initial_capital=initial_capital,
        episode_length=200
    )

    print(f"Environment created:")
    print(f"  State dimensions:  {train_env.observation_space.shape[0]}")
    print(f"  Action space:      {train_env.action_space.n} (HOLD/BUY/SELL)")
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
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            agent.store_reward_and_terminal(reward, terminated or truncated)

            episode_reward += reward
            state = next_state
            step += 1

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
            print(f"  [VALIDATION] Testing on 2022 bear periods...")
            val_metrics = evaluate_on_validation(agent, device, initial_capital)
            val_sharpe = val_metrics['sharpe_ratio']
            val_return = val_metrics['total_return'] * 100

            validation_sharpes.append(val_sharpe)
            validation_episodes.append(episode)

            print(f"  [VALIDATION] Sharpe: {val_sharpe:+.3f} | Return: {val_return:+.2f}%")

            # Check for improvement
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                patience_counter = 0

                best_model_path = model_dir / "best_model.pth"
                torch.save(agent.policy.state_dict(), best_model_path)
                print(f"  [NEW BEST] Validation Sharpe: {val_sharpe:+.3f} (saved)")
            else:
                patience_counter += 1
                print(f"  [NO IMPROVEMENT] Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print()
                    print(f"  [EARLY STOPPING] No improvement for {patience} validations")
                    print(f"  [EARLY STOPPING] Best validation Sharpe: {best_val_sharpe:+.3f}")
                    print(f"  [EARLY STOPPING] Stopping at episode {episode}")
                    break

            print()

    print()
    print("=" * 80)
    print("BEAR SPECIALIST TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print(f"Episodes Completed: {episode}/{num_episodes}")
    print(f"Best Validation Sharpe: {best_val_sharpe:+.3f}")
    print(f"Final Training Sharpe (avg last 10): {np.mean(episode_sharpes[-10:]):+.3f}")
    print()

    # Save metrics
    training_metrics = {
        "regime": "BEAR",
        "episode_rewards": episode_rewards,
        "episode_returns": episode_returns,
        "episode_sharpes": episode_sharpes,
        "validation_sharpes": validation_sharpes,
        "validation_episodes": validation_episodes,
        "best_val_sharpe": best_val_sharpe,
        "total_episodes": episode,
        "early_stopped": (episode < num_episodes),
        "reward_type": "sharpe_direct"
    }

    metrics_path = model_dir / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    print()

    return training_metrics


if __name__ == "__main__":
    try:
        metrics = train_bear_specialist()
        print("=" * 80)
        print("✅ BEAR SPECIALIST TRAINING COMPLETED SUCCESSFULLY")
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

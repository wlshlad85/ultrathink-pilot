#!/usr/bin/env python3
"""
Universal Agent Training with Sharpe-Optimized Rewards
======================================================

This script trains a single universal agent using Sharpe-optimized rewards
to validate that the reward function fixes the negative Sharpe issue.

Key Differences from train_professional.py:
- Uses use_sharpe_reward=True in TradingEnv
- 200 episodes (quick validation, not 1000)
- Early stopping based on validation Sharpe (patience=50)
- Validates every 10 episodes
- Tracks Sharpe ratio as primary metric (not just return)

Expected Outcome:
- Validation Sharpe > 0.0 (beats current -1.172)
- Proves reward function fixes core issue
"""

import sys
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl.ppo_agent import PPOAgent
from rl.trading_env import TradingEnv
from rl.sharpe_reward import SharpeRewardCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_on_validation(agent, device, initial_capital):
    """
    Evaluate agent on 2022 validation set and return Sharpe ratio.

    Args:
        agent: Trained PPO agent
        device: torch device
        initial_capital: Initial capital

    Returns:
        dict with validation metrics including Sharpe ratio
    """
    val_env = TradingEnv(
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2022-12-31",
        initial_capital=initial_capital,
        commission_rate=0.001,
        use_sharpe_reward=True  # Use same reward for consistency
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

    # Get episode metrics from Sharpe calculator
    if val_env.sharpe_calculator:
        metrics = val_env.sharpe_calculator.get_episode_metrics()
    else:
        # Fallback if calculator not available
        final_value = val_env.portfolio.get_total_value()
        total_return = (final_value - initial_capital) / initial_capital
        metrics = {
            'sharpe_ratio': 0.0,
            'total_return': total_return,
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'trade_count': 0
        }

    metrics['final_value'] = val_env.portfolio.get_total_value()
    agent.policy.train()  # Back to training mode

    return metrics


def train_universal_sharpe():
    """Train universal agent with Sharpe-optimized rewards."""

    print("=" * 80)
    print("UNIVERSAL AGENT TRAINING - SHARPE-OPTIMIZED REWARDS")
    print("=" * 80)
    print()
    print("Training Configuration:")
    print("  Training Data:     2017-2021 (5 years, all market regimes)")
    print("  Validation Data:   2022 (1 year, bear market)")
    print()
    print("  Episodes:          200 (with early stopping)")
    print("  Early Stopping:    50 episode patience on validation Sharpe")
    print("  Validation Freq:   Every 10 episodes")
    print("  GPU:               CUDA (RTX 5070)")
    print("  Algorithm:         PPO with Sharpe rewards")
    print()
    print("Key Features:")
    print("  [+] Sharpe-optimized reward function (4 components)")
    print("  [+] Risk-adjusted performance optimization")
    print("  [+] Drawdown penalty for capital preservation")
    print("  [+] Trading cost penalty to avoid overtrading")
    print("  [+] Exploration bonus to prevent convergence")
    print("=" * 80)
    print()

    root_dir = Path(__file__).parent
    model_dir = root_dir / "rl" / "models" / "sharpe_universal"
    log_dir = root_dir / "rl" / "logs" / "sharpe_universal"

    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment with Sharpe rewards
    print("=" * 80)
    print("INITIALIZING TRAINING ENVIRONMENT WITH SHARPE REWARDS")
    print("=" * 80)
    print()

    initial_capital = 100000.0

    train_env = TradingEnv(
        symbol="BTC-USD",
        start_date="2017-01-01",
        end_date="2021-12-31",
        initial_capital=initial_capital,
        commission_rate=0.001,
        use_sharpe_reward=True  # KEY CHANGE: Use Sharpe rewards!
    )

    print(f"  [DONE] Sharpe reward environment created")
    print(f"         State dimensions:  {train_env.observation_space.shape[0]}")
    print(f"         Action space:      {train_env.action_space.n} (BUY/HOLD/SELL)")
    print(f"         Training period:   2017-2021 (5 years)")
    print(f"         Initial capital:   ${initial_capital:,.2f}")
    print(f"         Sharpe calculator: Enabled ✓")
    print()

    # Initialize PPO agent
    print("=" * 80)
    print("INITIALIZING PPO AGENT")
    print("=" * 80)
    print()

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

    print(f"PPO Agent initialized:")
    print(f"  Device:            {device}")
    print(f"  Learning rate:     3e-4")
    print(f"  Discount factor:   0.99")
    print(f"  PPO clip:          0.2")
    print(f"  Update epochs:     4")
    print()

    # Training configuration
    print("=" * 80)
    print("TRAINING WITH EARLY STOPPING")
    print("=" * 80)
    print()

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
    print(f"Validating every {val_freq} episodes...")
    print(f"Early stopping patience: {patience} episodes")
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

            # Store reward and terminal
            agent.store_reward_and_terminal(reward, terminated or truncated)

            episode_reward += reward
            state = next_state
            step += 1

            # Update policy
            if step % update_freq == 0:
                agent.update()

            if terminated or truncated:
                break

        # Record episode metrics
        final_value = train_env.portfolio.get_total_value()
        episode_return = ((final_value - initial_capital) / initial_capital) * 100

        # Get Sharpe ratio from calculator
        if train_env.sharpe_calculator:
            train_metrics = train_env.sharpe_calculator.get_episode_metrics()
            episode_sharpe = train_metrics['sharpe_ratio']
        else:
            episode_sharpe = 0.0

        episode_rewards.append(episode_reward)
        episode_returns.append(episode_return)
        episode_sharpes.append(episode_sharpe)

        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_return = np.mean(episode_returns[-10:])
            avg_sharpe = np.mean(episode_sharpes[-10:])
            print(f"Episode {episode}/{num_episodes} | "
                  f"Reward: {episode_reward:.4f} | "
                  f"Return: {episode_return:+.2f}% | "
                  f"Sharpe: {episode_sharpe:+.3f} | "
                  f"Avg(10) Sharpe: {avg_sharpe:+.3f}")

        # Validation and early stopping check
        if episode % val_freq == 0:
            print(f"  [VALIDATION] Running validation on 2022 bear market...")
            val_metrics = evaluate_on_validation(agent, device, initial_capital)
            val_sharpe = val_metrics['sharpe_ratio']
            val_return = val_metrics['total_return'] * 100

            validation_sharpes.append(val_sharpe)
            validation_episodes.append(episode)

            print(f"  [VALIDATION] Sharpe: {val_sharpe:+.3f} | "
                  f"Return: {val_return:+.2f}% | "
                  f"MaxDD: {val_metrics['max_drawdown']*100:.2f}%")

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
    print(f"Model saved to: {model_dir / 'best_model.pth'}")
    print()

    # Save training metrics
    training_metrics = {
        "episode_rewards": episode_rewards,
        "episode_returns": episode_returns,
        "episode_sharpes": episode_sharpes,
        "validation_sharpes": validation_sharpes,
        "validation_episodes": validation_episodes,
        "best_val_sharpe": best_val_sharpe,
        "total_episodes": episode,
        "early_stopped": (episode < num_episodes),
        "training_period": "2017-2021",
        "reward_type": "sharpe_optimized"
    }

    metrics_path = model_dir / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Training metrics saved to: {metrics_path}")
    print()

    # Final validation
    print("=" * 80)
    print("FINAL VALIDATION ON 2022 BEAR MARKET")
    print("=" * 80)
    print()
    val_metrics = evaluate_on_validation(agent, device, initial_capital)
    print(f"  Sharpe Ratio:      {val_metrics['sharpe_ratio']:+.3f}")
    print(f"  Total Return:      {val_metrics['total_return']*100:+.2f}%")
    print(f"  Volatility:        {val_metrics['volatility']*100:.2f}%")
    print(f"  Max Drawdown:      {val_metrics['max_drawdown']*100:.2f}%")
    print(f"  Total Trades:      {val_metrics['trade_count']}")
    print()

    # Compare to baseline
    baseline_sharpe = -1.172  # From current model
    improvement = val_metrics['sharpe_ratio'] - baseline_sharpe

    print("=" * 80)
    print("COMPARISON TO BASELINE")
    print("=" * 80)
    print()
    print(f"  Baseline (old reward):  {baseline_sharpe:+.3f}")
    print(f"  New (Sharpe reward):    {val_metrics['sharpe_ratio']:+.3f}")
    print(f"  Improvement:            {improvement:+.3f}")
    print()

    if val_metrics['sharpe_ratio'] > 0.0:
        print("  ✅ SUCCESS: Achieved positive Sharpe ratio!")
        print("     Reward function fixes the core issue.")
    elif val_metrics['sharpe_ratio'] > baseline_sharpe:
        print("  ⚠️  PROGRESS: Sharpe improved but still negative.")
        print("     Further tuning needed.")
    else:
        print("  ❌ FAILURE: No improvement over baseline.")
        print("     Reward function may need redesign.")
    print()

    return training_metrics


if __name__ == "__main__":
    try:
        metrics = train_universal_sharpe()
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

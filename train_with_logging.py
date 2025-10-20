#!/usr/bin/env python3
"""
Professional Training with Database Logging
============================================

This script includes full experiment tracking to the SQLite database.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from rl.trading_env_v2 import TradingEnvV2
from rl.ppo_agent import PPOAgent
from experiment_logger import ExperimentLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def create_professional_splits():
    """Create 60/20/20 temporal splits."""
    logger.info("="*80)
    logger.info("PROFESSIONAL DATA SPLITS (60/20/20)")
    logger.info("="*80)
    logger.info("")

    train_env = TradingEnvV2(
        symbol="BTC-USD",
        start_date="2017-01-01",
        end_date="2021-12-31",
        initial_capital=100000.0,
        use_regime_rewards=True
    )
    logger.info(f"✓ Training Set: 2017-2021 ({len(train_env.market_data)} days)")

    test_env = TradingEnvV2(
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2023-06-30",
        initial_capital=100000.0,
        use_regime_rewards=True
    )
    logger.info(f"✓ Test Set: 2022-2023 ({len(test_env.market_data)} days)")

    held_out_env = TradingEnvV2(
        symbol="BTC-USD",
        start_date="2023-07-01",
        end_date="2024-12-31",
        initial_capital=100000.0,
        use_regime_rewards=True
    )
    logger.info(f"✓ Held-out Set: 2023-2024 ({len(held_out_env.market_data)} days)")
    logger.info("")
    logger.info("="*80)
    logger.info("")

    return train_env, test_env, held_out_env


def evaluate(agent, env, device, num_episodes=3):
    """Evaluate agent and return metrics."""
    agent.policy.eval()

    returns = []
    sharpes = []

    with torch.no_grad():
        for ep in range(num_episodes):
            state, info = env.reset()
            done = False
            steps = 0

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_probs, _ = agent.policy(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()

                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

            final_value = env.portfolio.get_total_value()
            episode_return = ((final_value - 100000) / 100000) * 100
            returns.append(episode_return)

            if 'sharpe_ratio' in info:
                sharpes.append(info['sharpe_ratio'])

    agent.policy.train()

    return {
        'mean_return': np.mean(returns),
        'mean_sharpe': np.mean(sharpes) if sharpes else 0.0
    }


def train_with_logging(experiment_name=None, lr=3e-4, gamma=0.99, max_episodes=300):
    """Train with full database logging."""

    if experiment_name is None:
        experiment_name = f"Professional_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("="*80)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info("="*80)
    logger.info("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Gamma: {gamma}")
    logger.info(f"Max episodes: {max_episodes}")
    logger.info("")

    # Create environments
    train_env, test_env, held_out_env = create_professional_splits()

    # Create agent
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr=lr,
        gamma=gamma
    )

    # Initialize experiment logger
    exp_logger = ExperimentLogger()
    exp_id = exp_logger.start_experiment(
        name=experiment_name,
        description=f"300-ep training, lr={lr}, gamma={gamma}",
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=lr,
        gamma=gamma,
        train_start_date="2017-01-01",
        train_end_date="2021-12-31",
        test_start_date="2022-01-01",
        test_end_date="2023-06-30"
    )

    logger.info(f"📊 Experiment ID: {exp_id}")
    logger.info("")

    # Setup save directory
    save_dir = Path("rl/models/professional")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info("="*80)
    logger.info("TRAINING LOOP")
    logger.info("="*80)
    logger.info("")

    best_test_sharpe = -np.inf
    patience = 0
    max_patience = 4
    test_freq = 15
    final_episode = 0

    for ep in range(1, max_episodes + 1):
        final_episode = ep

        # Training episode
        state, info = train_env.reset()
        ep_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated

            agent.store_reward_and_terminal(reward, done)

            ep_reward += reward
            state = next_state
            steps += 1

        # Update policy
        agent.update()

        # Episode metrics
        final_value = train_env.portfolio.get_total_value()
        ep_return = ((final_value - 100000) / 100000) * 100

        # Log to database
        exp_logger.log_episode(
            episode_num=ep,
            train_return=ep_return,
            train_reward=ep_reward,
            episode_length=steps
        )

        # Console log
        if ep % 10 == 0:
            logger.info(f"Ep {ep:3d}/{max_episodes}: Return {ep_return:+7.2f}%  Reward {ep_reward:7.1f}")

        # Test set evaluation
        if ep % test_freq == 0:
            logger.info("")
            logger.info(f"--- TEST EVALUATION (Episode {ep}) ---")

            test_metrics = evaluate(agent, test_env, device, num_episodes=3)

            logger.info(f"  Return: {test_metrics['mean_return']:+.2f}%")
            logger.info(f"  Sharpe: {test_metrics['mean_sharpe']:+.3f}")

            is_best = test_metrics['mean_sharpe'] > best_test_sharpe

            # Log test metrics to database
            exp_logger.log_episode(
                episode_num=ep,
                train_return=ep_return,
                train_reward=ep_reward,
                episode_length=steps,
                test_return=test_metrics['mean_return'],
                test_sharpe=test_metrics['mean_sharpe'],
                is_best_model=is_best
            )

            # Early stopping
            if is_best:
                best_test_sharpe = test_metrics['mean_sharpe']
                patience = 0

                best_model_path = save_dir / "best_model.pth"
                agent.save(str(best_model_path))
                logger.info(f"  ✅ NEW BEST! Saved.")
            else:
                patience += 1
                logger.info(f"  Patience: {patience}/{max_patience}")

                if patience >= max_patience:
                    logger.info("")
                    logger.info("⏹️  EARLY STOPPING")
                    break

            logger.info("")

        # Save checkpoints
        if ep % 50 == 0:
            checkpoint_path = save_dir / f"checkpoint_ep{ep}.pth"
            agent.save(str(checkpoint_path))

    # Final evaluation
    logger.info("="*80)
    logger.info("FINAL TEST EVALUATION")
    logger.info("="*80)
    logger.info("")

    best_model_path = save_dir / "best_model.pth"
    agent.load(str(best_model_path))

    final_test_metrics = evaluate(agent, test_env, device, num_episodes=10)

    logger.info(f"Final test return: {final_test_metrics['mean_return']:+.2f}%")
    logger.info(f"Final test Sharpe: {final_test_metrics['mean_sharpe']:+.3f}")
    logger.info("")

    # End experiment in database
    exp_logger.end_experiment(
        final_episode=final_episode,
        best_test_sharpe=best_test_sharpe,
        best_test_return=final_test_metrics['mean_return'],
        early_stopped=(final_episode < max_episodes)
    )

    # Save final model
    final_model_path = save_dir / "final_model.pth"
    agent.save(str(final_model_path))

    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Experiment ID: {exp_id}")
    logger.info(f"Best test Sharpe: {best_test_sharpe:+.3f}")
    logger.info(f"Final test return: {final_test_metrics['mean_return']:+.2f}%")
    logger.info("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--episodes", type=int, default=300, help="Max episodes")

    args = parser.parse_args()

    train_with_logging(
        experiment_name=args.name,
        lr=args.lr,
        gamma=args.gamma,
        max_episodes=args.episodes
    )

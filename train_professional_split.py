#!/usr/bin/env python3
"""
Professional Training with Industry-Standard 60/20/20 Temporal Split
=====================================================================

Data Philosophy:
- Training (60%):  2017-2022 → Learn regime-conditional policies
- Test (20%):      2022-2023 → Evaluate during development
- Held-out (20%):  2023-today → NEVER use until deployment decision
- Future (>today): Live validation after deployment

This prevents data leakage and simulates real deployment conditions.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

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
    """
    Create 60/20/20 temporal splits following industry best practices.

    Returns:
        tuple: (train_env, test_env, held_out_env)
    """
    logger.info("="*80)
    logger.info("PROFESSIONAL DATA SPLITS (60/20/20)")
    logger.info("="*80)
    logger.info("")

    # Training: 2017-2021 (60% - 5 years)
    # This is where the model LEARNS
    train_env = TradingEnvV2(
        symbol="BTC-USD",
        start_date="2017-01-01",
        end_date="2021-12-31",
        initial_capital=100000.0,
        use_regime_rewards=True
    )
    logger.info(f"✓ Training Set (60%):   2017-2022")
    logger.info(f"  Purpose: Learn regime-conditional policies")
    logger.info(f"  Data points: {len(train_env.market_data)} days")
    logger.info(f"  Use: Policy optimization, weight updates")
    logger.info("")

    # Test: 2022-2023 (20% - ~1.5 years)
    # This is for EVALUATION during development
    test_env = TradingEnvV2(
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2023-06-30",
        initial_capital=100000.0,
        use_regime_rewards=True
    )
    logger.info(f"✓ Test Set (20%):       2022-2023")
    logger.info(f"  Purpose: Evaluate during training (early stopping)")
    logger.info(f"  Data points: {len(test_env.market_data)} days")
    logger.info(f"  Use: Monitor generalization, prevent overfitting")
    logger.info("")

    # Held-out: 2023-today (20% - ~2 years)
    # This is COMPLETELY UNTOUCHED until deployment decision
    held_out_env = TradingEnvV2(
        symbol="BTC-USD",
        start_date="2023-07-01",
        end_date="2024-12-31",
        initial_capital=100000.0,
        use_regime_rewards=True
    )
    logger.info(f"✓ Held-out Set (20%):   2023-today")
    logger.info(f"  Purpose: Final unbiased performance assessment")
    logger.info(f"  Data points: {len(held_out_env.market_data)} days")
    logger.info(f"  Use: ONLY evaluate ONCE before deployment")
    logger.info("")
    logger.info("⚠️  CRITICAL: Held-out set must NOT be used during training!")
    logger.info("   Only evaluate on it when deciding to deploy to production.")
    logger.info("")
    logger.info("="*80)
    logger.info("")

    return train_env, test_env, held_out_env


def evaluate(agent, env, device, num_episodes=3, desc="Evaluation"):
    """Evaluate agent on environment."""
    agent.policy.eval()

    returns = []
    sharpes = []

    with torch.no_grad():
        for ep in range(num_episodes):
            state, info = env.reset()
            done = False

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_probs, _ = agent.policy(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()

                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            final_value = env.portfolio.get_total_value()
            episode_return = ((final_value - 100000) / 100000) * 100
            returns.append(episode_return)

            if 'sharpe_ratio' in info:
                sharpes.append(info['sharpe_ratio'])

    agent.policy.train()

    mean_return = np.mean(returns)
    mean_sharpe = np.mean(sharpes) if sharpes else 0.0

    logger.info(f"{desc}:")
    logger.info(f"  Return: {mean_return:+.2f}% (avg over {num_episodes} episodes)")
    logger.info(f"  Sharpe: {mean_sharpe:+.3f}")

    return {
        'mean_return': mean_return,
        'mean_sharpe': mean_sharpe
    }


def train_300_episodes():
    """Train for 300 episodes with proper validation."""

    logger.info("="*80)
    logger.info("PROFESSIONAL 300-EPISODE TRAINING")
    logger.info("="*80)
    logger.info("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info("")

    # Setup save directory
    save_dir = Path("rl/models/professional")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models will be saved to: {save_dir}")
    logger.info("")

    # Create environments with 60/20/20 split
    train_env, test_env, held_out_env = create_professional_splits()

    # Create agent
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr=3e-4,
        gamma=0.99
    )

    logger.info(f"Agent Architecture: {state_dim} states → {action_dim} actions")
    logger.info("")

    # Training loop
    logger.info("="*80)
    logger.info("TRAINING LOOP (300 EPISODES)")
    logger.info("="*80)
    logger.info("")

    best_test_sharpe = -np.inf
    patience = 0
    max_patience = 4
    test_freq = 15

    for ep in range(1, 301):
        # Training episode
        state, info = train_env.reset()
        ep_reward = 0
        done = False

        while not done:
            # Select action using agent's method (automatically stores state, action, log_prob, value)
            action = agent.select_action(state)

            # Execute action in environment
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated

            # Store reward and terminal flag
            agent.store_reward_and_terminal(reward, done)

            ep_reward += reward
            state = next_state

        # Update policy after each episode
        agent.update()

        # Episode metrics
        final_value = train_env.portfolio.get_total_value()
        ep_return = ((final_value - 100000) / 100000) * 100

        # Log progress
        if ep % 10 == 0:
            logger.info(f"Episode {ep:3d}/300: Return {ep_return:+7.2f}%  Reward {ep_reward:7.1f}")

        # Test set evaluation (for early stopping)
        if ep % test_freq == 0:
            logger.info("")
            logger.info("-" * 80)
            logger.info(f"TEST SET EVALUATION (Episode {ep})")
            logger.info("-" * 80)

            test_metrics = evaluate(agent, test_env, device, num_episodes=3, desc="Test Set (2022-2023)")

            # Early stopping based on test set
            if test_metrics['mean_sharpe'] > best_test_sharpe:
                best_test_sharpe = test_metrics['mean_sharpe']
                patience = 0

                # Save best model
                best_model_path = save_dir / "best_model.pth"
                agent.save(str(best_model_path))
                logger.info(f"✅ NEW BEST! Sharpe: {best_test_sharpe:+.3f} (saved)")
            else:
                patience += 1
                logger.info(f"No improvement. Patience: {patience}/{max_patience}")

                if patience >= max_patience:
                    logger.info("")
                    logger.info("="*80)
                    logger.info(f"⏹️  EARLY STOPPING at Episode {ep}")
                    logger.info("="*80)
                    logger.info(f"Best test Sharpe: {best_test_sharpe:+.3f}")
                    break

            logger.info("-" * 80)
            logger.info("")

        # Save checkpoints
        if ep % 50 == 0:
            checkpoint_path = save_dir / f"checkpoint_ep{ep}.pth"
            agent.save(str(checkpoint_path))

    # Load best model
    logger.info("")
    logger.info("="*80)
    logger.info("TRAINING COMPLETE - LOADING BEST MODEL")
    logger.info("="*80)
    logger.info("")

    best_model_path = save_dir / "best_model.pth"
    agent.load(str(best_model_path))
    logger.info(f"Loaded best model: {best_model_path}")
    logger.info(f"Best test Sharpe: {best_test_sharpe:+.3f}")
    logger.info("")

    # Final evaluation on test set
    logger.info("="*80)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("="*80)
    logger.info("")

    test_metrics = evaluate(agent, test_env, device, num_episodes=10, desc="Test Set (2022-2023) - Final")
    logger.info("")

    # Save final model
    final_model_path = save_dir / "final_model.pth"
    agent.save(str(final_model_path))

    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info("")
    logger.info(f"✓ Training completed on 2017-2021 data")
    logger.info(f"✓ Best test Sharpe (2022-2023): {best_test_sharpe:+.3f}")
    logger.info(f"✓ Final test return: {test_metrics['mean_return']:+.2f}%")
    logger.info(f"✓ Models saved to: {save_dir}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Analyze regime-conditional behavior:")
    logger.info("   python analyze_regime_aware.py")
    logger.info("")
    logger.info("2. ONLY WHEN READY TO DEPLOY - Evaluate on held-out set:")
    logger.info("   python evaluate_professional.py")
    logger.info("")
    logger.info("3. If held-out performance is satisfactory:")
    logger.info("   Deploy to paper trading on live data (>today)")
    logger.info("")
    logger.info("⚠️  Remember: NEVER use held-out set (2023-today) during development!")
    logger.info("   It's your final unbiased performance check before production.")
    logger.info("")
    logger.info("="*80)


if __name__ == "__main__":
    train_300_episodes()

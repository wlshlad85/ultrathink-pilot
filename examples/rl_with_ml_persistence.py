#!/usr/bin/env python3
"""
Example: RL Training with ML Persistence

Demonstrates how to integrate the ML persistence system with RL training.
This is a complete working example that extends the existing rl/train.py
with comprehensive experiment tracking.
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.trading_env import TradingEnv
from rl.ppo_agent import PPOAgent
from ml_persistence import ExperimentTracker, ModelRegistry, DatasetManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_persistence(
    num_episodes: int = 100,
    max_steps: int = 1000,
    update_freq: int = 2048,
    save_freq: int = 10,
    symbol: str = "BTC-USD",
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    initial_capital: float = 100000.0,
    experiment_name: str = None,
    tags: list = None
):
    """
    Train PPO agent with full ML persistence tracking.

    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        update_freq: Update policy after this many steps
        save_freq: Save model every N episodes
        symbol: Trading symbol
        start_date: Training data start date
        end_date: Training data end date
        initial_capital: Initial portfolio capital
        experiment_name: Custom experiment name (auto-generated if None)
        tags: List of tags for experiment
    """

    # ========================================================================
    # STEP 1: Initialize ML Persistence System
    # ========================================================================

    # Create experiment tracker
    tracker = ExperimentTracker("ml_experiments.db")

    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"PPO_{symbol}_{timestamp}"

    # Start experiment with full metadata
    exp_id = tracker.start_experiment(
        name=experiment_name,
        experiment_type="rl",
        description=f"PPO agent training on {symbol} from {start_date} to {end_date}",
        tags=tags or ["rl", "ppo", symbol.lower()],
        random_seed=42,  # TODO: Make configurable
        capture_git=True,
        metadata={
            "framework": "PPO",
            "environment": "TradingEnv",
            "symbol": symbol,
            "initial_capital": initial_capital
        }
    )

    logger.info(f"Started experiment: {experiment_name} (ID: {exp_id})")

    # ========================================================================
    # STEP 2: Initialize Environment
    # ========================================================================

    env = TradingEnv(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )

    logger.info(f"Environment initialized: {len(env.market_data)} data points")

    # ========================================================================
    # STEP 3: Register Dataset
    # ========================================================================

    dataset_mgr = DatasetManager("ml_experiments.db")

    train_dataset_id = dataset_mgr.register_dataset(
        name=f"{symbol}-Daily",
        version=f"{start_date}_{end_date}",
        split_type="train",
        dataset_type="timeseries",
        num_samples=len(env.market_data),
        start_date=start_date,
        end_date=end_date,
        feature_columns=[
            "close", "rsi_14", "macd", "macd_signal", "atr_14",
            "bb_upper", "bb_lower", "volume_ratio", "returns_1d",
            "returns_5d", "returns_20d"
        ],
        metadata={
            "skip_days": env.start_idx,
            "window_size": env.window_size
        }
    )

    # Link dataset to experiment
    dataset_mgr.link_dataset_to_experiment(exp_id, train_dataset_id, "train")

    # ========================================================================
    # STEP 4: Initialize Agent
    # ========================================================================

    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4
    )

    logger.info(f"Agent initialized on device: {agent.device}")

    # Log hyperparameters
    tracker.log_hyperparameters_batch({
        "state_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.n,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "eps_clip": 0.2,
        "k_epochs": 4,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "update_freq": update_freq,
        "window_size": env.window_size,
        "commission_rate": env.commission_rate
    })

    # ========================================================================
    # STEP 5: Training Loop with Metrics Logging
    # ========================================================================

    model_registry = ModelRegistry("ml_experiments.db")
    model_dir = Path("rl/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    best_return = -float('inf')

    logger.info("Starting training loop...")

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0

        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store reward and terminal
            agent.store_reward_and_terminal(reward, terminated or truncated)

            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            state = next_state

            # Update policy
            if total_steps % update_freq == 0:
                metrics = agent.update()

                # Log training metrics
                tracker.log_metrics_batch({
                    "policy_loss": metrics['policy_loss'],
                    "value_loss": metrics['value_loss'],
                    "total_loss": metrics['loss'],
                    "entropy": metrics['entropy']
                }, step=total_steps, split="train", phase="training")

                logger.info(f"Policy updated at step {total_steps}")

            if terminated or truncated:
                break

        # Log episode metrics
        final_return = info['total_return']

        tracker.log_metrics_batch({
            "episode_return": final_return,
            "episode_reward": episode_reward,
            "episode_length": episode_steps,
            "portfolio_value": info['portfolio_value'],
            "total_trades": info['total_trades']
        }, episode=episode, split="train", phase="training")

        logger.info(
            f"Episode {episode}/{num_episodes}: "
            f"Return={final_return:.2f}%, "
            f"Reward={episode_reward:.4f}, "
            f"Steps={episode_steps}"
        )

        # ====================================================================
        # STEP 6: Save Checkpoints with Model Registry
        # ====================================================================

        if episode % save_freq == 0:
            checkpoint_path = model_dir / f"checkpoint_ep{episode}.pth"
            agent.save(str(checkpoint_path))

            # Check if this is the best model
            is_best = final_return > best_return
            if is_best:
                best_return = final_return

            # Register model in database
            model_id = model_registry.register_model(
                experiment_id=exp_id,
                checkpoint_path=str(checkpoint_path),
                name=f"{experiment_name}_ep{episode}",
                version=f"ep{episode}",
                architecture_type="ppo",
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                episode_num=episode,
                global_step=total_steps,
                train_metric=final_return,
                is_best=is_best,
                hyperparameters={
                    "lr": 3e-4,
                    "gamma": 0.99,
                    "eps_clip": 0.2
                },
                metadata={
                    "portfolio_value": info['portfolio_value'],
                    "total_trades": info['total_trades']
                }
            )

            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # ========================================================================
    # STEP 7: End Experiment
    # ========================================================================

    # Get portfolio stats
    portfolio_stats = env.get_portfolio_stats()

    # Save final model
    final_checkpoint = model_dir / "final_model.pth"
    agent.save(str(final_checkpoint))

    model_registry.register_model(
        experiment_id=exp_id,
        checkpoint_path=str(final_checkpoint),
        name=f"{experiment_name}_final",
        version="final",
        architecture_type="ppo",
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        episode_num=num_episodes,
        global_step=total_steps,
        train_metric=final_return,
        is_best=False,
        metadata=portfolio_stats
    )

    # End experiment with summary
    notes = f"""
    Training completed successfully.
    Best return: {best_return:.2f}%
    Final return: {final_return:.2f}%
    Total episodes: {num_episodes}
    Total steps: {total_steps}
    Win rate: {portfolio_stats.get('win_rate_pct', 0):.2f}%
    """

    tracker.end_experiment(status="completed", notes=notes)

    logger.info("="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Best return: {best_return:.2f}%")
    logger.info(f"Final return: {final_return:.2f}%")
    logger.info(f"Experiment ID: {exp_id}")
    logger.info("="*80)

    return exp_id


def main():
    parser = argparse.ArgumentParser(description="RL training with ML persistence")

    # Training parameters
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--update-freq", type=int, default=2048, help="Policy update frequency")
    parser.add_argument("--save-freq", type=int, default=10, help="Model save frequency")

    # Environment parameters
    parser.add_argument("--symbol", default="BTC-USD", help="Trading symbol")
    parser.add_argument("--start-date", default="2023-01-01", help="Training start date")
    parser.add_argument("--end-date", default="2024-01-01", help="Training end date")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")

    # Experiment metadata
    parser.add_argument("--name", help="Experiment name (auto-generated if not provided)")
    parser.add_argument("--tags", nargs="+", help="Tags for experiment")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("RL TRAINING WITH ML PERSISTENCE")
    print("="*80)
    print(f"Symbol:          {args.symbol}")
    print(f"Period:          {args.start_date} to {args.end_date}")
    print(f"Episodes:        {args.episodes}")
    print(f"Update Freq:     {args.update_freq}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print("="*80 + "\n")

    exp_id = train_with_persistence(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        update_freq=args.update_freq,
        save_freq=args.save_freq,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        experiment_name=args.name,
        tags=args.tags
    )

    print(f"\n✅ Training complete! Experiment ID: {exp_id}")
    print(f"   View results: SELECT * FROM experiment_summary WHERE id = {exp_id};")


if __name__ == "__main__":
    main()

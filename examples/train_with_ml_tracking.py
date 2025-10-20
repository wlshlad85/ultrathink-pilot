#!/usr/bin/env python3
"""
Example: Professional RL Training WITH ML Persistence Tracking
================================================================

This example shows how to integrate ml_persistence tracking into your RL training.

Key Features Demonstrated:
1. Experiment initialization with metadata
2. Hyperparameter logging
3. Dataset registration for train/val/test splits
4. Real-time metrics logging during training
5. Model checkpoint registration
6. Validation and test metrics tracking
7. Proper experiment lifecycle management

To integrate into train_professional.py:
- Add the ml_persistence imports
- Initialize tracker after creating directories
- Log hyperparameters after creating agent
- Register datasets after creating environments
- Log metrics in training loop
- Register model checkpoints when saving
- Update validation/test metrics
- End experiment at completion
"""

import sys
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.ppo_agent import PPOAgent
from rl.trading_env import TradingEnv

# === ML Persistence Integration ===
from ml_persistence import ExperimentTracker, ModelRegistry, DatasetManager


def train_with_tracking_example():
    """
    Demonstration of complete ML tracking integration.
    """

    print("=" * 80)
    print("RL TRAINING WITH ML PERSISTENCE TRACKING")
    print("=" * 80)
    print()

    # Setup directories
    root_dir = Path(__file__).parent.parent
    model_dir = root_dir / "rl" / "models" / "tracked_example"
    model_dir.mkdir(parents=True, exist_ok=True)

    # === STEP 1: Initialize Experiment Tracking ===
    print("Initializing experiment tracking...")
    tracker = ExperimentTracker()
    exp_id = tracker.start_experiment(
        name=f"PPO_Example_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        experiment_type="rl",
        description="Example: PPO training with full ML tracking",
        tags=["rl", "ppo", "bitcoin", "example"],
        random_seed=42
    )
    print(f"✅ Started experiment {exp_id}")
    print()

    # === STEP 2: Create Environment & Agent ===
    print("Creating environment...")
    initial_capital = 100000.0
    train_start = "2020-01-01"
    train_end = "2021-12-31"

    train_env = TradingEnv(
        symbol="BTC-USD",
        start_date=train_start,
        end_date=train_end,
        initial_capital=initial_capital,
        commission_rate=0.001
    )
    print(f"✅ Environment created: {len(train_env.market_data)} days of data")
    print()

    print("Creating PPO agent...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training hyperparameters
    lr = 3e-4
    gamma = 0.99
    eps_clip = 0.2
    k_epochs = 4

    agent = PPOAgent(
        state_dim=int(train_env.observation_space.shape[0]),
        action_dim=int(train_env.action_space.n),
        device=device,
        lr=lr,
        gamma=gamma,
        eps_clip=eps_clip,
        k_epochs=k_epochs
    )
    print(f"✅ Agent created on {device}")
    print()

    # === STEP 3: Log Hyperparameters ===
    print("Logging hyperparameters...")
    tracker.log_hyperparameters_batch({
        "learning_rate": float(lr),
        "gamma": float(gamma),
        "eps_clip": float(eps_clip),
        "k_epochs": int(k_epochs),
        "initial_capital": float(initial_capital),
        "commission_rate": 0.001,
        "state_dim": int(train_env.observation_space.shape[0]),
        "action_dim": int(train_env.action_space.n),
        "update_freq": 2048
    })
    print("✅ Hyperparameters logged")
    print()

    # === STEP 4: Register Training Dataset ===
    print("Registering dataset...")
    dataset_mgr = DatasetManager()
    train_dataset_id = dataset_mgr.register_dataset(
        name="BTC-USD-Daily",
        version=f"{train_start}_{train_end}",
        split_type="train",
        start_date=train_start,
        end_date=train_end,
        num_samples=int(len(train_env.market_data)),
        metadata={"data_source": "yfinance"},
        
    )
    dataset_mgr.link_dataset_to_experiment(exp_id, train_dataset_id, "train")
    print(f"✅ Dataset registered (ID: {train_dataset_id})")
    print()

    # === STEP 5: Training Loop with Metrics Logging ===
    print("Starting training with metrics tracking...")
    num_episodes = 100  # Reduced for example
    update_freq = 2048
    save_freq = 25

    episode_rewards = []
    episode_returns = []
    best_return = -float('inf')

    # Initialize model registry for checkpoints
    model_registry = ModelRegistry()

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

        # Calculate episode metrics
        final_value = train_env.portfolio.get_total_value()
        episode_return = ((final_value - initial_capital) / initial_capital) * 100

        episode_rewards.append(episode_reward)
        episode_returns.append(episode_return)

        # === LOG METRICS TO DATABASE ===
        tracker.log_metrics_batch({
            "train_reward": float(episode_reward),
            "train_return": float(episode_return),
            "episode_length": int(step),
            "final_portfolio_value": float(final_value),
            "avg_return_last_10": np.mean(episode_returns[-10:])
        }, episode=episode)

        # Print progress
        if episode % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            print(f"Episode {episode}/{num_episodes} | "
                  f"Return: {episode_return:+.2f}% | "
                  f"Avg(10): {avg_return:+.2f}%")

        # === SAVE AND REGISTER CHECKPOINTS ===
        if episode % save_freq == 0:
            checkpoint_path = model_dir / f"checkpoint_ep{episode}.pth"
            torch.save(agent.policy.state_dict(), checkpoint_path)

            # Register model in database
            model_id = model_registry.register_model(
                experiment_id=exp_id,
                name=f"PPO_checkpoint_ep{episode}",
                checkpoint_path=str(checkpoint_path),
                architecture_type="ppo",
                state_dim=int(train_env.observation_space.shape[0]),
                action_dim=int(train_env.action_space.n),
                episode_num=int(episode),
                train_metric=float(episode_return),
                is_best=False,
                metadata={
                    "avg_return_last_10": float(np.mean(episode_returns[-10:])),
                    "total_episodes_so_far": episode
                }
            )
            print(f"  [CHECKPOINT] Saved and registered model (ID: {model_id})")

        # === TRACK BEST MODEL ===
        if episode_return > best_return:
            best_return = episode_return
            best_model_path = model_dir / "best_model.pth"
            torch.save(agent.policy.state_dict(), best_model_path)

            # Register as best model
            best_model_id = model_registry.register_model(
                experiment_id=exp_id,
                name="PPO_best_model",
                checkpoint_path=str(best_model_path),
                architecture_type="ppo",
                state_dim=int(train_env.observation_space.shape[0]),
                action_dim=int(train_env.action_space.n),
                episode_num=int(episode),
                train_metric=float(episode_return),
                is_best=True,
                metadata={"best_return": float(best_return)}
            )
            print(f"  [NEW BEST] Episode {episode}: {episode_return:+.2f}% (Model ID: {best_model_id})")

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Training Return: {best_return:+.2f}%")
    print()

    # === STEP 6: Validation Evaluation ===
    print("Running validation...")
    val_env = TradingEnv(
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2022-12-31",
        initial_capital=initial_capital,
        commission_rate=0.001
    )

    # Register validation dataset
    val_dataset_id = dataset_mgr.register_dataset(
        name="BTC-USD-Daily",
        version="2022-01-01_2022-12-31",
        split_type="val",
        start_date="2022-01-01",
        end_date="2022-12-31",
        num_samples=int(len(val_env.market_data)),
        data_source="yfinance"
    )
    dataset_mgr.link_dataset_to_experiment(exp_id, val_dataset_id, "val")

    # Run validation
    agent.policy.eval()
    state, info = val_env.reset()
    val_reward = 0.0

    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs = agent.policy(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()

        state, reward, terminated, truncated, info = val_env.step(action)
        val_reward += reward

        if terminated or truncated:
            break

    val_final_value = val_env.portfolio.get_total_value()
    val_return = ((val_final_value - initial_capital) / initial_capital) * 100

    # === LOG VALIDATION METRICS ===
    tracker.log_metrics_batch({
        "val_return": float(val_return),
        "val_reward": float(val_reward),
        "val_final_value": float(val_final_value)
    }, split="val")

    print(f"✅ Validation Return: {val_return:+.2f}%")
    print()

    # === STEP 7: End Experiment ===
    print("Ending experiment...")
    tracker.end_experiment(
        status="completed",
        notes=f"Training completed successfully. Best return: {best_return:.2f}%, Val return: {val_return:.2f}%"
    )
    print("✅ Experiment tracking ended")
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Experiment ID:      {exp_id}")
    print(f"Training Episodes:  {num_episodes}")
    print(f"Best Train Return:  {best_return:+.2f}%")
    print(f"Validation Return:  {val_return:+.2f}%")
    print()
    print("To view experiment:")
    print(f"  python3 -c \"from ml_persistence import ExperimentTracker; t=ExperimentTracker(); print(t.get_experiment({exp_id}))\"")
    print()
    print("=" * 80)

    return {
        "exp_id": exp_id,
        "best_return": best_return,
        "val_return": val_return
    }


if __name__ == "__main__":
    try:
        results = train_with_tracking_example()
        print("\n✅ Training with tracking completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

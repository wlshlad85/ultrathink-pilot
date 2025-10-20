#!/usr/bin/env python3
"""
Script to integrate ML persistence tracking into train_professional.py
"""

# Read the original file
with open('train_professional.py', 'r') as f:
    content = f.read()

# 1. Add ML persistence imports after existing imports
import_addition = '''from rl.ppo_agent import PPOAgent
from rl.trading_env import TradingEnv
from backtesting.data_fetcher import DataFetcher

# ML Persistence for experiment tracking
from ml_persistence import ExperimentTracker, ModelRegistry, DatasetManager

logging.basicConfig('''

content = content.replace(
    '''from rl.ppo_agent import PPOAgent
from rl.trading_env import TradingEnv
from backtesting.data_fetcher import DataFetcher

logging.basicConfig(''',
    import_addition
)

# 2. Add experiment tracker initialization after directory creation
tracker_init = '''    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # === ML PERSISTENCE: Initialize Experiment Tracking ===
    print("\\n" + "=" * 80)
    print("ML EXPERIMENT TRACKING")
    print("=" * 80)
    print()

    tracker = ExperimentTracker()
    exp_id = tracker.start_experiment(
        name=f"PPO_Professional_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        experiment_type="rl",
        description="Professional institutional RL training: Single agent on full market cycle (2017-2021), 1000 episodes",
        tags=["rl", "ppo", "bitcoin", "professional", "multi-regime", "institutional"],
        random_seed=42
    )
    print(f"✅ Experiment tracking started (ID: {exp_id})")
    print()

    # STEP 1: Create training environment'''

content = content.replace(
    '''    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1: Create training environment''',
    tracker_init
)

# 3. Add dataset registration after environment creation
dataset_reg = '''    print(f"         Commission rate:   0.1% per trade")
    print()

    # === ML PERSISTENCE: Register Training Dataset ===
    dataset_mgr = DatasetManager()
    train_dataset_id = dataset_mgr.register_dataset(
        name="BTC-USD-Daily",
        version="2017-2021",
        split_type="train",
        start_date="2017-01-01",
        end_date="2021-12-31",
        num_samples=int(len(train_env.market_data)),
        metadata={"data_source": "yfinance", "market_regimes": "bull/bear/sideways"}
    )
    dataset_mgr.link_dataset_to_experiment(exp_id, train_dataset_id, "train")
    print()

    # STEP 2: Initialize PPO agent'''

content = content.replace(
    '''    print(f"         Commission rate:   0.1% per trade")
    print()

    # STEP 2: Initialize PPO agent''',
    dataset_reg
)

# 4. Add hyperparameter logging after agent creation
hyperparam_log = '''    print(f"  Update epochs:     4")
    print()

    # === ML PERSISTENCE: Log Hyperparameters ===
    tracker.log_hyperparameters_batch({
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "eps_clip": 0.2,
        "k_epochs": 4,
        "initial_capital": float(initial_capital),
        "commission_rate": 0.001,
        "state_dim": int(train_env.observation_space.shape[0]),
        "action_dim": int(train_env.action_space.n),
        "update_freq": 2048,
        "num_episodes": 1000,
        "save_freq": 50,
        "device": str(device)
    })
    print("✅ Hyperparameters logged to database")
    print()

    # STEP 3: Train for 1,000 episodes'''

content = content.replace(
    '''    print(f"  Update epochs:     4")
    print()

    # STEP 3: Train for 1,000 episodes''',
    hyperparam_log
)

# 5. Initialize model registry before training loop
registry_init = '''    episode_rewards = []
    episode_returns = []
    episode_lengths = []
    best_return = -float('inf')

    # === ML PERSISTENCE: Initialize Model Registry ===
    model_registry = ModelRegistry()

    for episode in range(1, num_episodes + 1):'''

content = content.replace(
    '''    episode_rewards = []
    episode_returns = []
    episode_lengths = []
    best_return = -float('inf')

    for episode in range(1, num_episodes + 1):''',
    registry_init
)

# 6. Add metrics logging in training loop
metrics_log = '''        episode_rewards.append(episode_reward)
        episode_returns.append(episode_return)
        episode_lengths.append(step)

        # === ML PERSISTENCE: Log Episode Metrics ===
        tracker.log_metrics_batch({
            "train_reward": float(episode_reward),
            "train_return": float(episode_return),
            "episode_length": int(step),
            "final_portfolio_value": float(final_value),
            "avg_return_last_10": float(np.mean(episode_returns[-10:])),
            "avg_return_last_100": float(np.mean(episode_returns[-100:])) if len(episode_returns) >= 100 else float(np.mean(episode_returns))
        }, episode=episode)

        # Print progress every 10 episodes'''

content = content.replace(
    '''        episode_rewards.append(episode_reward)
        episode_returns.append(episode_return)
        episode_lengths.append(step)

        # Print progress every 10 episodes''',
    metrics_log
)

# 7. Add model checkpoint registration
checkpoint_reg = '''        # Save checkpoint every save_freq episodes
        if episode % save_freq == 0:
            checkpoint_path = model_dir / f"episode_{episode}.pth"
            torch.save(agent.policy.state_dict(), checkpoint_path)
            print(f"  [CHECKPOINT] Saved model at episode {episode}")

            # === ML PERSISTENCE: Register Checkpoint ===
            model_registry.register_model(
                experiment_id=exp_id,
                name=f"PPO_Professional_ep{episode}",
                checkpoint_path=str(checkpoint_path),
                architecture_type="ppo",
                state_dim=int(train_env.observation_space.shape[0]),
                action_dim=int(train_env.action_space.n),
                episode_num=int(episode),
                train_metric=float(episode_return),
                is_best=False,
                metadata={
                    "avg_return_last_50": float(np.mean(episode_returns[-50:])),
                    "checkpoint_type": "periodic"
                }
            )

        # Save best model'''

content = content.replace(
    '''        # Save checkpoint every save_freq episodes
        if episode % save_freq == 0:
            checkpoint_path = model_dir / f"episode_{episode}.pth"
            torch.save(agent.policy.state_dict(), checkpoint_path)
            print(f"  [CHECKPOINT] Saved model at episode {episode}")

        # Save best model''',
    checkpoint_reg
)

# 8. Add best model registration
best_model_reg = '''        # Save best model
        if episode_return > best_return:
            best_return = episode_return
            best_model_path = model_dir / "best_model.pth"
            torch.save(agent.policy.state_dict(), best_model_path)
            print(f"  [NEW BEST] Episode {episode}: {episode_return:+.2f}% (saved)")

            # === ML PERSISTENCE: Register Best Model ===
            model_registry.register_model(
                experiment_id=exp_id,
                name="PPO_Professional_BEST",
                checkpoint_path=str(best_model_path),
                architecture_type="ppo",
                state_dim=int(train_env.observation_space.shape[0]),
                action_dim=int(train_env.action_space.n),
                episode_num=int(episode),
                train_metric=float(best_return),
                is_best=True,
                metadata={
                    "achievement": "best_training_return",
                    "episode_achieved": int(episode)
                }
            )

    print()'''

content = content.replace(
    '''        # Save best model
        if episode_return > best_return:
            best_return = episode_return
            best_model_path = model_dir / "best_model.pth"
            torch.save(agent.policy.state_dict(), best_model_path)
            print(f"  [NEW BEST] Episode {episode}: {episode_return:+.2f}% (saved)")

    print()''',
    best_model_reg
)

# 9. Add validation dataset and metrics
val_metrics = '''    print(f"  Return:           {val_return:+.2f}%")
    print()

    # === ML PERSISTENCE: Register Validation Dataset & Log Metrics ===
    val_dataset_id = dataset_mgr.register_dataset(
        name="BTC-USD-Daily",
        version="2022",
        split_type="val",
        start_date="2022-01-01",
        end_date="2022-12-31",
        num_samples=int(len(val_env.market_data)),
        metadata={"data_source": "yfinance", "market_regime": "bear_market"}
    )
    dataset_mgr.link_dataset_to_experiment(exp_id, val_dataset_id, "val")

    tracker.log_metrics_batch({
        "val_return": float(val_return),
        "val_reward": float(val_reward),
        "val_final_value": float(val_final_value)
    }, split="val")
    print("✅ Validation metrics logged to database")
    print()

    # STEP 5: Evaluate on test set (2023-2024)'''

content = content.replace(
    '''    print(f"  Return:           {val_return:+.2f}%")
    print()

    # STEP 5: Evaluate on test set (2023-2024)''',
    val_metrics
)

# 10. Add test dataset and metrics
test_metrics = '''    print(f"  Return:           {test_return:+.2f}%")
    print()

    # === ML PERSISTENCE: Register Test Dataset & Log Metrics ===
    test_dataset_id = dataset_mgr.register_dataset(
        name="BTC-USD-Daily",
        version="2023-2024",
        split_type="test",
        start_date="2023-01-01",
        end_date="2024-12-31",
        num_samples=int(len(test_env.market_data)),
        metadata={"data_source": "yfinance", "market_regime": "recovery_and_recent"}
    )
    dataset_mgr.link_dataset_to_experiment(exp_id, test_dataset_id, "test")

    tracker.log_metrics_batch({
        "test_return": float(test_return),
        "test_reward": float(test_reward),
        "test_final_value": float(test_final_value)
    }, split="test")
    print("✅ Test metrics logged to database")
    print()

    # FINAL SUMMARY'''

content = content.replace(
    '''    print(f"  Return:           {test_return:+.2f}%")
    print()

    # FINAL SUMMARY''',
    test_metrics
)

# 11. Add experiment completion
exp_complete = '''    print("=" * 80)

    # === ML PERSISTENCE: End Experiment ===
    tracker.end_experiment(
        status="completed",
        notes=f"Professional training completed successfully. Train: {best_return:.2f}%, Val: {val_return:.2f}%, Test: {test_return:.2f}%. Total episodes: {num_episodes}"
    )
    print()
    print("=" * 80)
    print("ML EXPERIMENT TRACKING COMPLETE")
    print("=" * 80)
    print(f"  Experiment ID:     {exp_id}")
    print(f"  Status:            Completed")
    print(f"  Git Commit:        Auto-captured")
    print(f"  Database:          ml_experiments.db")
    print()
    print("View this experiment:")
    print(f"  python3 -c \\"from ml_persistence import ExperimentTracker; t=ExperimentTracker(); import pprint; pprint.pprint(t.get_experiment({exp_id}))\\"")
    print("=" * 80)

    return {
        'train_return': best_return,
        'val_return': val_return,
        'test_return': test_return,
        'model_path': str(best_model_path),
        'experiment_id': exp_id
    }'''

content = content.replace(
    '''    print("=" * 80)

    return {
        'train_return': best_return,
        'val_return': val_return,
        'test_return': test_return,
        'model_path': str(best_model_path)
    }''',
    exp_complete
)

# Write the modified content
with open('train_professional.py', 'w') as f:
    f.write(content)

print("✅ Successfully integrated ML persistence tracking into train_professional.py")
print()
print("Changes made:")
print("  1. Added ML persistence imports")
print("  2. Initialize experiment tracking")
print("  3. Registered training dataset")
print("  4. Logged all hyperparameters")
print("  5. Log metrics every episode")
print("  6. Register model checkpoints")
print("  7. Register best models")
print("  8. Registered validation dataset and metrics")
print("  9. Registered test dataset and metrics")
print("  10. End experiment with completion status")

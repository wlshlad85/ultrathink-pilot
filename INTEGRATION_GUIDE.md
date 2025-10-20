# ML Persistence Integration Guide

## Quick Integration for train_professional.py

Follow these steps to add ML tracking to your existing training scripts:

### Step 1: Add Imports (Line 45)

```python
from rl.ppo_agent import PPOAgent
from rl.trading_env import TradingEnv
from backtesting.data_fetcher import DataFetcher

# ADD THESE:
from ml_persistence import ExperimentTracker, ModelRegistry, DatasetManager
```

### Step 2: Initialize Tracker (After Line 94)

After creating directories, add:

```python
# Create directories
model_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

# ADD THIS:
tracker = ExperimentTracker()
exp_id = tracker.start_experiment(
    name=f"PPO_Professional_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    experiment_type="rl",
    description="Professional RL training on BTC-USD (2017-2024)",
    tags=["rl", "ppo", "bitcoin", "professional"],
    random_seed=42
)
print(f"✅ Experiment tracking started (ID: {exp_id})")
```

### Step 3: Log Hyperparameters (After Line 147)

After creating the agent, add:

```python
print(f"  Update epochs:     4")
print()

# ADD THIS:
tracker.log_hyperparameters_batch({
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "eps_clip": 0.2,
    "k_epochs": 4,
    "initial_capital": initial_capital,
    "commission_rate": 0.001,
    "state_dim": train_env.observation_space.shape[0],
    "action_dim": train_env.action_space.n,
    "update_freq": 2048,
    "num_episodes": 1000
})
```

### Step 4: Register Datasets (After Line 121)

After creating train_env, add:

```python
print()

# ADD THIS:
dataset_mgr = DatasetManager()
train_dataset_id = dataset_mgr.register_dataset(
    name="BTC-USD-Daily",
    version="2017-2021",
    split_type="train",
    start_date="2017-01-01",
    end_date="2021-12-31",
    num_samples=len(train_env.market_data)
)
dataset_mgr.link_dataset_to_experiment(exp_id, train_dataset_id, "train")
```

### Step 5: Log Training Metrics (In training loop, after Line 199)

In the training loop, add:

```python
episode_lengths.append(step)

# ADD THIS:
tracker.log_metrics_batch({
    "train_reward": episode_reward,
    "train_return": episode_return,
    "episode_length": step,
    "final_portfolio_value": final_value
}, episode=episode)
```

### Step 6: Register Model Checkpoints (After Line 214)

When saving checkpoints, add:

```python
# Save checkpoint every save_freq episodes
if episode % save_freq == 0:
    checkpoint_path = model_dir / f"episode_{episode}.pth"
    torch.save(agent.policy.state_dict(), checkpoint_path)
    print(f"  [CHECKPOINT] Saved model at episode {episode}")

    # ADD THIS:
    model_registry = ModelRegistry()
    model_registry.register_model(
        experiment_id=exp_id,
        checkpoint_path=str(checkpoint_path),
        architecture_type="ppo",
        state_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.n,
        episode_num=episode,
        train_metric=episode_return,
        is_best=False
    )
```

### Step 7: Register Best Model (After Line 222)

When saving best model, add:

```python
# Save best model
if episode_return > best_return:
    best_return = episode_return
    best_model_path = model_dir / "best_model.pth"
    torch.save(agent.policy.state_dict(), best_model_path)
    print(f"  [NEW BEST] Episode {episode}: {episode_return:+.2f}% (saved)")

    # ADD THIS:
    model_registry = ModelRegistry()
    model_registry.register_model(
        experiment_id=exp_id,
        checkpoint_path=str(best_model_path),
        architecture_type="ppo",
        state_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.n,
        episode_num=episode,
        train_metric=episode_return,
        is_best=True
    )
```

### Step 8: Log Validation Metrics (After Line 290)

After validation evaluation, add:

```python
print(f"  Return:           {val_return:+.2f}%")
print()

# ADD THIS:
tracker.log_metrics_batch({
    "val_return": val_return,
    "val_reward": val_reward,
    "val_final_value": val_final_value
}, split="val")
```

### Step 9: Log Test Metrics (After Line 330)

After test evaluation, add:

```python
print(f"  Return:           {test_return:+.2f}%")
print()

# ADD THIS:
tracker.log_metrics_batch({
    "test_return": test_return,
    "test_reward": test_reward,
    "test_final_value": test_final_value
}, split="test")
```

### Step 10: End Experiment (Before return statement, Line 370)

Before the final return, add:

```python
print("=" * 80)

# ADD THIS:
tracker.end_experiment(
    status="completed",
    notes=f"Professional training completed. Train: {best_return:.2f}%, Val: {val_return:.2f}%, Test: {test_return:.2f}%"
)
print(f"✅ Experiment {exp_id} completed and tracked")
print()

return {
    'train_return': best_return,
    'val_return': val_return,
    'test_return': test_return,
    'model_path': str(best_model_path),
    'experiment_id': exp_id  # ADD THIS TOO
}
```

## Complete Example

See `examples/train_with_ml_tracking.py` for a complete working example that demonstrates all integration points.

## Testing the Integration

Run the example:
```bash
cd ~/ultrathink-pilot
source .venv/bin/activate
python3 examples/train_with_ml_tracking.py
```

## Query Your Tracked Experiments

```bash
# List all experiments
python3 -c "from ml_persistence import ExperimentTracker; t=ExperimentTracker(); print(t.list_experiments())"

# Get specific experiment
python3 -c "from ml_persistence import ExperimentTracker; t=ExperimentTracker(); import pprint; pprint.pprint(t.get_experiment(1))"

# View metrics for an experiment
python3 -c "from ml_persistence import MetricsLogger; m=MetricsLogger(); import pandas as pd; print(pd.DataFrame(m.get_metrics_for_experiment(1)))"
```

## Benefits You Get

1. **Reproducibility**: Automatic git commit tracking
2. **Comparison**: Compare hyperparameters across runs
3. **Model Registry**: Track all checkpoints with metrics
4. **Dataset Versioning**: Know exactly what data was used
5. **Metrics History**: Full time-series of all metrics
6. **SQL Queries**: Powerful analysis with SQL

## Next Steps

1. Apply the changes to `train_professional.py`
2. Run a training session
3. Query the database to see your tracked results
4. Use `quick_dashboard.py` to visualize experiments

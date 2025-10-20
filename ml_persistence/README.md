# ML Database Persistence Skill

A comprehensive system for tracking ML experiments, models, datasets, and metrics with full reproducibility and auditability.

## Features

- **Experiment Tracking**: Complete experiment lifecycle management
- **Model Registry**: Version and track model checkpoints
- **Dataset Management**: Dataset versioning and train/val/test split tracking
- **Metrics Logging**: Time-series metrics with aggregation and analysis
- **Reproducibility**: Git commit tracking, environment capture, random seeds
- **Artifact Management**: Track plots, logs, configs, and reports

## Quick Start

```python
from ml_persistence import ExperimentTracker, ModelRegistry, DatasetManager

# 1. Start an experiment
tracker = ExperimentTracker()
exp_id = tracker.start_experiment(
    name="PPO Bitcoin Trading v1",
    experiment_type="rl",
    description="Training PPO agent on BTC-USD 2023-2024",
    tags=["rl", "bitcoin", "ppo"],
    random_seed=42
)

# 2. Log hyperparameters
tracker.log_hyperparameters_batch({
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "batch_size": 128,
    "hidden_dim": 256
})

# 3. Register datasets
dataset_mgr = DatasetManager()
train_id = dataset_mgr.register_dataset(
    name="BTC-USD-Daily",
    version="2023-2024",
    split_type="train",
    start_date="2023-01-01",
    end_date="2023-12-31",
    num_samples=365
)
dataset_mgr.link_dataset_to_experiment(exp_id, train_id, "train")

# 4. Training loop
for episode in range(100):
    # ... train agent ...

    # Log metrics
    tracker.log_metrics_batch({
        "train_return": episode_return,
        "train_reward": total_reward,
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss
    }, episode=episode)

    # Save checkpoint every 10 episodes
    if episode % 10 == 0:
        model_registry = ModelRegistry()
        model_id = model_registry.register_model(
            experiment_id=exp_id,
            checkpoint_path=f"models/checkpoint_ep{episode}.pth",
            architecture_type="ppo",
            state_dim=43,
            action_dim=3,
            episode_num=episode,
            val_metric=val_sharpe,
            is_best=(val_sharpe > best_sharpe)
        )

# 5. End experiment
tracker.end_experiment(status="completed", notes="Training completed successfully")
```

## Database Schema

### Core Tables

- **experiments**: Training runs with reproducibility metadata
- **models**: Model checkpoints and versioning
- **datasets**: Dataset versions and splits
- **metrics**: Time-series training metrics
- **hyperparameters**: Model and experiment configurations
- **artifacts**: Additional files (plots, logs, etc.)
- **experiment_datasets**: Dataset-experiment associations

### Views

- **experiment_summary**: Quick overview of all experiments
- **best_models**: Best-performing model checkpoints

## Integration with UltraThink RL Training

### Modify `rl/train.py`

```python
from ml_persistence import ExperimentTracker, ModelRegistry, DatasetManager

def train(...):
    # Initialize tracker
    tracker = ExperimentTracker("ml_experiments.db")
    exp_id = tracker.start_experiment(
        name=f"PPO_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        experiment_type="rl",
        tags=["rl", "ppo", symbol.lower()],
        random_seed=42
    )

    # Log hyperparameters
    tracker.log_hyperparameters_batch({
        "learning_rate": agent.lr,
        "gamma": agent.gamma,
        "eps_clip": agent.eps_clip,
        "k_epochs": agent.k_epochs
    })

    # Register datasets
    dataset_mgr = DatasetManager()
    train_id = dataset_mgr.register_dataset(
        name=f"{symbol}-Daily",
        version=f"{start_date}_{end_date}",
        split_type="train",
        start_date=start_date,
        end_date=end_date,
        num_samples=len(env.market_data)
    )
    dataset_mgr.link_dataset_to_experiment(exp_id, train_id, "train")

    # Training loop
    model_registry = ModelRegistry()
    for episode in range(num_episodes):
        # ... existing training code ...

        # Log metrics
        tracker.log_metrics_batch({
            "train_return": final_return,
            "train_reward": episode_reward,
            "episode_length": episode_steps,
            "policy_loss": metrics['policy_loss'],
            "value_loss": metrics['value_loss'],
            "entropy": metrics['entropy']
        }, episode=episode)

        # Save checkpoints
        if episode % save_freq == 0:
            model_id = model_registry.register_model(
                experiment_id=exp_id,
                checkpoint_path=checkpoint_path,
                architecture_type="ppo",
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                episode_num=episode,
                train_metric=final_return,
                is_best=(final_return > best_return)
            )

    # End experiment
    tracker.end_experiment(status="completed")
```

## Querying and Analysis

### List All Experiments

```python
tracker = ExperimentTracker()
experiments = tracker.list_experiments(experiment_type="rl", limit=20)

for exp in experiments:
    print(f"{exp['id']}: {exp['name']}")
    print(f"  Status: {exp['status']}")
    print(f"  Best metric: {exp['best_val_metric']}")
```

### Compare Experiments

```python
from ml_persistence import MetricsLogger

logger = MetricsLogger()
comparison = logger.compare_metrics_across_experiments(
    experiment_ids=[1, 2, 3],
    metric_name="val_sharpe",
    split="val"
)

for exp_id, stats in comparison.items():
    print(f"Experiment {exp_id}: Mean={stats['mean']:.4f}, Max={stats['max']:.4f}")
```

### Get Best Models

```python
registry = ModelRegistry()

# Get best model for an experiment
best = registry.get_best_model(experiment_id=1)
print(f"Best checkpoint: {best['checkpoint_path']}")
print(f"Val metric: {best['val_metric']}")

# Compare all models
models = registry.list_models(is_best=True, order_by="val_metric", limit=10)
for model in models:
    print(f"{model['id']}: {model['name']} - {model['val_metric']:.4f}")
```

## Command-Line Tools

### Initialize Database

```bash
python3 -m ml_persistence.core
```

### Query Experiments

```bash
python3 -c "
from ml_persistence import ExperimentTracker
tracker = ExperimentTracker()
for exp in tracker.list_experiments(limit=10):
    print(f\"{exp['id']}: {exp['name']} - {exp['status']}\")
"
```

## Best Practices

1. **Always track git commit**: Enables perfect reproducibility
2. **Use semantic versioning**: Version both models and datasets
3. **Log frequently**: Capture metrics at every important step
4. **Tag experiments**: Use tags for easy filtering and organization
5. **Link datasets**: Always link datasets to experiments for traceability
6. **Mark best models**: Clearly identify best-performing checkpoints
7. **Add metadata**: Store extra context in metadata fields
8. **Archive artifacts**: Track all generated plots, logs, and reports

## Migration from Legacy `experiments.db`

If you have data in the old `experiments.db` format:

```python
# TODO: Create migration script
# Old schema -> New schema mapping
# - experiments -> experiments (with extensions)
# - episodes -> metrics (with restructuring)
# - regime_analysis -> artifacts or custom table
```

## Database Maintenance

### Vacuum Database

```bash
sqlite3 ml_experiments.db "VACUUM;"
```

### Backup Database

```bash
cp ml_experiments.db ml_experiments_backup_$(date +%Y%m%d).db
```

### Export to CSV

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect("ml_experiments.db")
df = pd.read_sql_query("SELECT * FROM experiment_summary", conn)
df.to_csv("experiments.csv", index=False)
```

## Advanced Features

### Custom Metrics

```python
# Log complex metrics (distributions, histograms, etc.)
import json

tracker.log_metric(
    metric_name="action_distribution",
    value=0.0,  # Not used for complex metrics
    value_json=json.dumps({
        "hold": 0.5,
        "buy": 0.3,
        "sell": 0.2
    }),
    episode=episode
)
```

### Regime-Specific Tracking

```python
# Track performance by market regime
for regime in ['bull', 'bear', 'sideways']:
    tracker.log_metric(
        f"return_{regime}",
        returns[regime],
        episode=episode,
        metadata={"regime": regime}
    )
```

### Multi-Asset Tracking

```python
# Track experiments across multiple assets
for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
    exp_id = tracker.start_experiment(
        name=f"PPO_{symbol}",
        tags=["multi-asset", symbol],
        metadata={"asset": symbol}
    )
    # ... train ...
```

## License

MIT (same as UltraThink project)

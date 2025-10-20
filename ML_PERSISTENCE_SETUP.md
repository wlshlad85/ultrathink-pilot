# ML Persistence Setup Guide

## Overview

The `ml_persistence` module is a comprehensive system for tracking ML experiments, models, datasets, and metrics. This guide will help you set it up and start using it.

## What's Included

The module has already been created with the following components:

- **ExperimentTracker**: Track training runs and experiments
- **ModelRegistry**: Version and manage model checkpoints  
- **DatasetManager**: Track dataset versions and splits
- **MetricsLogger**: Log training metrics and performance data
- **MLDatabase**: Core database infrastructure

## Setup Steps

### 1. Install Dependencies

Make sure all Python dependencies are installed. Run in WSL:

```bash
cd /home/rich/ultrathink-pilot
pip3 install -r requirements.txt
```

The key dependencies are:
- numpy (for metrics calculations)
- pandas (for data analysis)
- torch (for RL training)
- gymnasium (for RL environments)

### 2. Initialize Database

You have two options:

**Option A: Run the setup script**
```bash
chmod +x setup_ml_persistence.sh
./setup_ml_persistence.sh
```

**Option B: Initialize manually**
```bash
python3 -m ml_persistence.core
```

This will create `ml_experiments.db` with all necessary tables and views.

### 3. Verify Setup

Run the verification script:

```bash
python3 verify_ml_persistence.py
```

This will:
- Check if all imports work
- Verify database schema
- Test basic operations
- Create a test experiment

## Quick Start Example

Once set up, you can use the module like this:

```python
from ml_persistence import ExperimentTracker, ModelRegistry, DatasetManager

# Start an experiment
tracker = ExperimentTracker()
exp_id = tracker.start_experiment(
    name="PPO Bitcoin Trading",
    experiment_type="rl",
    description="Training PPO agent on BTC-USD",
    tags=["rl", "bitcoin", "ppo"],
    random_seed=42
)

# Log hyperparameters
tracker.log_hyperparameters_batch({
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "batch_size": 128
})

# During training - log metrics
tracker.log_metrics_batch({
    "train_return": episode_return,
    "train_reward": total_reward,
    "policy_loss": policy_loss
}, episode=episode_num)

# Register a model checkpoint
registry = ModelRegistry()
model_id = registry.register_model(
    experiment_id=exp_id,
    checkpoint_path="models/checkpoint_ep100.pth",
    architecture_type="ppo",
    state_dim=43,
    action_dim=3,
    episode_num=100,
    val_metric=sharpe_ratio,
    is_best=True
)

# End experiment
tracker.end_experiment(status="completed")
```

## Database Schema

The database includes these main tables:

- **experiments**: Training runs with metadata and timing
- **models**: Model checkpoints and performance metrics
- **datasets**: Dataset versions and splits
- **metrics**: Time-series training/validation metrics
- **hyperparameters**: Model configurations
- **artifacts**: Additional files (plots, logs, etc.)
- **experiment_datasets**: Links between datasets and experiments

## Useful Views

Two views are automatically created:

- **experiment_summary**: Overview of all experiments with key metrics
- **best_models**: Best-performing model checkpoints

Query them like regular tables:

```python
import sqlite3
conn = sqlite3.connect("ml_experiments.db")
df = pd.read_sql_query("SELECT * FROM experiment_summary", conn)
```

## Integration with Existing Training

To integrate with your existing RL training scripts (e.g., `train_professional.py`), add:

```python
from ml_persistence import ExperimentTracker, ModelRegistry

# At the start of training
tracker = ExperimentTracker()
exp_id = tracker.start_experiment(
    name=f"PPO_BTC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    experiment_type="rl"
)

# Inside your training loop
tracker.log_metrics_batch({
    "train_return": episode_return,
    # ... other metrics
}, episode=episode)

# When saving checkpoints
registry = ModelRegistry()
model_id = registry.register_model(
    experiment_id=exp_id,
    checkpoint_path=checkpoint_path,
    # ... other params
)

# At the end
tracker.end_experiment(status="completed")
```

## Troubleshooting

### Import Error: No module named 'numpy'

Make sure you've installed requirements:
```bash
pip3 install -r requirements.txt
```

### Database not found

Initialize the database:
```bash
python3 -m ml_persistence.core
```

### Permission denied on setup script

Make it executable:
```bash
chmod +x setup_ml_persistence.sh
```

## Next Steps

1. Run the setup script: `./setup_ml_persistence.sh`
2. Verify with: `python3 verify_ml_persistence.py`
3. Read the full documentation: `ml_persistence/README.md`
4. Integrate with your training scripts

## Files Created

- `ml_experiments.db` - SQLite database with all experiment data
- `setup_ml_persistence.sh` - Setup script
- `verify_ml_persistence.py` - Verification script
- `ML_PERSISTENCE_SETUP.md` - This guide

For more detailed examples and advanced features, see `ml_persistence/README.md`.


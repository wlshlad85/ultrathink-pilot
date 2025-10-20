#!/usr/bin/env python3
"""
Model registry for tracking and versioning ML model checkpoints.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from .core import MLDatabase


class ModelRegistry:
    """
    Registry for ML model checkpoints.

    Handles:
    - Model versioning and storage
    - Checkpoint metadata
    - Best model tracking
    - Model comparison and selection
    """

    def __init__(self, db_path: str = "ml_experiments.db"):
        self.db = MLDatabase(db_path)

    def register_model(
        self,
        experiment_id: int,
        checkpoint_path: str,
        name: str = None,
        version: str = None,
        architecture_type: str = None,
        state_dim: int = None,
        action_dim: int = None,
        episode_num: int = None,
        global_step: int = None,
        train_metric: float = None,
        val_metric: float = None,
        test_metric: float = None,
        is_best: bool = False,
        hyperparameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Register a model checkpoint.

        Args:
            experiment_id: Parent experiment ID
            checkpoint_path: Path to model checkpoint file
            name: Model name (defaults to experiment name + version)
            version: Model version string
            architecture_type: Architecture type ('ppo', 'dqn', etc.)
            state_dim: State space dimension
            action_dim: Action space dimension
            episode_num: Episode number when checkpoint was saved
            global_step: Global training step
            train_metric: Training metric value
            val_metric: Validation metric value
            test_metric: Test metric value
            is_best: Whether this is the best model so far
            hyperparameters: Model hyperparameters
            metadata: Additional metadata

        Returns:
            Model ID
        """
        # Calculate checkpoint size
        path = Path(checkpoint_path)
        checkpoint_size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else None

        # Convert to JSON
        hyperparams_json = json.dumps(hyperparameters) if hyperparameters else None
        metadata_json = json.dumps(metadata) if metadata else None

        # If this is the best model, unmark previous best models for this experiment
        if is_best:
            self.db.execute_update(
                "UPDATE models SET is_best = 0 WHERE experiment_id = ?",
                (experiment_id,)
            )

        query = """
            INSERT INTO models (
                experiment_id, name, version, architecture_type,
                state_dim, action_dim, checkpoint_path, checkpoint_size_mb,
                episode_num, global_step,
                train_metric, val_metric, test_metric, is_best,
                hyperparameters, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        model_id = self.db.execute_update(query, (
            experiment_id, name, version, architecture_type,
            state_dim, action_dim, str(checkpoint_path), checkpoint_size_mb,
            episode_num, global_step,
            train_metric, val_metric, test_metric, is_best,
            hyperparams_json, metadata_json
        ))

        status_str = "✨ BEST" if is_best else "✓"
        print(f"{status_str} Registered model ID {model_id}")
        if episode_num is not None:
            print(f"   Episode: {episode_num}")
        if val_metric is not None:
            print(f"   Val metric: {val_metric:.4f}")

        return model_id

    def get_best_model(self, experiment_id: int) -> Optional[Dict]:
        """
        Get the best model for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Model info dict or None
        """
        results = self.db.execute_query(
            "SELECT * FROM models WHERE experiment_id = ? AND is_best = 1",
            (experiment_id,)
        )

        if not results:
            return None

        return dict(results[0])

    def get_model(self, model_id: int) -> Optional[Dict]:
        """
        Get model by ID.

        Args:
            model_id: Model ID

        Returns:
            Model info dict or None
        """
        results = self.db.execute_query(
            "SELECT * FROM models WHERE id = ?",
            (model_id,)
        )

        if not results:
            return None

        return dict(results[0])

    def list_models(
        self,
        experiment_id: int = None,
        architecture_type: str = None,
        is_best: bool = None,
        order_by: str = "val_metric",
        ascending: bool = False,
        limit: int = 50
    ):
        """
        List models with optional filters.

        Args:
            experiment_id: Filter by experiment
            architecture_type: Filter by architecture
            is_best: Filter by best models only
            order_by: Sort by this column
            ascending: Sort order
            limit: Max results

        Returns:
            List of model dicts
        """
        query = "SELECT * FROM models WHERE 1=1"
        params = []

        if experiment_id is not None:
            query += " AND experiment_id = ?"
            params.append(experiment_id)

        if architecture_type:
            query += " AND architecture_type = ?"
            params.append(architecture_type)

        if is_best is not None:
            query += " AND is_best = ?"
            params.append(1 if is_best else 0)

        order_dir = "ASC" if ascending else "DESC"
        query += f" ORDER BY {order_by} {order_dir} LIMIT ?"
        params.append(limit)

        results = self.db.execute_query(query, tuple(params))
        return [dict(row) for row in results]

    def compare_models(self, model_ids: list) -> Dict:
        """
        Compare multiple models.

        Args:
            model_ids: List of model IDs to compare

        Returns:
            Comparison dict with model metrics
        """
        models = []
        for model_id in model_ids:
            model = self.get_model(model_id)
            if model:
                models.append(model)

        if not models:
            return {}

        return {
            "models": models,
            "best_train": max(models, key=lambda m: m['train_metric'] or -float('inf')),
            "best_val": max(models, key=lambda m: m['val_metric'] or -float('inf')),
            "best_test": max(models, key=lambda m: m['test_metric'] or -float('inf')),
        }

    def mark_best_model(self, model_id: int):
        """
        Mark a model as the best for its experiment.

        Args:
            model_id: Model ID to mark as best
        """
        # Get experiment ID
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        experiment_id = model['experiment_id']

        # Unmark previous best
        self.db.execute_update(
            "UPDATE models SET is_best = 0 WHERE experiment_id = ?",
            (experiment_id,)
        )

        # Mark new best
        self.db.execute_update(
            "UPDATE models SET is_best = 1 WHERE id = ?",
            (model_id,)
        )

        print(f"✨ Model {model_id} marked as best for experiment {experiment_id}")


if __name__ == "__main__":
    # Demo usage
    from .experiment_tracker import ExperimentTracker

    # Create a test experiment
    tracker = ExperimentTracker("test_ml_experiments.db")
    exp_id = tracker.start_experiment(
        name="Model Registry Test",
        experiment_type="rl"
    )

    # Register models
    registry = ModelRegistry("test_ml_experiments.db")

    for ep in [10, 20, 30]:
        model_id = registry.register_model(
            experiment_id=exp_id,
            checkpoint_path=f"models/checkpoint_ep{ep}.pth",
            architecture_type="ppo",
            state_dim=43,
            action_dim=3,
            episode_num=ep,
            val_metric=0.5 + ep * 0.01,
            is_best=(ep == 30)
        )

    # Get best model
    best = registry.get_best_model(exp_id)
    print(f"\nBest model: {best}")

    # List all models for experiment
    print("\nAll models:")
    for model in registry.list_models(experiment_id=exp_id):
        print(f"  {model['id']}: Episode {model['episode_num']}, Val: {model['val_metric']}")

    tracker.end_experiment()

#!/usr/bin/env python3
"""
Experiment tracking interface for ML experiments.
"""

from typing import Optional, Dict, List, Any
from datetime import datetime
import json
import subprocess
from pathlib import Path
import platform

from .core import MLDatabase


class ExperimentTracker:
    """
    High-level interface for tracking ML experiments.

    Usage:
        tracker = ExperimentTracker()
        exp_id = tracker.start_experiment(
            name="PPO Bitcoin Trading",
            experiment_type="rl",
            description="Training PPO agent on BTC-USD 2023-2024"
        )
        tracker.log_metric("train_return", 5.2, episode=10)
        tracker.end_experiment()
    """

    def __init__(self, db_path: str = "ml_experiments.db"):
        self.db = MLDatabase(db_path)
        self.experiment_id: Optional[int] = None
        self.experiment_name: Optional[str] = None

    def start_experiment(
        self,
        name: str,
        experiment_type: str = "rl",
        description: str = "",
        tags: List[str] = None,
        random_seed: int = None,
        capture_git: bool = True,
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Start a new experiment.

        Args:
            name: Experiment name (should be unique and descriptive)
            experiment_type: Type of experiment ('rl', 'supervised', 'backtest', etc.)
            description: Detailed description
            tags: List of tags for organization
            random_seed: Random seed for reproducibility
            capture_git: Whether to capture git commit/branch
            metadata: Additional metadata as dict

        Returns:
            Experiment ID
        """
        # Capture reproducibility info
        git_commit = None
        git_branch = None

        if capture_git:
            try:
                git_commit = subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'],
                    stderr=subprocess.DEVNULL
                ).decode().strip()

                git_branch = subprocess.check_output(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except Exception:
                pass  # Not a git repo or git not available

        python_version = platform.python_version()

        # Convert tags and metadata to JSON
        tags_json = json.dumps(tags) if tags else None
        metadata_json = json.dumps(metadata) if metadata else None

        # Insert experiment
        query = """
            INSERT INTO experiments (
                name, description, experiment_type, status,
                git_commit, git_branch, python_version, random_seed,
                start_time, tags, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        self.experiment_id = self.db.execute_update(query, (
            name, description, experiment_type, 'running',
            git_commit, git_branch, python_version, random_seed,
            datetime.now(), tags_json, metadata_json
        ))

        self.experiment_name = name

        print(f"✅ Started experiment '{name}' (ID: {self.experiment_id})")
        if git_commit:
            print(f"   Git: {git_branch}@{git_commit[:8]}")
        if random_seed is not None:
            print(f"   Seed: {random_seed}")

        return self.experiment_id

    def log_metric(
        self,
        metric_name: str,
        value: float,
        episode: int = None,
        step: int = None,
        epoch: int = None,
        split: str = "train",
        phase: str = "training",
        model_id: int = None
    ):
        """
        Log a scalar metric value.

        Args:
            metric_name: Name of the metric (e.g., 'train_return', 'val_sharpe')
            value: Metric value
            episode: Episode number (for RL)
            step: Global step number
            epoch: Epoch number (for supervised learning)
            split: Data split ('train', 'val', 'test')
            phase: Training phase ('training', 'evaluation', 'inference')
            model_id: Optional model ID if metric relates to specific checkpoint
        """
        if self.experiment_id is None:
            raise ValueError("No active experiment. Call start_experiment() first.")

        query = """
            INSERT INTO metrics (
                experiment_id, model_id, metric_name, metric_type,
                episode, step, epoch, value, split, phase
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        self.db.execute_update(query, (
            self.experiment_id, model_id, metric_name, 'scalar',
            episode, step, epoch, value, split, phase
        ))

    def log_metrics_batch(
        self,
        metrics: Dict[str, float],
        episode: int = None,
        step: int = None,
        epoch: int = None,
        split: str = "train",
        phase: str = "training"
    ):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dict of metric_name -> value
            episode: Episode number
            step: Global step
            epoch: Epoch number
            split: Data split
            phase: Training phase
        """
        for name, value in metrics.items():
            self.log_metric(name, value, episode, step, epoch, split, phase)

    def log_hyperparameter(
        self,
        param_name: str,
        param_value: Any,
        param_type: str = None,
        model_id: int = None
    ):
        """
        Log a hyperparameter.

        Args:
            param_name: Parameter name (e.g., 'learning_rate', 'gamma')
            param_value: Parameter value
            param_type: Type hint ('float', 'int', 'str', 'bool', 'json')
            model_id: Optional model ID if param is model-specific
        """
        if self.experiment_id is None:
            raise ValueError("No active experiment.")

        # Auto-detect type if not provided
        if param_type is None:
            if isinstance(param_value, float):
                param_type = 'float'
            elif isinstance(param_value, int):
                param_type = 'int'
            elif isinstance(param_value, bool):
                param_type = 'bool'
            elif isinstance(param_value, str):
                param_type = 'str'
            else:
                param_type = 'json'
                param_value = json.dumps(param_value)

        query = """
            INSERT INTO hyperparameters (
                experiment_id, model_id, param_name, param_value, param_type
            ) VALUES (?, ?, ?, ?, ?)
        """

        self.db.execute_update(query, (
            self.experiment_id, model_id, param_name, str(param_value), param_type
        ))

    def log_hyperparameters_batch(self, hyperparams: Dict[str, Any], model_id: int = None):
        """Log multiple hyperparameters at once."""
        for name, value in hyperparams.items():
            self.log_hyperparameter(name, value, model_id=model_id)

    def add_artifact(
        self,
        artifact_type: str,
        artifact_path: str,
        artifact_name: str = None,
        description: str = None,
        metadata: Dict[str, Any] = None,
        model_id: int = None
    ) -> int:
        """
        Register an artifact (plot, log file, config, etc.).

        Args:
            artifact_type: Type of artifact ('plot', 'log', 'config', 'report', etc.)
            artifact_path: Path to the artifact file
            artifact_name: Optional display name
            description: Optional description
            metadata: Additional metadata
            model_id: Optional model ID if artifact relates to specific checkpoint

        Returns:
            Artifact ID
        """
        if self.experiment_id is None:
            raise ValueError("No active experiment.")

        path = Path(artifact_path)
        file_size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else None
        metadata_json = json.dumps(metadata) if metadata else None

        query = """
            INSERT INTO artifacts (
                experiment_id, model_id, artifact_type, artifact_path,
                artifact_name, file_size_mb, description, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        return self.db.execute_update(query, (
            self.experiment_id, model_id, artifact_type, str(artifact_path),
            artifact_name, file_size_mb, description, metadata_json
        ))

    def end_experiment(
        self,
        status: str = "completed",
        notes: str = None
    ):
        """
        Mark experiment as finished.

        Args:
            status: Final status ('completed', 'failed', 'stopped')
            notes: Optional final notes
        """
        if self.experiment_id is None:
            raise ValueError("No active experiment.")

        # Calculate duration
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT start_time FROM experiments WHERE id = ?",
            (self.experiment_id,)
        )
        row = cursor.fetchone()
        start_time = datetime.fromisoformat(row['start_time'])
        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()
        conn.close()

        query = """
            UPDATE experiments SET
                status = ?,
                end_time = ?,
                duration_seconds = ?,
                notes = ?,
                updated_at = ?
            WHERE id = ?
        """

        self.db.execute_update(query, (
            status, end_time, duration_seconds, notes,
            datetime.now(), self.experiment_id
        ))

        print(f"✅ Experiment {self.experiment_id} ended: {status}")
        print(f"   Duration: {duration_seconds/3600:.2f} hours")

        self.experiment_id = None
        self.experiment_name = None

    def update_experiment_metadata(self, metadata: Dict[str, Any]):
        """Update experiment metadata."""
        if self.experiment_id is None:
            raise ValueError("No active experiment.")

        metadata_json = json.dumps(metadata)

        query = """
            UPDATE experiments SET
                metadata = ?,
                updated_at = ?
            WHERE id = ?
        """

        self.db.execute_update(query, (
            metadata_json, datetime.now(), self.experiment_id
        ))

    def get_experiment_summary(self, experiment_id: int = None) -> Dict:
        """
        Get summary of an experiment.

        Args:
            experiment_id: Experiment ID (defaults to current experiment)

        Returns:
            Dict with experiment summary
        """
        if experiment_id is None:
            experiment_id = self.experiment_id

        if experiment_id is None:
            raise ValueError("No experiment ID provided.")

        results = self.db.execute_query(
            "SELECT * FROM experiment_summary WHERE id = ?",
            (experiment_id,)
        )

        if not results:
            return None

        return dict(results[0])

    def list_experiments(
        self,
        experiment_type: str = None,
        status: str = None,
        tags: List[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        List experiments with optional filters.

        Args:
            experiment_type: Filter by experiment type
            status: Filter by status
            tags: Filter by tags (match any)
            limit: Maximum number of results

        Returns:
            List of experiment summaries
        """
        query = "SELECT * FROM experiment_summary WHERE 1=1"
        params = []

        if experiment_type:
            query += " AND experiment_type = ?"
            params.append(experiment_type)

        if status:
            query += " AND status = ?"
            params.append(status)

        # TODO: Add tag filtering (requires JSON parsing in SQLite)

        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        results = self.db.execute_query(query, tuple(params))
        return [dict(row) for row in results]


if __name__ == "__main__":
    # Demo usage
    tracker = ExperimentTracker("test_ml_experiments.db")

    # Start experiment
    exp_id = tracker.start_experiment(
        name="Test RL Experiment",
        experiment_type="rl",
        description="Testing the ML persistence system",
        tags=["test", "rl", "bitcoin"],
        random_seed=42
    )

    # Log hyperparameters
    tracker.log_hyperparameters_batch({
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "hidden_dim": 256
    })

    # Log metrics
    for ep in range(1, 11):
        tracker.log_metric("train_return", ep * 0.5, episode=ep)
        tracker.log_metric("val_sharpe", 0.3 + ep * 0.01, episode=ep, split="val")

    # End experiment
    tracker.end_experiment(status="completed", notes="Test successful")

    # List experiments
    print("\n" + "="*80)
    print("EXPERIMENTS:")
    print("="*80)
    for exp in tracker.list_experiments():
        print(f"\n#{exp['id']}: {exp['name']}")
        print(f"  Type: {exp['experiment_type']}")
        print(f"  Status: {exp['status']}")
        print(f"  Checkpoints: {exp['num_checkpoints']}")

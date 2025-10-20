#!/usr/bin/env python3
"""
Core ML Database infrastructure.
"""

import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class MLDatabase:
    """
    Central database for ML experiment tracking.

    Provides a unified schema for:
    - Experiments (training runs)
    - Models (checkpoints and metadata)
    - Datasets (versions and splits)
    - Metrics (training/validation/test)
    - Hyperparameters
    - Artifacts (plots, logs, etc.)
    """

    def __init__(self, db_path: str = "ml_experiments.db"):
        self.db_path = Path(db_path)
        self._initialize_schema()

    def _initialize_schema(self):
        """Create all tables and indices."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Experiments table - tracks training runs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                experiment_type TEXT,  -- 'rl', 'supervised', 'backtest', etc.
                status TEXT DEFAULT 'running',  -- 'running', 'completed', 'failed', 'stopped'

                -- Reproducibility
                git_commit TEXT,
                git_branch TEXT,
                python_version TEXT,
                random_seed INTEGER,

                -- Timing
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration_seconds REAL,

                -- Tags for organization
                tags TEXT,  -- JSON array

                -- Notes and metadata
                notes TEXT,
                metadata TEXT,  -- JSON for arbitrary key-value pairs

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Models table - tracks model checkpoints
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                version TEXT,

                -- Model architecture
                architecture_type TEXT,  -- 'ppo', 'dqn', 'lstm', etc.
                state_dim INTEGER,
                action_dim INTEGER,

                -- Checkpoint info
                checkpoint_path TEXT NOT NULL,
                checkpoint_size_mb REAL,
                episode_num INTEGER,
                global_step INTEGER,

                -- Performance snapshot
                train_metric REAL,  -- e.g., train return
                val_metric REAL,    -- e.g., val sharpe
                test_metric REAL,   -- e.g., test return
                is_best BOOLEAN DEFAULT 0,

                -- Metadata
                hyperparameters TEXT,  -- JSON
                metadata TEXT,  -- JSON for arbitrary info

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)

        # Datasets table - tracks dataset versions and splits
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT,
                dataset_type TEXT,  -- 'timeseries', 'tabular', 'image', etc.

                -- Data location
                data_path TEXT,
                data_hash TEXT,  -- For version control

                -- Split info
                split_type TEXT,  -- 'train', 'val', 'test', 'train_val', etc.
                num_samples INTEGER,
                start_date TEXT,
                end_date TEXT,

                -- Features
                feature_columns TEXT,  -- JSON array
                target_columns TEXT,   -- JSON array

                -- Metadata
                metadata TEXT,  -- JSON

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(name, version, split_type)
            )
        """)

        # Metrics table - detailed time-series metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                model_id INTEGER,  -- Optional: link to specific checkpoint

                -- Metric identification
                metric_name TEXT NOT NULL,  -- 'train_return', 'val_sharpe', etc.
                metric_type TEXT,  -- 'scalar', 'distribution', 'image', etc.

                -- Tracking position
                episode INTEGER,
                step INTEGER,
                epoch INTEGER,

                -- Value
                value REAL,
                value_json TEXT,  -- For complex metrics (distributions, etc.)

                -- Context
                split TEXT,  -- 'train', 'val', 'test'
                phase TEXT,  -- 'training', 'evaluation', 'inference'

                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)

        # Hyperparameters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hyperparameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                model_id INTEGER,

                param_name TEXT NOT NULL,
                param_value TEXT NOT NULL,  -- Store as string, parse as needed
                param_type TEXT,  -- 'float', 'int', 'str', 'bool', 'json'

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)

        # Artifacts table - track additional files (plots, logs, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                model_id INTEGER,

                artifact_type TEXT NOT NULL,  -- 'plot', 'log', 'config', 'report', etc.
                artifact_path TEXT NOT NULL,
                artifact_name TEXT,
                file_size_mb REAL,

                description TEXT,
                metadata TEXT,  -- JSON

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)

        # Relationships table - track dataset-experiment associations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_datasets (
                experiment_id INTEGER NOT NULL,
                dataset_id INTEGER NOT NULL,
                usage_type TEXT,  -- 'train', 'val', 'test'

                FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                FOREIGN KEY (dataset_id) REFERENCES datasets(id),
                PRIMARY KEY (experiment_id, dataset_id, usage_type)
            )
        """)

        # Create indices for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_type ON experiments(experiment_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_exp ON models(experiment_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_best ON models(is_best)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_exp ON metrics(experiment_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_exp ON artifacts(experiment_id)")

        # Create useful views
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS experiment_summary AS
            SELECT
                e.id,
                e.name,
                e.experiment_type,
                e.status,
                e.start_time,
                e.end_time,
                e.duration_seconds,
                COUNT(DISTINCT m.id) as num_checkpoints,
                MAX(CASE WHEN m.is_best = 1 THEN m.val_metric END) as best_val_metric,
                MAX(CASE WHEN m.is_best = 1 THEN m.test_metric END) as best_test_metric,
                e.tags,
                e.notes
            FROM experiments e
            LEFT JOIN models m ON e.id = m.experiment_id
            GROUP BY e.id
            ORDER BY e.start_time DESC
        """)

        cursor.execute("""
            CREATE VIEW IF NOT EXISTS best_models AS
            SELECT
                m.*,
                e.name as experiment_name,
                e.experiment_type
            FROM models m
            INNER JOIN experiments e ON m.experiment_id = e.id
            WHERE m.is_best = 1
            ORDER BY m.val_metric DESC
        """)

        conn.commit()
        conn.close()

    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn

    def execute_query(self, query: str, params: tuple = ()):
        """Execute a query and return results."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        return results

    def execute_update(self, query: str, params: tuple = ()):
        """Execute an update/insert and commit."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        last_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return last_id


if __name__ == "__main__":
    # Initialize database
    db = MLDatabase("ml_experiments.db")
    print(f"✅ ML Database initialized: {db.db_path.absolute()}")
    print()
    print("Schema created:")
    print("  - experiments: Track training runs")
    print("  - models: Model checkpoints and versions")
    print("  - datasets: Dataset versions and splits")
    print("  - metrics: Time-series training metrics")
    print("  - hyperparameters: Model and experiment configs")
    print("  - artifacts: Additional files (plots, logs, etc.)")
    print("  - experiment_datasets: Dataset-experiment associations")
    print()
    print("Views created:")
    print("  - experiment_summary: Overview of all experiments")
    print("  - best_models: Best-performing model checkpoints")

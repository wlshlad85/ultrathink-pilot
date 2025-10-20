#!/usr/bin/env python3
"""
Dataset version management and tracking.
"""

from typing import Optional, Dict, List, Any
from pathlib import Path
import json
import hashlib

from .core import MLDatabase


class DatasetManager:
    """
    Manager for dataset versions and splits.

    Handles:
    - Dataset versioning and hashing
    - Train/val/test split tracking
    - Dataset-experiment associations
    """

    def __init__(self, db_path: str = "ml_experiments.db"):
        self.db = MLDatabase(db_path)

    def register_dataset(
        self,
        name: str,
        version: str,
        split_type: str,
        dataset_type: str = "timeseries",
        data_path: str = None,
        num_samples: int = None,
        start_date: str = None,
        end_date: str = None,
        feature_columns: List[str] = None,
        target_columns: List[str] = None,
        compute_hash: bool = True,
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Register a dataset version.

        Args:
            name: Dataset name (e.g., 'BTC-USD-Daily')
            version: Version string (e.g., '2023-01-01_2024-01-01')
            split_type: Split type ('train', 'val', 'test', etc.)
            dataset_type: Type of dataset ('timeseries', 'tabular', etc.)
            data_path: Path to dataset file
            num_samples: Number of samples in this split
            start_date: Start date (for time-series)
            end_date: End date (for time-series)
            feature_columns: List of feature column names
            target_columns: List of target column names
            compute_hash: Whether to compute data hash for versioning
            metadata: Additional metadata

        Returns:
            Dataset ID
        """
        # Compute hash if requested
        data_hash = None
        if compute_hash and data_path and Path(data_path).exists():
            data_hash = self._compute_file_hash(data_path)

        # Convert to JSON
        feature_columns_json = json.dumps(feature_columns) if feature_columns else None
        target_columns_json = json.dumps(target_columns) if target_columns else None
        metadata_json = json.dumps(metadata) if metadata else None

        query = """
            INSERT OR REPLACE INTO datasets (
                name, version, dataset_type, split_type,
                data_path, data_hash, num_samples,
                start_date, end_date,
                feature_columns, target_columns,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        dataset_id = self.db.execute_update(query, (
            name, version, dataset_type, split_type,
            str(data_path) if data_path else None,
            data_hash, num_samples,
            start_date, end_date,
            feature_columns_json, target_columns_json,
            metadata_json
        ))

        print(f"✅ Registered dataset '{name}' v{version} ({split_type}) - ID: {dataset_id}")
        if start_date and end_date:
            print(f"   Date range: {start_date} to {end_date}")
        if num_samples:
            print(f"   Samples: {num_samples}")

        return dataset_id

    def link_dataset_to_experiment(
        self,
        experiment_id: int,
        dataset_id: int,
        usage_type: str
    ):
        """
        Link a dataset to an experiment.

        Args:
            experiment_id: Experiment ID
            dataset_id: Dataset ID
            usage_type: How dataset is used ('train', 'val', 'test')
        """
        query = """
            INSERT OR REPLACE INTO experiment_datasets (
                experiment_id, dataset_id, usage_type
            ) VALUES (?, ?, ?)
        """

        self.db.execute_update(query, (experiment_id, dataset_id, usage_type))

    def get_experiment_datasets(self, experiment_id: int) -> Dict[str, Dict]:
        """
        Get all datasets for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dict with keys 'train', 'val', 'test' mapping to dataset info
        """
        query = """
            SELECT d.*, ed.usage_type
            FROM datasets d
            JOIN experiment_datasets ed ON d.id = ed.dataset_id
            WHERE ed.experiment_id = ?
        """

        results = self.db.execute_query(query, (experiment_id,))

        datasets = {}
        for row in results:
            row_dict = dict(row)
            usage_type = row_dict.pop('usage_type')
            datasets[usage_type] = row_dict

        return datasets

    def get_dataset(
        self,
        name: str = None,
        version: str = None,
        split_type: str = None,
        dataset_id: int = None
    ) -> Optional[Dict]:
        """
        Get dataset by name/version/split or by ID.

        Args:
            name: Dataset name
            version: Version string
            split_type: Split type
            dataset_id: Dataset ID (alternative to name/version/split)

        Returns:
            Dataset info dict or None
        """
        if dataset_id is not None:
            results = self.db.execute_query(
                "SELECT * FROM datasets WHERE id = ?",
                (dataset_id,)
            )
        elif name and version and split_type:
            results = self.db.execute_query(
                "SELECT * FROM datasets WHERE name = ? AND version = ? AND split_type = ?",
                (name, version, split_type)
            )
        else:
            raise ValueError("Must provide either dataset_id or (name, version, split_type)")

        if not results:
            return None

        return dict(results[0])

    def _compute_file_hash(self, filepath: str, algorithm: str = 'sha256') -> str:
        """Compute hash of a file for versioning."""
        hasher = hashlib.new(algorithm)

        with open(filepath, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)

        return hasher.hexdigest()


if __name__ == "__main__":
    # Demo usage
    from .experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker("test_ml_experiments.db")
    exp_id = tracker.start_experiment(
        name="Dataset Manager Test",
        experiment_type="rl"
    )

    manager = DatasetManager("test_ml_experiments.db")

    # Register datasets
    train_id = manager.register_dataset(
        name="BTC-USD-Daily",
        version="2023-2024",
        split_type="train",
        dataset_type="timeseries",
        start_date="2023-01-01",
        end_date="2023-12-31",
        num_samples=365
    )

    val_id = manager.register_dataset(
        name="BTC-USD-Daily",
        version="2023-2024",
        split_type="val",
        dataset_type="timeseries",
        start_date="2024-01-01",
        end_date="2024-06-30",
        num_samples=182
    )

    # Link to experiment
    manager.link_dataset_to_experiment(exp_id, train_id, "train")
    manager.link_dataset_to_experiment(exp_id, val_id, "val")

    # Get datasets for experiment
    datasets = manager.get_experiment_datasets(exp_id)
    print("\nDatasets for experiment:")
    for usage, ds_info in datasets.items():
        print(f"  {usage}: {ds_info['name']} ({ds_info['num_samples']} samples)")

    tracker.end_experiment()

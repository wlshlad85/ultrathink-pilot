#!/usr/bin/env python3
"""
MLflow Checkpoint Cleanup Script

Implements retention policy:
- Keep best 10 checkpoints per experiment (by metric)
- Keep all checkpoints from last 30 days
- Preserve production-tagged models
- Optional cold storage archiving

Usage:
    python checkpoint_cleanup.py --dry-run  # Test without deleting
    python checkpoint_cleanup.py            # Execute cleanup
    python checkpoint_cleanup.py --archive-path /path/to/archive
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import shutil

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Retention configuration
RETENTION_DAYS = 30
TOP_N_CHECKPOINTS = 10
PRODUCTION_TAGS = {'stage': 'production', 'protected': 'true'}
METRIC_NAME = 'sharpe_ratio'  # Primary metric for ranking


class CheckpointCleaner:
    """MLflow checkpoint cleanup with retention policy enforcement."""

    def __init__(
        self,
        tracking_uri: str,
        dry_run: bool = False,
        archive_path: Optional[str] = None,
        retention_days: int = RETENTION_DAYS,
        top_n: int = TOP_N_CHECKPOINTS,
        metric_name: str = METRIC_NAME
    ):
        """Initialize checkpoint cleaner.

        Args:
            tracking_uri: MLflow tracking server URI
            dry_run: If True, only log actions without deleting
            archive_path: Optional path to archive old checkpoints
            retention_days: Days to retain all checkpoints
            top_n: Number of best checkpoints to keep per experiment
            metric_name: Metric name for ranking checkpoints
        """
        self.tracking_uri = tracking_uri
        self.dry_run = dry_run
        self.archive_path = Path(archive_path) if archive_path else None
        self.retention_days = retention_days
        self.top_n = top_n
        self.metric_name = metric_name

        # Initialize MLflow client
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

        # Statistics
        self.stats = {
            'total_runs': 0,
            'protected_runs': 0,
            'recent_runs': 0,
            'top_performers': 0,
            'deleted_runs': 0,
            'archived_runs': 0,
            'space_freed_mb': 0
        }

    def _is_protected(self, run: Run) -> bool:
        """Check if run is protected from deletion.

        Args:
            run: MLflow run to check

        Returns:
            True if run should be protected
        """
        # Check production tags
        for tag_key, tag_value in PRODUCTION_TAGS.items():
            if run.data.tags.get(tag_key) == tag_value:
                logger.info(f"Run {run.info.run_id} protected by tag {tag_key}={tag_value}")
                return True

        # Check if run is registered as a model version
        try:
            # Get all registered models
            registered_models = self.client.search_registered_models()
            for model in registered_models:
                for version in self.client.search_model_versions(f"name='{model.name}'"):
                    if version.run_id == run.info.run_id:
                        logger.info(f"Run {run.info.run_id} protected as registered model {model.name} v{version.version}")
                        return True
        except Exception as e:
            logger.warning(f"Error checking model versions: {e}")

        return False

    def _is_recent(self, run: Run) -> bool:
        """Check if run is within retention period.

        Args:
            run: MLflow run to check

        Returns:
            True if run is recent
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        run_date = datetime.fromtimestamp(run.info.start_time / 1000)
        return run_date > cutoff_date

    def _get_metric_value(self, run: Run) -> Optional[float]:
        """Extract metric value from run.

        Args:
            run: MLflow run

        Returns:
            Metric value or None if not found
        """
        try:
            return run.data.metrics.get(self.metric_name)
        except Exception as e:
            logger.debug(f"Error getting metric {self.metric_name} for run {run.info.run_id}: {e}")
            return None

    def _get_artifact_size(self, run: Run) -> int:
        """Estimate artifact storage size for run.

        Args:
            run: MLflow run

        Returns:
            Size in bytes
        """
        try:
            artifact_uri = run.info.artifact_uri
            if artifact_uri.startswith('file://'):
                artifact_path = Path(artifact_uri.replace('file://', ''))
                if artifact_path.exists():
                    return sum(f.stat().st_size for f in artifact_path.rglob('*') if f.is_file())
        except Exception as e:
            logger.debug(f"Error calculating artifact size for run {run.info.run_id}: {e}")
        return 0

    def _archive_run(self, run: Run) -> bool:
        """Archive run artifacts to cold storage.

        Args:
            run: MLflow run to archive

        Returns:
            True if archiving succeeded
        """
        if not self.archive_path:
            return False

        try:
            artifact_uri = run.info.artifact_uri
            if artifact_uri.startswith('file://'):
                source_path = Path(artifact_uri.replace('file://', ''))
                if source_path.exists():
                    archive_dest = self.archive_path / run.info.experiment_id / run.info.run_id
                    archive_dest.parent.mkdir(parents=True, exist_ok=True)

                    if not self.dry_run:
                        shutil.move(str(source_path), str(archive_dest))

                    logger.info(f"Archived run {run.info.run_id} to {archive_dest}")
                    self.stats['archived_runs'] += 1
                    return True
        except Exception as e:
            logger.error(f"Error archiving run {run.info.run_id}: {e}")

        return False

    def _delete_run(self, run: Run):
        """Delete run and its artifacts.

        Args:
            run: MLflow run to delete
        """
        artifact_size = self._get_artifact_size(run)

        if not self.dry_run:
            try:
                self.client.delete_run(run.info.run_id)
                logger.info(f"Deleted run {run.info.run_id}")
            except Exception as e:
                logger.error(f"Error deleting run {run.info.run_id}: {e}")
                return
        else:
            logger.info(f"[DRY-RUN] Would delete run {run.info.run_id}")

        self.stats['deleted_runs'] += 1
        self.stats['space_freed_mb'] += artifact_size / (1024 * 1024)

    def cleanup_experiment(self, experiment_id: str):
        """Clean up checkpoints for a single experiment.

        Args:
            experiment_id: MLflow experiment ID
        """
        logger.info(f"Processing experiment {experiment_id}")

        # Get all runs for experiment
        runs = self.client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time DESC"]
        )

        if not runs:
            logger.info(f"No runs found for experiment {experiment_id}")
            return

        self.stats['total_runs'] += len(runs)

        # Categorize runs
        protected_runs = []
        recent_runs = []
        candidate_runs = []

        for run in runs:
            if self._is_protected(run):
                protected_runs.append(run)
                self.stats['protected_runs'] += 1
            elif self._is_recent(run):
                recent_runs.append(run)
                self.stats['recent_runs'] += 1
            else:
                candidate_runs.append(run)

        logger.info(
            f"Experiment {experiment_id}: "
            f"{len(protected_runs)} protected, "
            f"{len(recent_runs)} recent, "
            f"{len(candidate_runs)} candidates for cleanup"
        )

        # Rank candidate runs by metric
        ranked_candidates = []
        for run in candidate_runs:
            metric_value = self._get_metric_value(run)
            if metric_value is not None:
                ranked_candidates.append((run, metric_value))
            else:
                ranked_candidates.append((run, float('-inf')))

        ranked_candidates.sort(key=lambda x: x[1], reverse=True)

        # Keep top N performers
        top_performers = [run for run, _ in ranked_candidates[:self.top_n]]
        self.stats['top_performers'] += len(top_performers)

        # Delete remaining runs
        for run, metric_value in ranked_candidates[self.top_n:]:
            logger.info(
                f"Deleting run {run.info.run_id} "
                f"(metric={metric_value:.4f if metric_value != float('-inf') else 'N/A'})"
            )

            # Archive if configured
            if self.archive_path:
                self._archive_run(run)

            # Delete run
            self._delete_run(run)

    def cleanup_all(self):
        """Clean up checkpoints across all experiments."""
        logger.info("Starting checkpoint cleanup")
        logger.info(f"Retention policy: {self.retention_days} days, top {self.top_n} checkpoints")
        logger.info(f"Dry run: {self.dry_run}")

        # Get all experiments
        experiments = self.client.search_experiments()
        logger.info(f"Found {len(experiments)} experiments")

        for experiment in experiments:
            try:
                self.cleanup_experiment(experiment.experiment_id)
            except Exception as e:
                logger.error(f"Error processing experiment {experiment.experiment_id}: {e}")

        # Print statistics
        self._print_stats()

    def _print_stats(self):
        """Print cleanup statistics."""
        logger.info("=" * 60)
        logger.info("Cleanup Statistics")
        logger.info("=" * 60)
        logger.info(f"Total runs processed:     {self.stats['total_runs']}")
        logger.info(f"Protected runs:           {self.stats['protected_runs']}")
        logger.info(f"Recent runs (kept):       {self.stats['recent_runs']}")
        logger.info(f"Top performers (kept):    {self.stats['top_performers']}")
        logger.info(f"Archived runs:            {self.stats['archived_runs']}")
        logger.info(f"Deleted runs:             {self.stats['deleted_runs']}")
        logger.info(f"Space freed:              {self.stats['space_freed_mb']:.2f} MB")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MLflow checkpoint cleanup with retention policy"
    )
    parser.add_argument(
        '--tracking-uri',
        default=os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
        help='MLflow tracking server URI'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without actually deleting checkpoints'
    )
    parser.add_argument(
        '--archive-path',
        help='Path to archive old checkpoints before deletion'
    )
    parser.add_argument(
        '--retention-days',
        type=int,
        default=RETENTION_DAYS,
        help=f'Days to retain all checkpoints (default: {RETENTION_DAYS})'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=TOP_N_CHECKPOINTS,
        help=f'Number of best checkpoints to keep (default: {TOP_N_CHECKPOINTS})'
    )
    parser.add_argument(
        '--metric',
        default=METRIC_NAME,
        help=f'Metric name for ranking (default: {METRIC_NAME})'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create cleaner and run
    cleaner = CheckpointCleaner(
        tracking_uri=args.tracking_uri,
        dry_run=args.dry_run,
        archive_path=args.archive_path,
        retention_days=args.retention_days,
        top_n=args.top_n,
        metric_name=args.metric
    )

    try:
        cleaner.cleanup_all()
        return 0
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

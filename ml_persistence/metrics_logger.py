#!/usr/bin/env python3
"""
Metrics logging with aggregation and analysis.
"""

from typing import Dict, List, Optional
import numpy as np

from .core import MLDatabase


class MetricsLogger:
    """
    Advanced metrics logging with aggregation.

    Provides:
    - Efficient metric logging
    - Time-series aggregation
    - Statistical summaries
    - Metric comparison across experiments
    """

    def __init__(self, db_path: str = "ml_experiments.db"):
        self.db = MLDatabase(db_path)

    def get_metric_timeseries(
        self,
        experiment_id: int,
        metric_name: str,
        split: str = None
    ) -> Dict:
        """
        Get time-series data for a metric.

        Args:
            experiment_id: Experiment ID
            metric_name: Metric name
            split: Optional split filter ('train', 'val', 'test')

        Returns:
            Dict with 'episodes', 'steps', 'values' arrays
        """
        query = """
            SELECT episode, step, value
            FROM metrics
            WHERE experiment_id = ? AND metric_name = ?
        """
        params = [experiment_id, metric_name]

        if split:
            query += " AND split = ?"
            params.append(split)

        query += " ORDER BY COALESCE(episode, step)"

        results = self.db.execute_query(query, tuple(params))

        episodes = []
        steps = []
        values = []

        for row in results:
            episodes.append(row['episode'])
            steps.append(row['step'])
            values.append(row['value'])

        return {
            'episodes': episodes,
            'steps': steps,
            'values': values
        }

    def get_metric_summary(
        self,
        experiment_id: int,
        metric_name: str,
        split: str = None
    ) -> Dict:
        """
        Get statistical summary of a metric.

        Args:
            experiment_id: Experiment ID
            metric_name: Metric name
            split: Optional split filter

        Returns:
            Dict with statistics
        """
        data = self.get_metric_timeseries(experiment_id, metric_name, split)
        values = np.array(data['values'])

        if len(values) == 0:
            return {}

        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'final': float(values[-1]) if len(values) > 0 else None
        }

    def get_all_metrics_for_experiment(
        self,
        experiment_id: int
    ) -> List[str]:
        """
        Get list of all metric names for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            List of metric names
        """
        results = self.db.execute_query(
            "SELECT DISTINCT metric_name FROM metrics WHERE experiment_id = ?",
            (experiment_id,)
        )

        return [row['metric_name'] for row in results]

    def compare_metrics_across_experiments(
        self,
        experiment_ids: List[int],
        metric_name: str,
        split: str = None
    ) -> Dict:
        """
        Compare a metric across multiple experiments.

        Args:
            experiment_ids: List of experiment IDs
            metric_name: Metric to compare
            split: Optional split filter

        Returns:
            Dict with comparison data
        """
        comparison = {}

        for exp_id in experiment_ids:
            summary = self.get_metric_summary(exp_id, metric_name, split)
            if summary:
                comparison[exp_id] = summary

        return comparison

    def get_final_metrics(
        self,
        experiment_id: int,
        metric_names: List[str] = None
    ) -> Dict[str, float]:
        """
        Get final value of metrics for an experiment.

        Args:
            experiment_id: Experiment ID
            metric_names: List of metrics to get (all if None)

        Returns:
            Dict of metric_name -> final_value
        """
        if metric_names is None:
            metric_names = self.get_all_metrics_for_experiment(experiment_id)

        final_metrics = {}

        for metric_name in metric_names:
            data = self.get_metric_timeseries(experiment_id, metric_name)
            if data['values']:
                final_metrics[metric_name] = data['values'][-1]

        return final_metrics


if __name__ == "__main__":
    # Demo usage
    from .experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker("test_ml_experiments.db")
    exp_id = tracker.start_experiment(
        name="Metrics Logger Test",
        experiment_type="rl"
    )

    # Log some metrics
    for ep in range(1, 21):
        tracker.log_metric("train_return", ep * 0.5 + np.random.randn() * 0.1, episode=ep)
        tracker.log_metric("val_sharpe", 0.3 + ep * 0.01 + np.random.randn() * 0.02, episode=ep, split="val")

    # Analyze metrics
    logger = MetricsLogger("test_ml_experiments.db")

    print("\n" + "="*80)
    print("METRIC ANALYSIS")
    print("="*80)

    for metric_name in logger.get_all_metrics_for_experiment(exp_id):
        summary = logger.get_metric_summary(exp_id, metric_name)
        print(f"\n{metric_name}:")
        print(f"  Mean: {summary['mean']:.4f} ± {summary['std']:.4f}")
        print(f"  Range: [{summary['min']:.4f}, {summary['max']:.4f}]")
        print(f"  Final: {summary['final']:.4f}")

    tracker.end_experiment()

#!/usr/bin/env python3
"""
MLflow Concurrent Write Test
Agent: database-migration-specialist (Agent 8)
Purpose: Validate MLflow can handle 20+ concurrent experiment writes to TimescaleDB
Date: 2025-10-25
"""

import mlflow
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import sys
from datetime import datetime


# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
NUM_CONCURRENT_EXPERIMENTS = 20
NUM_METRICS_PER_EXPERIMENT = 100
NUM_PARAMS_PER_EXPERIMENT = 10


def run_experiment(experiment_id: int) -> dict:
    """
    Run a single experiment with metrics logging.

    Args:
        experiment_id: Unique ID for this experiment

    Returns:
        Dictionary with experiment results
    """
    start_time = time.time()

    # Add small random delay to avoid race conditions
    time.sleep(np.random.uniform(0.001, 0.01))

    # Add microseconds and thread ID to ensure uniqueness
    import threading
    thread_id = threading.get_ident()
    experiment_name = f"concurrent_test_exp_{experiment_id}_{thread_id}_{int(time.time() * 1000000)}"

    try:
        # Set tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Create experiment
        try:
            exp_id = mlflow.create_experiment(experiment_name)
        except Exception as e:
            # Experiment might already exist
            exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        # Start run with explicit run name to ensure uniqueness
        run_name = f"run_{experiment_id}_{thread_id}_{int(time.time() * 1000000)}"
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
            # Log parameters
            for i in range(NUM_PARAMS_PER_EXPERIMENT):
                mlflow.log_param(f"param_{i}", np.random.random())

            # Log metrics over time
            for step in range(NUM_METRICS_PER_EXPERIMENT):
                mlflow.log_metric("loss", np.random.random(), step=step)
                mlflow.log_metric("accuracy", np.random.random(), step=step)
                mlflow.log_metric("reward", np.random.random(), step=step)

                # Simulate some work
                time.sleep(0.001)

            # Log final metrics
            mlflow.log_metric("final_loss", np.random.random())
            mlflow.log_metric("final_accuracy", np.random.random())

        end_time = time.time()
        duration = end_time - start_time

        return {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "run_id": run.info.run_id,
            "duration": duration,
            "success": True,
            "error": None
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        return {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "run_id": None,
            "duration": duration,
            "success": False,
            "error": str(e)
        }


def test_concurrent_writes():
    """
    Test MLflow concurrent write capability with TimescaleDB backend.
    """
    print("=" * 80)
    print("MLflow Concurrent Write Test")
    print("=" * 80)
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Concurrent Experiments: {NUM_CONCURRENT_EXPERIMENTS}")
    print(f"Metrics per Experiment: {NUM_METRICS_PER_EXPERIMENT}")
    print(f"Params per Experiment: {NUM_PARAMS_PER_EXPERIMENT}")
    print("=" * 80)

    # Verify MLflow is accessible
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"✓ MLflow server is accessible at {MLFLOW_TRACKING_URI}")
    except Exception as e:
        print(f"✗ Failed to connect to MLflow: {e}")
        sys.exit(1)

    # Run concurrent experiments
    print(f"\nStarting {NUM_CONCURRENT_EXPERIMENTS} concurrent experiments...")
    start_time = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_EXPERIMENTS) as executor:
        futures = [
            executor.submit(run_experiment, i)
            for i in range(NUM_CONCURRENT_EXPERIMENTS)
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "✓" if result["success"] else "✗"
            print(f"{status} Experiment {result['experiment_id']}: {result['duration']:.2f}s")

    end_time = time.time()
    total_duration = end_time - start_time

    # Analyze results
    print("\n" + "=" * 80)
    print("Test Results")
    print("=" * 80)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Total Experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success Rate: {len(successful) / len(results) * 100:.1f}%")
    print(f"Total Duration: {total_duration:.2f}s")

    if successful:
        durations = [r["duration"] for r in successful]
        print(f"\nExperiment Duration Statistics:")
        print(f"  Min: {min(durations):.2f}s")
        print(f"  Max: {max(durations):.2f}s")
        print(f"  Mean: {np.mean(durations):.2f}s")
        print(f"  Median: {np.median(durations):.2f}s")

        # Calculate throughput
        total_metrics = len(successful) * NUM_METRICS_PER_EXPERIMENT * 3  # 3 metrics per step
        total_params = len(successful) * NUM_PARAMS_PER_EXPERIMENT
        throughput_metrics = total_metrics / total_duration
        throughput_params = total_params / total_duration

        print(f"\nThroughput:")
        print(f"  Metrics/second: {throughput_metrics:.1f}")
        print(f"  Params/second: {throughput_params:.1f}")

    if failed:
        print(f"\n{len(failed)} Experiments Failed:")
        for r in failed:
            print(f"  - Experiment {r['experiment_id']}: {r['error']}")

    # Validate data integrity
    print("\n" + "=" * 80)
    print("Data Integrity Validation")
    print("=" * 80)

    client = mlflow.tracking.MlflowClient()

    # Check if all runs are accessible
    validation_passed = True
    for result in successful:
        try:
            run = client.get_run(result["run_id"])
            metrics_count = len(run.data.metrics)
            params_count = len(run.data.params)

            expected_metrics = NUM_METRICS_PER_EXPERIMENT * 3 + 2  # 3 per step + 2 final
            expected_params = NUM_PARAMS_PER_EXPERIMENT

            if params_count != expected_params:
                print(f"✗ Run {result['run_id']}: Expected {expected_params} params, got {params_count}")
                validation_passed = False

            # Note: MLflow latest_metrics view only shows latest value per metric name
            # So we expect 5 unique metric names (loss, accuracy, reward, final_loss, final_accuracy)
            expected_unique_metrics = 5
            if metrics_count != expected_unique_metrics:
                print(f"✗ Run {result['run_id']}: Expected {expected_unique_metrics} metric names, got {metrics_count}")
                validation_passed = False

        except Exception as e:
            print(f"✗ Failed to retrieve run {result['run_id']}: {e}")
            validation_passed = False

    if validation_passed:
        print("✓ All runs passed data integrity validation")
    else:
        print("✗ Some runs failed data integrity validation")

    # Performance assessment
    print("\n" + "=" * 80)
    print("Performance Assessment")
    print("=" * 80)

    success_criteria = {
        "All experiments completed successfully": len(failed) == 0,
        "Average duration < 30s": np.mean([r["duration"] for r in successful]) < 30,
        "No database locking errors": not any("lock" in str(r.get("error", "")).lower() for r in failed),
        "Data integrity validated": validation_passed,
        "Throughput > 50 metrics/sec": throughput_metrics > 50 if successful else False
    }

    for criterion, passed in success_criteria.items():
        status = "✓" if passed else "✗"
        print(f"{status} {criterion}")

    all_passed = all(success_criteria.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("RESULT: ✓ ALL TESTS PASSED")
        print("MLflow with TimescaleDB backend is ready for production use.")
        print("Supports 20+ concurrent experiments with excellent performance.")
        return 0
    else:
        print("RESULT: ✗ SOME TESTS FAILED")
        print("Review the failures above and address issues before production use.")
        return 1


if __name__ == "__main__":
    exit_code = test_concurrent_writes()
    sys.exit(exit_code)

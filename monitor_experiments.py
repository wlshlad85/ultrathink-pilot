#!/usr/bin/env python3
"""
Monitor the three parallel experiments and report completion.
"""
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Experiment definitions
EXPERIMENTS = {
    'exp1_strong': {
        'name': 'Strong Volatility Penalty',
        'model_dir': 'rl/models/exp1_strong',
        'pid': 47382,
        'log': '/tmp/exp1.log'
    },
    'exp2_exp': {
        'name': 'Exponential Decay',
        'model_dir': 'rl/models/exp2_exp',
        'pid': 47387,
        'log': '/tmp/exp2.log'
    },
    'exp3_sharpe': {
        'name': 'Direct Sharpe',
        'model_dir': 'rl/models/exp3_sharpe',
        'pid': 46892,
        'log': '/tmp/exp3.log'
    }
}

def is_process_running(pid):
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def get_metrics_summary(metrics_path):
    """Extract key metrics from training_metrics.json."""
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)

        val_sharpes = data.get('validation_sharpes', [])
        best_val = data.get('best_val_sharpe', 0.0)
        episode_count = len(data.get('episode_rewards', []))

        return {
            'episodes': episode_count,
            'best_val_sharpe': best_val,
            'latest_val_sharpe': val_sharpes[-1] if val_sharpes else 0.0,
            'num_validations': len(val_sharpes)
        }
    except Exception as e:
        return None

def get_log_tail(log_path, n=3):
    """Get last n lines from log file."""
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        return ''.join(lines[-n:])
    except Exception:
        return ""

def main():
    print("=" * 80)
    print("EXPERIMENT MONITOR - Started at", datetime.now().strftime("%H:%M:%S"))
    print("=" * 80)
    print()
    print("Monitoring 3 parallel experiments:")
    for key, exp in EXPERIMENTS.items():
        print(f"  {key}: {exp['name']} (PID {exp['pid']})")
    print()
    print("Will check every 60 seconds and report when each completes...")
    print()

    completed = set()
    start_time = time.time()

    while len(completed) < 3:
        time.sleep(60)  # Check every minute

        elapsed = int(time.time() - start_time)
        elapsed_mins = elapsed // 60

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Status check ({elapsed_mins} min elapsed):")

        for key, exp in EXPERIMENTS.items():
            if key in completed:
                print(f"  {key}: ✅ COMPLETED")
                continue

            # Check if process is still running
            running = is_process_running(exp['pid'])

            # Check if metrics file exists
            metrics_path = Path(exp['model_dir']) / 'training_metrics.json'
            has_metrics = metrics_path.exists()

            if not running and has_metrics:
                # Experiment completed!
                print(f"  {key}: 🎉 JUST COMPLETED!")
                print()

                metrics = get_metrics_summary(metrics_path)
                if metrics:
                    print(f"    {exp['name']} Results:")
                    print(f"      Episodes: {metrics['episodes']}")
                    print(f"      Best Validation Sharpe: {metrics['best_val_sharpe']:.4f}")
                    print(f"      Latest Validation Sharpe: {metrics['latest_val_sharpe']:.4f}")
                    print(f"      Total Validations: {metrics['num_validations']}")
                    print()

                completed.add(key)
            elif running:
                print(f"  {key}: ⏳ Running (PID {exp['pid']})")
            else:
                print(f"  {key}: ⚠️  Process stopped but no metrics found")

        print()

    # All completed!
    total_time = int(time.time() - start_time)
    print("=" * 80)
    print(f"ALL EXPERIMENTS COMPLETED! Total time: {total_time // 60} minutes")
    print("=" * 80)
    print()

    # Print final summary
    print("FINAL RESULTS SUMMARY:")
    print("-" * 80)

    for key, exp in EXPERIMENTS.items():
        metrics_path = Path(exp['model_dir']) / 'training_metrics.json'
        metrics = get_metrics_summary(metrics_path)

        if metrics:
            print(f"{exp['name']}:")
            print(f"  Best Validation Sharpe: {metrics['best_val_sharpe']:+.4f}")
            print(f"  Latest Validation Sharpe: {metrics['latest_val_sharpe']:+.4f}")
            print(f"  Episodes Trained: {metrics['episodes']}")
            print()

    print("=" * 80)
    print("Run compare_training_results.py to see detailed comparison")
    print("=" * 80)

if __name__ == "__main__":
    main()

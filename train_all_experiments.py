#!/usr/bin/env python3
"""
Launch all three reward system experiments in sequence.

This script trains three different reward approaches:
1. Strong Volatility Penalty (sensitivity=100)
2. Exponential Volatility Penalty (exp decay)
3. Direct Sharpe Reward (with floor)
"""

import sys
import subprocess
from pathlib import Path
import argparse

# Training scripts for each experiment
EXPERIMENTS = {
    'strong': {
        'script': 'train_exp1_strong.py',
        'description': 'Strong Volatility Penalty (sensitivity=100)',
        'model_dir': 'rl/models/exp1_strong'
    },
    'exp': {
        'script': 'train_exp2_exp.py',
        'description': 'Exponential Volatility Penalty',
        'model_dir': 'rl/models/exp2_exp'
    },
    'sharpe': {
        'script': 'train_exp3_sharpe.py',
        'description': 'Direct Sharpe Reward',
        'model_dir': 'rl/models/exp3_sharpe'
    }
}

def main():
    parser = argparse.ArgumentParser(description='Train all reward system experiments')
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel (requires multiple GPUs or high RAM)')
    parser.add_argument('--experiment', choices=list(EXPERIMENTS.keys()),
                       help='Run only specific experiment')
    args = parser.parse_args()

    root_dir = Path(__file__).parent

    print("=" * 80)
    print("PARALLEL REWARD SYSTEM EXPERIMENTS")
    print("=" * 80)
    print()

    if args.experiment:
        experiments_to_run = {args.experiment: EXPERIMENTS[args.experiment]}
    else:
        experiments_to_run = EXPERIMENTS

    print(f"Running {len(experiments_to_run)} experiment(s):")
    for key, exp in experiments_to_run.items():
        print(f"  {key}: {exp['description']}")
    print()

    if args.parallel:
        print("Mode: PARALLEL (all experiments simultaneously)")
        print("⚠️  Warning: Requires significant GPU memory or multiple GPUs")
        print()

        processes = []
        for key, exp in experiments_to_run.items():
            script_path = root_dir / exp['script']
            print(f"Launching {key}...")
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            processes.append((key, proc))

        print()
        print("All experiments launched! Waiting for completion...")
        print()

        # Wait for all to complete
        for key, proc in processes:
            print(f"Waiting for {key} to complete...")
            proc.wait()
            if proc.returncode == 0:
                print(f"✅ {key} completed successfully!")
            else:
                print(f"❌ {key} failed with return code {proc.returncode}")
            print()

    else:
        print("Mode: SEQUENTIAL (one after another)")
        print()

        for key, exp in experiments_to_run.items():
            script_path = root_dir / exp['script']

            print("=" * 80)
            print(f"Starting Experiment: {exp['description']}")
            print("=" * 80)
            print()

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=root_dir
            )

            if result.returncode == 0:
                print()
                print(f"✅ {key} completed successfully!")
            else:
                print()
                print(f"❌ {key} failed with return code {result.returncode}")
                print("Stopping further experiments due to failure.")
                sys.exit(1)

            print()

    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 80)
    print()
    print("Results saved in:")
    for key, exp in experiments_to_run.items():
        print(f"  {key}: {exp['model_dir']}/training_metrics.json")
    print()
    print("Run compare_all_experiments.py to analyze results.")

if __name__ == "__main__":
    main()

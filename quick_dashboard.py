#!/usr/bin/env python3
"""
Quick Training Dashboard - Visualize all training progress
"""
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_all_metrics():
    """Load all metrics JSON files."""
    metrics_files = sorted(glob.glob('rl/logs/metrics_*.json'))
    all_data = []

    for f in metrics_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
                data['filename'] = Path(f).name
                all_data.append(data)
        except:
            continue

    return all_data

def create_dashboard():
    """Create comprehensive training dashboard."""
    print("Loading training data...")
    all_runs = load_all_metrics()

    if not all_runs:
        print("No training data found!")
        return

    print(f"Found {len(all_runs)} training runs")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'UltraThink Training Dashboard - {len(all_runs)} Runs', fontsize=16, fontweight='bold')

    # Plot 1: Returns over time across all runs
    ax1 = axes[0, 0]
    for idx, run in enumerate(all_runs[-10:]):  # Last 10 runs
        if 'episode_returns' in run:
            returns = run['episode_returns']
            ax1.plot(returns, alpha=0.6, label=f'Run {idx+1}')
    ax1.set_title('Episode Returns (Last 10 Runs)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return (%)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # Plot 2: Best return per run
    ax2 = axes[0, 1]
    best_returns = []
    for run in all_runs:
        if 'episode_returns' in run and run['episode_returns']:
            best_returns.append(max(run['episode_returns']) * 100)
    ax2.plot(best_returns, marker='o', linestyle='-', linewidth=2, markersize=6)
    ax2.set_title('Best Return Per Training Run')
    ax2.set_xlabel('Training Run')
    ax2.set_ylabel('Best Return (%)')
    ax2.grid(True, alpha=0.3)
    if best_returns:
        ax2.axhline(y=np.mean(best_returns), color='g', linestyle='--',
                    label=f'Mean: {np.mean(best_returns):.1f}%', alpha=0.7)
        ax2.legend()

    # Plot 3: Training loss progression (last run)
    ax3 = axes[1, 0]
    latest_run = all_runs[-1]
    if 'training_metrics' in latest_run:
        metrics = latest_run['training_metrics']
        losses = [m.get('loss', 0) for m in metrics]
        policy_losses = [m.get('policy_loss', 0) for m in metrics]
        value_losses = [m.get('value_loss', 0) for m in metrics]

        ax3.plot(losses, label='Total Loss', linewidth=2)
        ax3.plot(policy_losses, label='Policy Loss', alpha=0.7)
        ax3.plot(value_losses, label='Value Loss', alpha=0.7)
        ax3.set_title('Training Loss (Latest Run)')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No loss data available', ha='center', va='center')

    # Plot 4: Return distribution
    ax4 = axes[1, 1]
    all_returns = []
    for run in all_runs:
        if 'episode_returns' in run:
            all_returns.extend([r * 100 for r in run['episode_returns']])

    if all_returns:
        ax4.hist(all_returns, bins=50, edgecolor='black', alpha=0.7)
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Break-even')
        ax4.axvline(x=np.mean(all_returns), color='g', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(all_returns):.1f}%')
        ax4.set_title('Return Distribution (All Episodes)')
        ax4.set_xlabel('Return (%)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save dashboard
    output_file = 'training_dashboard.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nDashboard saved to: {output_file}")

    # Print summary stats
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total training runs: {len(all_runs)}")
    print(f"Total episodes: {sum(len(r.get('episode_returns', [])) for r in all_runs)}")
    if all_returns:
        print(f"Overall mean return: {np.mean(all_returns):.2f}%")
        print(f"Best return ever: {max(all_returns):.2f}%")
        print(f"Worst return: {min(all_returns):.2f}%")
        print(f"Positive return rate: {100 * sum(1 for r in all_returns if r > 0) / len(all_returns):.1f}%")

    # Model counts
    model_files = list(Path('rl/models').rglob('*.pth'))
    print(f"\nTrained model checkpoints: {len(model_files)}")

    # Model directories
    model_dirs = set(p.parent.name for p in model_files if p.parent.name != 'models')
    print(f"Model variants: {', '.join(sorted(model_dirs))}")

    print("="*60)

    # Don't show - just save
    # plt.show()

if __name__ == '__main__':
    create_dashboard()

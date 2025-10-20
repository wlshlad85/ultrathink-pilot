#!/usr/bin/env python3
"""
Plot training comparison between Simple Reward-Only and Sharpe-Optimized systems.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(path: Path) -> dict:
    """Load training metrics from JSON."""
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)

def plot_comparison():
    """Generate comparison plots."""

    root_dir = Path(__file__).parent

    # Load metrics
    sharpe_path = root_dir / "rl" / "models" / "sharpe_universal" / "training_metrics.json"
    simple_path = root_dir / "rl" / "models" / "simple_reward" / "training_metrics.json"

    sharpe_metrics = load_metrics(sharpe_path)
    simple_metrics = load_metrics(simple_path)

    if not sharpe_metrics or not simple_metrics:
        print("ERROR: Metrics files not found!")
        return

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Comparison: Simple Reward-Only vs Sharpe-Optimized',
                 fontsize=16, fontweight='bold')

    # === PLOT 1: Episode Rewards ===
    ax1 = axes[0, 0]

    sharpe_rewards = sharpe_metrics['episode_rewards']
    simple_rewards = simple_metrics['episode_rewards']

    ax1.plot(sharpe_rewards, label='Sharpe-Optimized', alpha=0.7, linewidth=1)
    ax1.plot(simple_rewards, label='Simple Reward-Only', alpha=0.7, linewidth=1)

    # Add moving average
    window = 10
    if len(sharpe_rewards) >= window:
        sharpe_ma = np.convolve(sharpe_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(sharpe_rewards)), sharpe_ma,
                label='Sharpe MA(10)', linewidth=2, linestyle='--')

    if len(simple_rewards) >= window:
        simple_ma = np.convolve(simple_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(simple_rewards)), simple_ma,
                label='Simple MA(10)', linewidth=2, linestyle='--')

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # === PLOT 2: Validation Sharpe Ratios ===
    ax2 = axes[0, 1]

    sharpe_val_episodes = sharpe_metrics['validation_episodes']
    sharpe_val_sharpes = sharpe_metrics['validation_sharpes']
    simple_val_episodes = simple_metrics['validation_episodes']
    simple_val_sharpes = simple_metrics['validation_sharpes']

    ax2.plot(sharpe_val_episodes, sharpe_val_sharpes,
            marker='o', label='Sharpe-Optimized', linewidth=2)
    ax2.plot(simple_val_episodes, simple_val_sharpes,
            marker='s', label='Simple Reward-Only', linewidth=2)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    ax2.axhline(y=sharpe_metrics['best_val_sharpe'],
               color='tab:blue', linestyle=':', alpha=0.5,
               label=f"Sharpe Best: {sharpe_metrics['best_val_sharpe']:+.3f}")
    ax2.axhline(y=simple_metrics['best_val_sharpe'],
               color='tab:orange', linestyle=':', alpha=0.5,
               label=f"Simple Best: {simple_metrics['best_val_sharpe']:+.3f}")

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Validation Sharpe Ratio')
    ax2.set_title('Validation Performance Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # === PLOT 3: Episode Returns ===
    ax3 = axes[1, 0]

    sharpe_returns = sharpe_metrics['episode_returns']
    simple_returns = simple_metrics['episode_returns']

    ax3.plot(sharpe_returns, label='Sharpe-Optimized', alpha=0.7, linewidth=1)
    ax3.plot(simple_returns, label='Simple Reward-Only', alpha=0.7, linewidth=1)

    # Add moving average
    if len(sharpe_returns) >= window:
        sharpe_ma = np.convolve(sharpe_returns, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(sharpe_returns)), sharpe_ma,
                label='Sharpe MA(10)', linewidth=2, linestyle='--')

    if len(simple_returns) >= window:
        simple_ma = np.convolve(simple_returns, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(simple_returns)), simple_ma,
                label='Simple MA(10)', linewidth=2, linestyle='--')

    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Return (%)')
    ax3.set_title('Episode Returns Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # === PLOT 4: Episode Sharpe Ratios ===
    ax4 = axes[1, 1]

    sharpe_sharpes = sharpe_metrics['episode_sharpes']
    simple_sharpes = simple_metrics['episode_sharpes']

    ax4.plot(sharpe_sharpes, label='Sharpe-Optimized', alpha=0.7, linewidth=1)
    ax4.plot(simple_sharpes, label='Simple Reward-Only', alpha=0.7, linewidth=1)

    # Add moving average
    if len(sharpe_sharpes) >= window:
        sharpe_ma = np.convolve(sharpe_sharpes, np.ones(window)/window, mode='valid')
        ax4.plot(range(window-1, len(sharpe_sharpes)), sharpe_ma,
                label='Sharpe MA(10)', linewidth=2, linestyle='--')

    if len(simple_sharpes) >= window:
        simple_ma = np.convolve(simple_sharpes, np.ones(window)/window, mode='valid')
        ax4.plot(range(window-1, len(simple_sharpes)), simple_ma,
                label='Simple MA(10)', linewidth=2, linestyle='--')

    ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Training Sharpe Ratio')
    ax4.set_title('Training Sharpe Ratios Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()

    output_path = root_dir / "training_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Also show if in interactive mode
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    plot_comparison()

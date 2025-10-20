can you not remember claude i asked him what tools would makwe him ejoy amd get satisfaction out of getting posditive results together and he looked at postgres sqlite mcp plottable mcp for charts and plotting visual matrics and i cant remember the 3rd
#!/usr/bin/env python3
"""
Create comprehensive visualizations comparing all reward system experiments.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

# Experiment definitions
EXPERIMENTS = {
    'sharpe_universal': {
        'name': 'Sharpe-Optimized (Old)',
        'color': '#e74c3c',
        'marker': 'x',
        'linestyle': '--'
    },
    'simple_reward': {
        'name': 'Simple Reward (Old)',
        'color': '#e67e22',
        'marker': 's',
        'linestyle': '--'
    },
    'exp1_strong': {
        'name': 'Exp1: Strong Penalty',
        'color': '#3498db',
        'marker': 'o',
        'linestyle': '-'
    },
    'exp2_exp': {
        'name': 'Exp2: Exponential Decay',
        'color': '#2ecc71',
        'marker': '^',
        'linestyle': '-'
    },
    'exp3_sharpe': {
        'name': 'Exp3: Direct Sharpe ⭐',
        'color': '#9b59b6',
        'marker': 'D',
        'linestyle': '-',
        'linewidth': 2.5
    }
}

def load_metrics(exp_key):
    """Load metrics for an experiment."""
    path = Path(f"rl/models/{exp_key}/training_metrics.json")
    if not path.exists():
        return None

    with open(path, 'r') as f:
        return json.load(f)

def plot_validation_sharpes(ax, all_metrics):
    """Plot validation Sharpe ratios over episodes."""
    ax.set_title('Validation Sharpe Ratio Progression', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Validation Sharpe Ratio', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)

    for exp_key, exp_info in EXPERIMENTS.items():
        metrics = all_metrics.get(exp_key)
        if metrics and 'validation_sharpes' in metrics:
            episodes = metrics['validation_episodes']
            sharpes = metrics['validation_sharpes']

            linewidth = exp_info.get('linewidth', 2)
            ax.plot(episodes, sharpes,
                   label=exp_info['name'],
                   color=exp_info['color'],
                   marker=exp_info['marker'],
                   linestyle=exp_info['linestyle'],
                   linewidth=linewidth,
                   markersize=8,
                   alpha=0.8)

    ax.legend(loc='best', fontsize=10)

    # Add horizontal lines for key values
    ax.axhline(y=0.5, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Good (0.5)')
    ax.axhline(y=1.0, color='darkgreen', linestyle=':', linewidth=1, alpha=0.5, label='Excellent (1.0)')

def plot_episode_rewards(ax, all_metrics):
    """Plot episode rewards distribution."""
    ax.set_title('Episode Rewards Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Reward', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for exp_key, exp_info in EXPERIMENTS.items():
        metrics = all_metrics.get(exp_key)
        if metrics and 'episode_rewards' in metrics:
            rewards = metrics['episode_rewards']

            # For exp3 with huge rewards, use log scale or normalize
            if exp_key == 'exp3_sharpe':
                rewards = np.array(rewards) / 100  # Scale down for visibility

            ax.hist(rewards, bins=30,
                   label=f"{exp_info['name']} (μ={np.mean(rewards):.1f})",
                   color=exp_info['color'],
                   alpha=0.5,
                   edgecolor='black',
                   linewidth=1)

    ax.legend(loc='best', fontsize=9)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

def plot_episode_returns(ax, all_metrics):
    """Plot episode returns over time."""
    ax.set_title('Episode Returns Over Time (Rolling Average)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)

    window = 10  # Rolling average window

    for exp_key, exp_info in EXPERIMENTS.items():
        metrics = all_metrics.get(exp_key)
        if metrics and 'episode_returns' in metrics:
            returns = np.array(metrics['episode_returns'])
            episodes = np.arange(1, len(returns) + 1)

            # Calculate rolling average
            rolling_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
            rolling_episodes = episodes[window-1:]

            linewidth = exp_info.get('linewidth', 1.5)
            ax.plot(rolling_episodes, rolling_avg,
                   label=f"{exp_info['name']} (μ={np.mean(returns):.2f}%)",
                   color=exp_info['color'],
                   linestyle=exp_info['linestyle'],
                   linewidth=linewidth,
                   alpha=0.8)

    ax.legend(loc='best', fontsize=9)

def plot_training_sharpes(ax, all_metrics):
    """Plot training Sharpe ratios over time."""
    ax.set_title('Training Sharpe Ratio Over Time (Rolling Average)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Training Sharpe Ratio', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)

    window = 10  # Rolling average window

    for exp_key, exp_info in EXPERIMENTS.items():
        metrics = all_metrics.get(exp_key)
        if metrics and 'episode_sharpes' in metrics:
            sharpes = np.array(metrics['episode_sharpes'])
            episodes = np.arange(1, len(sharpes) + 1)

            # Calculate rolling average
            rolling_avg = np.convolve(sharpes, np.ones(window)/window, mode='valid')
            rolling_episodes = episodes[window-1:]

            linewidth = exp_info.get('linewidth', 1.5)
            ax.plot(rolling_episodes, rolling_avg,
                   label=f"{exp_info['name']} (μ={np.mean(sharpes):.2f})",
                   color=exp_info['color'],
                   linestyle=exp_info['linestyle'],
                   linewidth=linewidth,
                   alpha=0.8)

    ax.legend(loc='best', fontsize=9)

def plot_summary_comparison(ax, all_metrics):
    """Create a summary comparison bar chart."""
    ax.set_title('Summary Comparison - Key Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    experiments = []
    val_sharpes = []
    mean_returns = []

    for exp_key, exp_info in EXPERIMENTS.items():
        metrics = all_metrics.get(exp_key)
        if metrics:
            experiments.append(exp_info['name'])

            # Validation Sharpe
            if 'validation_sharpes' in metrics and metrics['validation_sharpes']:
                val_sharpes.append(metrics['validation_sharpes'][-1])
            else:
                val_sharpes.append(0)

            # Mean return
            if 'episode_returns' in metrics:
                mean_returns.append(np.mean(metrics['episode_returns']))
            else:
                mean_returns.append(0)

    x = np.arange(len(experiments))
    width = 0.35

    bars1 = ax.bar(x - width/2, val_sharpes, width,
                   label='Validation Sharpe',
                   color='steelblue',
                   edgecolor='black',
                   linewidth=1)

    # Create secondary axis for returns
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, mean_returns, width,
                    label='Mean Return (%)',
                    color='coral',
                    edgecolor='black',
                    linewidth=1)

    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=15, ha='right', fontsize=9)
    ax2.set_ylabel('Mean Return (%)', fontsize=12)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

def plot_reward_scale_comparison(ax, all_metrics):
    """Compare reward scales across experiments."""
    ax.set_title('Reward Scale Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    for exp_key, exp_info in EXPERIMENTS.items():
        metrics = all_metrics.get(exp_key)
        if metrics and 'episode_rewards' in metrics:
            rewards = np.array(metrics['episode_rewards'])
            # Handle negative rewards by adding offset
            rewards_positive = rewards - np.min(rewards) + 1
            episodes = np.arange(1, len(rewards) + 1)

            linewidth = exp_info.get('linewidth', 1.5)
            ax.plot(episodes, rewards_positive,
                   label=f"{exp_info['name']}",
                   color=exp_info['color'],
                   linestyle=exp_info['linestyle'],
                   linewidth=linewidth,
                   alpha=0.6)

    ax.legend(loc='best', fontsize=9)

def main():
    print("=" * 80)
    print("GENERATING EXPERIMENT VISUALIZATIONS")
    print("=" * 80)
    print()

    # Load all metrics
    all_metrics = {}
    for exp_key in EXPERIMENTS.keys():
        print(f"Loading {exp_key}...")
        metrics = load_metrics(exp_key)
        if metrics:
            all_metrics[exp_key] = metrics
            print(f"  ✓ Loaded {len(metrics.get('episode_rewards', []))} episodes")
        else:
            print(f"  ✗ Not found")

    print()
    print(f"Loaded {len(all_metrics)} experiments")
    print()

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Reward System Experiments - Comprehensive Comparison',
                 fontsize=18, fontweight='bold', y=0.995)

    # Create subplot layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])  # Validation Sharpes
    ax2 = fig.add_subplot(gs[0, 1])  # Summary comparison
    ax3 = fig.add_subplot(gs[1, 0])  # Episode returns
    ax4 = fig.add_subplot(gs[1, 1])  # Training Sharpes
    ax5 = fig.add_subplot(gs[2, 0])  # Episode rewards
    ax6 = fig.add_subplot(gs[2, 1])  # Reward scale comparison

    # Generate plots
    print("Generating plots...")
    plot_validation_sharpes(ax1, all_metrics)
    print("  ✓ Validation Sharpes")

    plot_summary_comparison(ax2, all_metrics)
    print("  ✓ Summary Comparison")

    plot_episode_returns(ax3, all_metrics)
    print("  ✓ Episode Returns")

    plot_training_sharpes(ax4, all_metrics)
    print("  ✓ Training Sharpes")

    plot_episode_rewards(ax5, all_metrics)
    print("  ✓ Episode Rewards")

    plot_reward_scale_comparison(ax6, all_metrics)
    print("  ✓ Reward Scale Comparison")

    # Save figure
    output_path = "experiment_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print()
    print(f"✅ Saved visualization to: {output_path}")

    # Create individual focused plots
    create_focused_plots(all_metrics)

    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)

def create_focused_plots(all_metrics):
    """Create individual focused plots."""

    # 1. Validation Sharpe - Large focused plot
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_validation_sharpes(ax, all_metrics)
    plt.tight_layout()
    plt.savefig("validation_sharpes_focused.png", dpi=150, bbox_inches='tight')
    print("  ✓ validation_sharpes_focused.png")
    plt.close()

    # 2. Returns comparison - Box plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Episode Returns Distribution - Box Plot Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    data = []
    labels = []
    colors = []

    for exp_key, exp_info in EXPERIMENTS.items():
        metrics = all_metrics.get(exp_key)
        if metrics and 'episode_returns' in metrics:
            data.append(metrics['episode_returns'])
            labels.append(exp_info['name'])
            colors.append(exp_info['color'])

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig("returns_boxplot.png", dpi=150, bbox_inches='tight')
    print("  ✓ returns_boxplot.png")
    plt.close()

    # 3. Winner spotlight - Exp3 details
    if 'exp3_sharpe' in all_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Experiment 3 (Direct Sharpe) - Detailed Analysis ⭐',
                     fontsize=16, fontweight='bold')

        metrics = all_metrics['exp3_sharpe']

        # Rewards over time
        axes[0, 0].plot(metrics['episode_rewards'],
                       color='#9b59b6', linewidth=1.5, alpha=0.7)
        axes[0, 0].set_title('Episode Rewards Progression')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # Returns over time
        axes[0, 1].plot(metrics['episode_returns'],
                       color='#2ecc71', linewidth=1.5, alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        axes[0, 1].set_title('Episode Returns Progression')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # Sharpe over time
        window = 10
        sharpes = np.array(metrics['episode_sharpes'])
        rolling_sharpe = np.convolve(sharpes, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window, len(sharpes) + 1), rolling_sharpe,
                       color='#e74c3c', linewidth=2, alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        axes[1, 0].set_title(f'Training Sharpe (Rolling Avg, window={window})')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].grid(True, alpha=0.3)

        # Validation Sharpe
        axes[1, 1].plot(metrics['validation_episodes'],
                       metrics['validation_sharpes'],
                       color='#f39c12', linewidth=3, marker='D',
                       markersize=10, alpha=0.8)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        axes[1, 1].axhline(y=0.5, color='green', linestyle=':', linewidth=1, alpha=0.5)
        axes[1, 1].set_title('Validation Sharpe Progression')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Validation Sharpe')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([-0.1, 0.6])

        plt.tight_layout()
        plt.savefig("exp3_detailed_analysis.png", dpi=150, bbox_inches='tight')
        print("  ✓ exp3_detailed_analysis.png")
        plt.close()

if __name__ == "__main__":
    main()

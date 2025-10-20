#!/usr/bin/env python3
"""
Reward Diagnostics Tool

Analyzes reward effectiveness by measuring correlation between rewards and outcomes.
Validates that reward functions incentivize profitable behavior.

Key metrics:
- Reward-Return Correlation: Should be > 0.7 for profitable episodes
- Reward Distribution: Should be mostly positive for winning strategies
- Component Analysis: Breakdown of reward sources
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RewardDiagnostics:
    """
    Diagnostic tool for analyzing reward function effectiveness.
    """

    def __init__(self):
        """Initialize diagnostics."""
        self.episode_data = []

    def log_step(
        self,
        episode: int,
        step: int,
        reward: float,
        portfolio_return: float,
        portfolio_value: float,
        action: int,
        reward_components: Dict[str, float] = None
    ):
        """
        Log a single step for analysis.

        Args:
            episode: Episode number
            step: Step within episode
            reward: Reward received
            portfolio_return: Portfolio return (%)
            portfolio_value: Current portfolio value
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            reward_components: Optional breakdown of reward components
        """
        self.episode_data.append({
            'episode': episode,
            'step': step,
            'reward': reward,
            'portfolio_return': portfolio_return,
            'portfolio_value': portfolio_value,
            'action': action,
            **(reward_components or {})
        })

    def analyze_episode(self, episode: int) -> Dict[str, float]:
        """
        Analyze a specific episode.

        Returns:
            Dictionary of metrics
        """
        df = pd.DataFrame(self.episode_data)
        episode_df = df[df['episode'] == episode]

        if len(episode_df) == 0:
            return {}

        # Calculate correlation between rewards and returns
        correlation = episode_df[['reward', 'portfolio_return']].corr().iloc[0, 1]

        # Reward statistics
        reward_mean = episode_df['reward'].mean()
        reward_std = episode_df['reward'].std()
        reward_min = episode_df['reward'].min()
        reward_max = episode_df['reward'].max()

        # Portfolio statistics
        final_return = episode_df['portfolio_return'].iloc[-1]
        total_reward = episode_df['reward'].sum()

        # Action distribution
        action_dist = episode_df['action'].value_counts().to_dict()

        return {
            'correlation': correlation,
            'reward_mean': reward_mean,
            'reward_std': reward_std,
            'reward_min': reward_min,
            'reward_max': reward_max,
            'final_return_pct': final_return,
            'total_reward': total_reward,
            'action_distribution': action_dist
        }

    def analyze_all(self) -> Dict[str, any]:
        """
        Analyze all logged episodes.

        Returns:
            Comprehensive analysis
        """
        df = pd.DataFrame(self.episode_data)

        if len(df) == 0:
            return {'error': 'No data logged'}

        # Overall correlation
        overall_corr = df[['reward', 'portfolio_return']].corr().iloc[0, 1]

        # Per-episode analysis
        episodes = df['episode'].unique()
        episode_metrics = []

        for ep in episodes:
            metrics = self.analyze_episode(ep)
            metrics['episode'] = ep
            episode_metrics.append(metrics)

        # Aggregate statistics
        df_metrics = pd.DataFrame(episode_metrics)

        return {
            'overall_correlation': overall_corr,
            'mean_episode_correlation': df_metrics['correlation'].mean(),
            'episodes_with_positive_corr': (df_metrics['correlation'] > 0).sum(),
            'episodes_with_high_corr': (df_metrics['correlation'] > 0.7).sum(),
            'mean_reward': df['reward'].mean(),
            'mean_return': df['portfolio_return'].mean(),
            'episode_metrics': episode_metrics
        }

    def plot_diagnostics(self, output_path: str = "rl/reward_diagnostics.png"):
        """
        Generate diagnostic plots.

        Args:
            output_path: Where to save the plot
        """
        df = pd.DataFrame(self.episode_data)

        if len(df) == 0:
            print("No data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Reward vs. Portfolio Return (Scatter)
        axes[0, 0].scatter(df['portfolio_return'], df['reward'], alpha=0.3)
        axes[0, 0].set_xlabel('Portfolio Return (%)')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Reward vs. Portfolio Return')
        axes[0, 0].grid(True)

        # Add correlation line
        z = np.polyfit(df['portfolio_return'], df['reward'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['portfolio_return'].min(), df['portfolio_return'].max(), 100)
        axes[0, 0].plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Correlation: {df[["reward", "portfolio_return"]].corr().iloc[0,1]:.2f}')
        axes[0, 0].legend()

        # Plot 2: Reward Distribution
        axes[0, 1].hist(df['reward'], bins=50, edgecolor='black')
        axes[0, 1].set_xlabel('Reward')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Reward Distribution')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', label='Zero')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot 3: Episode-wise Correlation
        episodes = df['episode'].unique()
        correlations = []
        for ep in episodes:
            ep_df = df[df['episode'] == ep]
            if len(ep_df) > 1:
                corr = ep_df[['reward', 'portfolio_return']].corr().iloc[0, 1]
                correlations.append(corr)

        axes[1, 0].plot(episodes[:len(correlations)], correlations, marker='o')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].set_title('Reward-Return Correlation by Episode')
        axes[1, 0].axhline(y=0.7, color='g', linestyle='--', label='Target (0.7)')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot 4: Cumulative Reward vs. Cumulative Return
        df_sorted = df.sort_values(['episode', 'step'])
        df_sorted['cumulative_reward'] = df_sorted.groupby('episode')['reward'].cumsum()
        df_sorted['cumulative_return'] = df_sorted.groupby('episode')['portfolio_return'].cumsum()

        for ep in episodes[:5]:  # Plot first 5 episodes
            ep_df = df_sorted[df_sorted['episode'] == ep]
            axes[1, 1].plot(ep_df['cumulative_return'], ep_df['cumulative_reward'], alpha=0.7, label=f'Ep {ep}')

        axes[1, 1].set_xlabel('Cumulative Return (%)')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].set_title('Cumulative Reward vs. Return (First 5 Episodes)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Diagnostics plot saved to {output_path}")
        plt.close()

    def save_report(self, output_path: str = "rl/reward_diagnostics_report.json"):
        """
        Save detailed analysis report.

        Args:
            output_path: Where to save the JSON report
        """
        analysis = self.analyze_all()

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        print(f"Diagnostics report saved to {output_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("REWARD DIAGNOSTICS SUMMARY")
        print("=" * 80)
        print(f"Overall Correlation: {analysis.get('overall_correlation', 0):.3f}")
        print(f"Mean Episode Correlation: {analysis.get('mean_episode_correlation', 0):.3f}")
        print(f"Episodes with Correlation > 0.7: {analysis.get('episodes_with_high_corr', 0)}")
        print(f"Mean Reward: {analysis.get('mean_reward', 0):.4f}")
        print(f"Mean Return: {analysis.get('mean_return', 0):.4f}%")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    """Test reward diagnostics."""

    print("Testing Reward Diagnostics Tool...\n")

    diagnostics = RewardDiagnostics()

    # Simulate some data
    np.random.seed(42)

    for episode in range(5):
        portfolio_value = 100000.0
        for step in range(50):
            # Simulate market movement
            ret = np.random.normal(0.001, 0.02)
            portfolio_value *= (1 + ret)
            portfolio_return = (portfolio_value / 100000.0 - 1) * 100

            # Simulate reward (should correlate with returns)
            reward = ret * 1000 + np.random.normal(0, 0.1)

            action = np.random.choice([0, 1, 2])

            diagnostics.log_step(
                episode=episode,
                step=step,
                reward=reward,
                portfolio_return=portfolio_return,
                portfolio_value=portfolio_value,
                action=action
            )

    # Analyze
    diagnostics.save_report("rl/test_diagnostics_report.json")
    diagnostics.plot_diagnostics("rl/test_diagnostics.png")

    print("\nTest complete! Check rl/test_diagnostics.png and rl/test_diagnostics_report.json")

#!/usr/bin/env python3
"""
Statistical Analysis Report for Reward Function Effectiveness

Analyzes correlation between rewards and portfolio returns, generates
statistical metrics, and creates visualization reports.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import seaborn as sns


class RewardAnalysisReport:
    """Generate comprehensive statistical analysis of reward effectiveness."""

    def __init__(self, metrics_path: str):
        """
        Initialize analyzer with training metrics.

        Args:
            metrics_path: Path to training metrics JSON file
        """
        self.metrics_path = Path(metrics_path)
        self.data = self._load_data()
        self.rewards = np.array(self.data['episode_rewards'])
        self.returns = np.array(self.data['episode_returns'])

    def _load_data(self) -> Dict:
        """Load training metrics from JSON."""
        with open(self.metrics_path, 'r') as f:
            return json.load(f)

    def calculate_overall_correlation(self) -> Dict[str, float]:
        """
        Calculate overall Pearson correlation between rewards and returns.

        Returns:
            Dict with correlation coefficient and p-value
        """
        corr, p_value = stats.pearsonr(self.rewards, self.returns)

        return {
            'correlation': float(corr),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'strength': self._interpret_correlation(corr)
        }

    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(corr)
        if abs_corr >= 0.9:
            return "Very Strong"
        elif abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.5:
            return "Moderate"
        elif abs_corr >= 0.3:
            return "Weak"
        else:
            return "Very Weak"

    def calculate_windowed_correlation(self, window: int = 10) -> List[float]:
        """
        Calculate rolling correlation over episodes.

        Args:
            window: Window size for rolling correlation

        Returns:
            List of correlation coefficients
        """
        correlations = []
        for i in range(window, len(self.rewards) + 1):
            window_rewards = self.rewards[i-window:i]
            window_returns = self.returns[i-window:i]
            corr, _ = stats.pearsonr(window_rewards, window_returns)
            correlations.append(corr)
        return correlations

    def calculate_reward_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate descriptive statistics for rewards and returns."""
        return {
            'rewards': {
                'mean': float(np.mean(self.rewards)),
                'std': float(np.std(self.rewards)),
                'min': float(np.min(self.rewards)),
                'max': float(np.max(self.rewards)),
                'median': float(np.median(self.rewards)),
                'q25': float(np.percentile(self.rewards, 25)),
                'q75': float(np.percentile(self.rewards, 75)),
            },
            'returns': {
                'mean': float(np.mean(self.returns)),
                'std': float(np.std(self.returns)),
                'min': float(np.min(self.returns)),
                'max': float(np.max(self.returns)),
                'median': float(np.median(self.returns)),
                'q25': float(np.percentile(self.returns, 25)),
                'q75': float(np.percentile(self.returns, 75)),
            }
        }

    def calculate_alignment_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics showing reward-return alignment.

        Returns:
            Dict with alignment metrics
        """
        # Sign agreement: do reward and return have same sign?
        sign_agreement = np.sum(np.sign(self.rewards) == np.sign(self.returns))
        sign_agreement_pct = (sign_agreement / len(self.rewards)) * 100

        # Magnitude correlation: does larger return → larger reward?
        magnitude_corr, _ = stats.pearsonr(np.abs(self.rewards), np.abs(self.returns))

        # Positive return episodes
        positive_return_mask = self.returns > 0
        positive_return_episodes = np.sum(positive_return_mask)
        positive_reward_for_positive_return = np.sum(
            (self.rewards > 0) & positive_return_mask
        )
        positive_alignment_pct = (
            (positive_reward_for_positive_return / positive_return_episodes * 100)
            if positive_return_episodes > 0 else 0.0
        )

        # Negative return episodes
        negative_return_mask = self.returns < 0
        negative_return_episodes = np.sum(negative_return_mask)
        negative_reward_for_negative_return = np.sum(
            (self.rewards < 0) & negative_return_mask
        )
        negative_alignment_pct = (
            (negative_reward_for_negative_return / negative_return_episodes * 100)
            if negative_return_episodes > 0 else 0.0
        )

        return {
            'sign_agreement_pct': float(sign_agreement_pct),
            'magnitude_correlation': float(magnitude_corr),
            'positive_alignment_pct': float(positive_alignment_pct),
            'negative_alignment_pct': float(negative_alignment_pct),
            'total_episodes': len(self.rewards),
            'positive_return_episodes': int(positive_return_episodes),
            'negative_return_episodes': int(negative_return_episodes)
        }

    def generate_report(self) -> Dict:
        """Generate complete statistical analysis report."""
        report = {
            'overall_correlation': self.calculate_overall_correlation(),
            'statistics': self.calculate_reward_statistics(),
            'alignment': self.calculate_alignment_metrics(),
            'windowed_correlation': {
                'window_size': 10,
                'correlations': self.calculate_windowed_correlation(10)
            }
        }
        return report

    def plot_analysis(self, output_dir: str = "rl/models/diagnostic_test"):
        """
        Generate comprehensive visualization plots.

        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create figure with 6 subplots
        fig = plt.figure(figsize=(18, 12))

        # 1. Scatter plot: Reward vs Return
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(self.returns, self.rewards, alpha=0.6, s=50)

        # Add regression line
        z = np.polyfit(self.returns, self.rewards, 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.returns.min(), self.returns.max(), 100)
        ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'y={z[0]:.3f}x+{z[1]:.3f}')

        corr, p_val = stats.pearsonr(self.rewards, self.returns)
        ax1.set_xlabel('Portfolio Return (%)', fontsize=11)
        ax1.set_ylabel('Episode Reward', fontsize=11)
        ax1.set_title(f'Reward vs. Return (r={corr:.3f}, p={p_val:.4f})', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Reward distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(self.rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(np.mean(self.rewards), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(self.rewards):.3f}')
        ax2.axvline(np.median(self.rewards), color='green', linestyle='--', linewidth=2, label=f'Median={np.median(self.rewards):.3f}')
        ax2.set_xlabel('Reward', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Reward Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Return distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(self.returns, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(np.mean(self.returns), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(self.returns):.3f}%')
        ax3.axvline(np.median(self.returns), color='blue', linestyle='--', linewidth=2, label=f'Median={np.median(self.returns):.3f}%')
        ax3.set_xlabel('Portfolio Return (%)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Return Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Episode-wise correlation trend
        ax4 = plt.subplot(2, 3, 4)
        window_corrs = self.calculate_windowed_correlation(10)
        episodes = range(10, 10 + len(window_corrs))
        ax4.plot(episodes, window_corrs, linewidth=2, color='purple')
        ax4.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Target (0.7)')
        ax4.axhline(0.0, color='gray', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Episode', fontsize=11)
        ax4.set_ylabel('Correlation (10-episode window)', fontsize=11)
        ax4.set_title('Rolling Correlation Trend', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim([-1.1, 1.1])

        # 5. Cumulative rewards and returns
        ax5 = plt.subplot(2, 3, 5)
        cumulative_rewards = np.cumsum(self.rewards)
        cumulative_returns = np.cumsum(self.returns)
        episodes_range = range(1, len(self.rewards) + 1)

        ax5_twin = ax5.twinx()
        line1 = ax5.plot(episodes_range, cumulative_rewards, color='blue', linewidth=2, label='Cumulative Reward')
        line2 = ax5_twin.plot(episodes_range, cumulative_returns, color='green', linewidth=2, label='Cumulative Return (%)')

        ax5.set_xlabel('Episode', fontsize=11)
        ax5.set_ylabel('Cumulative Reward', color='blue', fontsize=11)
        ax5_twin.set_ylabel('Cumulative Return (%)', color='green', fontsize=11)
        ax5.set_title('Cumulative Performance', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='upper left')

        # 6. Sign agreement matrix
        ax6 = plt.subplot(2, 3, 6)

        # Create confusion matrix
        reward_positive = self.rewards > 0
        return_positive = self.returns > 0

        matrix = np.array([
            [np.sum(~reward_positive & ~return_positive), np.sum(~reward_positive & return_positive)],
            [np.sum(reward_positive & ~return_positive), np.sum(reward_positive & return_positive)]
        ])

        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax6,
                    xticklabels=['Return < 0', 'Return > 0'],
                    yticklabels=['Reward < 0', 'Reward > 0'],
                    cbar_kws={'label': 'Count'})
        ax6.set_title('Sign Agreement Matrix', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Portfolio Return Sign', fontsize=11)
        ax6.set_ylabel('Reward Sign', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_path / 'reward_analysis.png', dpi=150, bbox_inches='tight')
        print(f"[OK] Saved analysis plots to {output_path / 'reward_analysis.png'}")
        plt.close()

    def save_report(self, output_path: str = "rl/models/diagnostic_test/reward_analysis.json"):
        """Save statistical report to JSON."""
        report = self.generate_report()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"[OK] Saved analysis report to {output_file}")
        return report


def print_report_summary(report: Dict):
    """Print formatted summary of analysis report."""
    print("\n" + "="*70)
    print("REWARD FUNCTION EFFECTIVENESS - STATISTICAL ANALYSIS REPORT")
    print("="*70)

    # Overall correlation
    corr_data = report['overall_correlation']
    print(f"\n[OVERALL CORRELATION]")
    print(f"   Pearson r: {corr_data['correlation']:.4f}")
    print(f"   P-value: {corr_data['p_value']:.6f}")
    print(f"   Strength: {corr_data['strength']}")
    print(f"   Significant: {'[YES]' if corr_data['significant'] else '[NO]'}")

    # Alignment metrics
    align_data = report['alignment']
    print(f"\n[ALIGNMENT METRICS]")
    print(f"   Sign Agreement: {align_data['sign_agreement_pct']:.1f}%")
    print(f"   Magnitude Correlation: {align_data['magnitude_correlation']:.4f}")
    print(f"   Positive Episodes Aligned: {align_data['positive_alignment_pct']:.1f}%")
    print(f"   Negative Episodes Aligned: {align_data['negative_alignment_pct']:.1f}%")
    print(f"   Total Episodes: {align_data['total_episodes']}")
    print(f"   Positive Return Episodes: {align_data['positive_return_episodes']}")
    print(f"   Negative Return Episodes: {align_data['negative_return_episodes']}")

    # Statistics
    stats_data = report['statistics']
    print(f"\n[REWARD STATISTICS]")
    print(f"   Mean: {stats_data['rewards']['mean']:.4f}")
    print(f"   Std Dev: {stats_data['rewards']['std']:.4f}")
    print(f"   Range: [{stats_data['rewards']['min']:.4f}, {stats_data['rewards']['max']:.4f}]")
    print(f"   Median: {stats_data['rewards']['median']:.4f}")

    print(f"\n[RETURN STATISTICS]")
    print(f"   Mean: {stats_data['returns']['mean']:.2f}%")
    print(f"   Std Dev: {stats_data['returns']['std']:.2f}%")
    print(f"   Range: [{stats_data['returns']['min']:.2f}%, {stats_data['returns']['max']:.2f}%]")
    print(f"   Median: {stats_data['returns']['median']:.2f}%")

    # Validation status
    print(f"\n[VALIDATION STATUS]")
    corr_pass = corr_data['correlation'] > 0.7
    sign_pass = align_data['sign_agreement_pct'] > 90
    pos_align_pass = align_data['positive_alignment_pct'] > 90

    print(f"   Correlation > 0.7: {'[PASS]' if corr_pass else '[FAIL]'}")
    print(f"   Sign Agreement > 90%: {'[PASS]' if sign_pass else '[FAIL]'}")
    print(f"   Positive Alignment > 90%: {'[PASS]' if pos_align_pass else '[FAIL]'}")

    overall_pass = corr_pass and sign_pass and pos_align_pass
    print(f"\n   Overall Status: {'*** ALL TESTS PASSED ***' if overall_pass else '[!] SOME TESTS FAILED'}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import sys

    # Default to diagnostic test metrics
    metrics_file = "rl/models/diagnostic_test/final_metrics.json"

    if len(sys.argv) > 1:
        metrics_file = sys.argv[1]

    print(f"Loading metrics from: {metrics_file}\n")

    # Run analysis
    analyzer = RewardAnalysisReport(metrics_file)
    report = analyzer.generate_report()

    # Save results
    analyzer.save_report()
    analyzer.plot_analysis()

    # Print summary
    print_report_summary(report)

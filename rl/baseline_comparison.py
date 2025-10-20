#!/usr/bin/env python3
"""
Baseline Comparison Script

Compares the fixed reward system against the broken baseline to quantify improvements.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple


class BaselineComparison:
    """Compare fixed reward system against broken baseline."""

    def __init__(self, baseline_path: str, fixed_path: str):
        """
        Initialize comparison with baseline and fixed metrics.

        Args:
            baseline_path: Path to baseline (broken) metrics JSON
            fixed_path: Path to fixed reward metrics JSON
        """
        self.baseline_path = Path(baseline_path)
        self.fixed_path = Path(fixed_path)

        self.baseline_data = self._load_data(self.baseline_path)
        self.fixed_data = self._load_data(self.fixed_path)

        # Extract arrays
        self.baseline_rewards = np.array(self.baseline_data['episode_rewards'])
        self.baseline_returns = np.array(self.baseline_data['episode_returns'])
        self.fixed_rewards = np.array(self.fixed_data['episode_rewards'])
        self.fixed_returns = np.array(self.fixed_data['episode_returns'])

    def _load_data(self, path: Path) -> Dict:
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def calculate_correlations(self) -> Dict[str, float]:
        """Calculate correlations for baseline and fixed systems."""
        baseline_corr, baseline_p = stats.pearsonr(
            self.baseline_rewards, self.baseline_returns
        )
        fixed_corr, fixed_p = stats.pearsonr(
            self.fixed_rewards, self.fixed_returns
        )

        return {
            'baseline': {
                'correlation': float(baseline_corr),
                'p_value': float(baseline_p)
            },
            'fixed': {
                'correlation': float(fixed_corr),
                'p_value': float(fixed_p)
            },
            'improvement': {
                'correlation_delta': float(fixed_corr - baseline_corr),
                'correlation_change_pct': float((fixed_corr - baseline_corr) / abs(baseline_corr) * 100)
            }
        }

    def calculate_reward_ranges(self) -> Dict:
        """Calculate reward range statistics."""
        return {
            'baseline': {
                'min': float(np.min(self.baseline_rewards)),
                'max': float(np.max(self.baseline_rewards)),
                'mean': float(np.mean(self.baseline_rewards)),
                'std': float(np.std(self.baseline_rewards)),
                'range': float(np.ptp(self.baseline_rewards))
            },
            'fixed': {
                'min': float(np.min(self.fixed_rewards)),
                'max': float(np.max(self.fixed_rewards)),
                'mean': float(np.mean(self.fixed_rewards)),
                'std': float(np.std(self.fixed_rewards)),
                'range': float(np.ptp(self.fixed_rewards))
            }
        }

    def calculate_alignment_comparison(self) -> Dict:
        """Compare sign alignment between baseline and fixed."""
        # Baseline alignment
        baseline_sign_agreement = np.sum(
            np.sign(self.baseline_rewards) == np.sign(self.baseline_returns)
        )
        baseline_alignment_pct = (baseline_sign_agreement / len(self.baseline_rewards)) * 100

        # Fixed alignment
        fixed_sign_agreement = np.sum(
            np.sign(self.fixed_rewards) == np.sign(self.fixed_returns)
        )
        fixed_alignment_pct = (fixed_sign_agreement / len(self.fixed_rewards)) * 100

        return {
            'baseline': {
                'sign_agreement_pct': float(baseline_alignment_pct),
                'correct_episodes': int(baseline_sign_agreement),
                'total_episodes': len(self.baseline_rewards)
            },
            'fixed': {
                'sign_agreement_pct': float(fixed_alignment_pct),
                'correct_episodes': int(fixed_sign_agreement),
                'total_episodes': len(self.fixed_rewards)
            },
            'improvement': {
                'alignment_delta_pct': float(fixed_alignment_pct - baseline_alignment_pct)
            }
        }

    def calculate_return_performance(self) -> Dict:
        """Compare portfolio return performance."""
        return {
            'baseline': {
                'mean_return_pct': float(np.mean(self.baseline_returns)),
                'total_return_pct': float(np.sum(self.baseline_returns)),
                'std_pct': float(np.std(self.baseline_returns)),
                'sharpe_ratio': float(
                    np.mean(self.baseline_returns) / np.std(self.baseline_returns)
                    if np.std(self.baseline_returns) > 0 else 0.0
                )
            },
            'fixed': {
                'mean_return_pct': float(np.mean(self.fixed_returns)),
                'total_return_pct': float(np.sum(self.fixed_returns)),
                'std_pct': float(np.std(self.fixed_returns)),
                'sharpe_ratio': float(
                    np.mean(self.fixed_returns) / np.std(self.fixed_returns)
                    if np.std(self.fixed_returns) > 0 else 0.0
                )
            }
        }

    def generate_comparison_report(self) -> Dict:
        """Generate complete comparison report."""
        return {
            'correlations': self.calculate_correlations(),
            'reward_ranges': self.calculate_reward_ranges(),
            'alignment': self.calculate_alignment_comparison(),
            'returns': self.calculate_return_performance(),
            'metadata': {
                'baseline_file': str(self.baseline_path),
                'fixed_file': str(self.fixed_path),
                'baseline_episodes': len(self.baseline_rewards),
                'fixed_episodes': len(self.fixed_rewards)
            }
        }

    def plot_comparison(self, output_path: str = "rl/models/diagnostic_test/baseline_comparison.png"):
        """Generate comprehensive comparison visualizations."""
        fig = plt.figure(figsize=(18, 12))

        # 1. Correlation comparison - Scatter plots
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(self.baseline_returns, self.baseline_rewards, alpha=0.6, s=30, label='Baseline', color='red')
        baseline_corr, _ = stats.pearsonr(self.baseline_rewards, self.baseline_returns)
        z_base = np.polyfit(self.baseline_returns, self.baseline_rewards, 1)
        p_base = np.poly1d(z_base)
        x_line = np.linspace(self.baseline_returns.min(), self.baseline_returns.max(), 100)
        ax1.plot(x_line, p_base(x_line), "r--", alpha=0.8, linewidth=2)
        ax1.set_xlabel('Portfolio Return (%)', fontsize=10)
        ax1.set_ylabel('Episode Reward', fontsize=10)
        ax1.set_title(f'Baseline: r={baseline_corr:.3f}', fontsize=11, fontweight='bold', color='red')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter(self.fixed_returns, self.fixed_rewards, alpha=0.6, s=30, label='Fixed', color='green')
        fixed_corr, _ = stats.pearsonr(self.fixed_rewards, self.fixed_returns)
        z_fixed = np.polyfit(self.fixed_returns, self.fixed_rewards, 1)
        p_fixed = np.poly1d(z_fixed)
        x_line_fixed = np.linspace(self.fixed_returns.min(), self.fixed_returns.max(), 100)
        ax2.plot(x_line_fixed, p_fixed(x_line_fixed), "g--", alpha=0.8, linewidth=2)
        ax2.set_xlabel('Portfolio Return (%)', fontsize=10)
        ax2.set_ylabel('Episode Reward', fontsize=10)
        ax2.set_title(f'Fixed: r={fixed_corr:.3f}', fontsize=11, fontweight='bold', color='green')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Correlation bar chart
        ax3 = plt.subplot(2, 3, 3)
        correlations = [baseline_corr, fixed_corr]
        colors = ['red', 'green']
        bars = ax3.bar(['Baseline', 'Fixed'], correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax3.axhline(0.7, color='blue', linestyle='--', alpha=0.5, label='Target (0.7)', linewidth=2)
        ax3.set_ylabel('Pearson Correlation', fontsize=11)
        ax3.set_title('Correlation Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylim([-1.0, 1.1])
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend()

        # Add value labels on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{corr:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

        # 4. Reward range comparison
        ax4 = plt.subplot(2, 3, 4)
        baseline_stats = [np.min(self.baseline_rewards), np.mean(self.baseline_rewards), np.max(self.baseline_rewards)]
        fixed_stats = [np.min(self.fixed_rewards), np.mean(self.fixed_rewards), np.max(self.fixed_rewards)]

        x = np.arange(3)
        width = 0.35
        bars1 = ax4.bar(x - width/2, baseline_stats, width, label='Baseline', color='red', alpha=0.7, edgecolor='black')
        bars2 = ax4.bar(x + width/2, fixed_stats, width, label='Fixed', color='green', alpha=0.7, edgecolor='black')

        ax4.set_ylabel('Reward Value', fontsize=11)
        ax4.set_title('Reward Statistics Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Min', 'Mean', 'Max'])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

        # 5. Sign alignment comparison
        ax5 = plt.subplot(2, 3, 5)
        baseline_alignment = (np.sum(np.sign(self.baseline_rewards) == np.sign(self.baseline_returns)) /
                            len(self.baseline_rewards)) * 100
        fixed_alignment = (np.sum(np.sign(self.fixed_rewards) == np.sign(self.fixed_returns)) /
                          len(self.fixed_rewards)) * 100

        alignments = [baseline_alignment, fixed_alignment]
        bars = ax5.bar(['Baseline', 'Fixed'], alignments, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax5.axhline(90, color='blue', linestyle='--', alpha=0.5, label='Target (90%)', linewidth=2)
        ax5.set_ylabel('Sign Agreement (%)', fontsize=11)
        ax5.set_title('Reward-Return Sign Alignment', fontsize=12, fontweight='bold')
        ax5.set_ylim([0, 105])
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.legend()

        # Add value labels
        for bar, align in zip(bars, alignments):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{align:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

        # 6. Improvement summary table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        improvement_data = [
            ['Metric', 'Baseline', 'Fixed', 'Improvement'],
            ['Correlation', f'{baseline_corr:.3f}', f'{fixed_corr:.3f}',
             f'+{(fixed_corr - baseline_corr):.3f}'],
            ['Sign Align %', f'{baseline_alignment:.1f}%', f'{fixed_alignment:.1f}%',
             f'+{(fixed_alignment - baseline_alignment):.1f}%'],
            ['Mean Reward', f'{np.mean(self.baseline_rewards):.3f}',
             f'{np.mean(self.fixed_rewards):.3f}',
             f'{(np.mean(self.fixed_rewards) - np.mean(self.baseline_rewards)):.3f}'],
            ['Reward Range', f'{np.ptp(self.baseline_rewards):.1f}',
             f'{np.ptp(self.fixed_rewards):.1f}',
             f'{(np.ptp(self.fixed_rewards) - np.ptp(self.baseline_rewards)):.1f}']
        ]

        table = ax6.table(cellText=improvement_data, cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(4):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')

        # Style improvement column based on positive/negative
        for i in range(1, 5):
            cell = table[(i, 3)]
            value_text = improvement_data[i][3]
            if value_text.startswith('+'):
                cell.set_facecolor('#C8E6C9')  # Light green
            else:
                cell.set_facecolor('#FFCDD2')  # Light red

        ax6.set_title('Improvement Summary', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved comparison plots to {output_file}")
        plt.close()

    def save_report(self, output_path: str = "rl/models/diagnostic_test/baseline_comparison.json"):
        """Save comparison report to JSON."""
        report = self.generate_comparison_report()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"[OK] Saved comparison report to {output_file}")
        return report


def print_comparison_summary(report: Dict):
    """Print formatted comparison summary."""
    print("\n" + "="*70)
    print("BASELINE vs. FIXED REWARD SYSTEM COMPARISON")
    print("="*70)

    # Correlation comparison
    corr_data = report['correlations']
    print("\n[CORRELATION ANALYSIS]")
    print(f"   Baseline Correlation: {corr_data['baseline']['correlation']:.4f}")
    print(f"   Fixed Correlation: {corr_data['fixed']['correlation']:.4f}")
    print(f"   Improvement: +{corr_data['improvement']['correlation_delta']:.4f} "
          f"({corr_data['improvement']['correlation_change_pct']:+.1f}%)")

    # Reward range comparison
    reward_data = report['reward_ranges']
    print("\n[REWARD STATISTICS]")
    print(f"   Baseline: Min={reward_data['baseline']['min']:.2f}, "
          f"Mean={reward_data['baseline']['mean']:.2f}, "
          f"Max={reward_data['baseline']['max']:.2f}")
    print(f"   Fixed: Min={reward_data['fixed']['min']:.2f}, "
          f"Mean={reward_data['fixed']['mean']:.2f}, "
          f"Max={reward_data['fixed']['max']:.2f}")

    # Alignment comparison
    align_data = report['alignment']
    print("\n[SIGN ALIGNMENT]")
    print(f"   Baseline: {align_data['baseline']['sign_agreement_pct']:.1f}% "
          f"({align_data['baseline']['correct_episodes']}/{align_data['baseline']['total_episodes']} episodes)")
    print(f"   Fixed: {align_data['fixed']['sign_agreement_pct']:.1f}% "
          f"({align_data['fixed']['correct_episodes']}/{align_data['fixed']['total_episodes']} episodes)")
    print(f"   Improvement: +{align_data['improvement']['alignment_delta_pct']:.1f}%")

    # Return performance
    return_data = report['returns']
    print("\n[PORTFOLIO PERFORMANCE]")
    print(f"   Baseline: Mean Return={return_data['baseline']['mean_return_pct']:.2f}%, "
          f"Sharpe={return_data['baseline']['sharpe_ratio']:.2f}")
    print(f"   Fixed: Mean Return={return_data['fixed']['mean_return_pct']:.2f}%, "
          f"Sharpe={return_data['fixed']['sharpe_ratio']:.2f}")

    # Validation
    print("\n[VALIDATION RESULTS]")
    baseline_pass = corr_data['baseline']['correlation'] > 0.7
    fixed_pass = corr_data['fixed']['correlation'] > 0.7
    print(f"   Baseline Passes (r>0.7): {'[PASS]' if baseline_pass else '[FAIL]'}")
    print(f"   Fixed Passes (r>0.7): {'[PASS]' if fixed_pass else '[FAIL]'}")

    if not baseline_pass and fixed_pass:
        print(f"\n   Status: *** REWARD FUNCTION SUCCESSFULLY FIXED ***")
    elif baseline_pass and fixed_pass:
        print(f"\n   Status: [INFO] Both systems pass (baseline already working)")
    else:
        print(f"\n   Status: [!] Fixed system did not meet targets")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import sys

    # Default paths
    baseline_file = "rl/models/phase3_test/final_metrics.json"
    fixed_file = "rl/models/diagnostic_test/final_metrics.json"

    if len(sys.argv) > 2:
        baseline_file = sys.argv[1]
        fixed_file = sys.argv[2]

    print(f"Baseline metrics: {baseline_file}")
    print(f"Fixed metrics: {fixed_file}\n")

    # Run comparison
    comparison = BaselineComparison(baseline_file, fixed_file)
    report = comparison.generate_comparison_report()

    # Save results
    comparison.save_report()
    comparison.plot_comparison()

    # Print summary
    print_comparison_summary(report)

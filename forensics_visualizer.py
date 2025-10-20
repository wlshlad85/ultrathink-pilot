#!/usr/bin/env python3
"""
Forensics Visualization Suite

Creates visualizations to understand model failure patterns:
- Timeline plots showing bad decisions on price charts
- Pattern distribution breakdowns
- Confidence vs outcome analysis
- Regime-specific performance
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import List, Optional, Dict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from trade_decision import TradeDecision, ForensicsReport


class ForensicsVisualizer:
    """
    Visualization suite for trade forensics analysis.
    """

    def __init__(self, figsize=(16, 10)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_decision_timeline(
        self,
        decisions: List[TradeDecision],
        market_data: pd.DataFrame,
        output_path: Optional[str] = None,
        highlight_bad_only: bool = False
    ):
        """
        Plot trading decisions on price timeline.

        This shows WHERE and WHEN the model made bad decisions,
        providing visual context for failure patterns.

        Args:
            decisions: List of TradeDecision objects
            market_data: DataFrame with price data (must have 'close' column and datetime index)
            output_path: Optional path to save figure
            highlight_bad_only: Only show bad decisions if True
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 1])

        # Convert timestamps to datetime for plotting
        timestamps = [pd.to_datetime(d.timestamp) for d in decisions]

        # Plot price
        ax1.plot(market_data.index, market_data['close'],
                label='Price', color='black', alpha=0.7, linewidth=1.5)

        # Plot decisions
        for decision in decisions:
            if highlight_bad_only and not decision.is_bad_decision:
                continue

            timestamp = pd.to_datetime(decision.timestamp)
            price = decision.price

            if decision.action == "BUY":
                if decision.is_bad_decision:
                    color, marker, size = 'red', 'v', 150
                    label = 'Bad BUY'
                else:
                    color, marker, size = 'green', '^', 80
                    label = 'Good BUY'

                ax1.scatter(timestamp, price, c=color, marker=marker,
                          s=size, alpha=0.7, edgecolors='black', linewidths=1,
                          zorder=5)

            elif decision.action == "SELL":
                color, marker, size = 'blue', 'o', 80
                ax1.scatter(timestamp, price, c=color, marker=marker,
                          s=size, alpha=0.7, edgecolors='black', linewidths=1,
                          zorder=5)

        # Formatting
        ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
        ax1.set_title('Trading Decisions Timeline', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(['Price', 'Bad BUY', 'Good BUY', 'SELL'])

        # Plot portfolio value over time
        portfolio_values = [d.portfolio_value for d in decisions]
        ax2.plot(timestamps, portfolio_values, color='navy', linewidth=2)
        ax2.fill_between(timestamps, portfolio_values, alpha=0.3)
        ax2.set_ylabel('Portfolio Value', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved timeline plot to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_failure_patterns(
        self,
        decisions: List[TradeDecision],
        output_path: Optional[str] = None
    ):
        """
        Plot distribution of failure patterns.

        Shows which types of mistakes are most common,
        helping prioritize what the model needs to learn.

        Args:
            decisions: List of TradeDecision objects
            output_path: Optional path to save figure
        """
        # Count failure patterns
        bad_decisions = [d for d in decisions if d.is_bad_decision]

        if not bad_decisions:
            print("No bad decisions to visualize")
            return

        pattern_counts = {}
        pattern_costs = {}

        for decision in bad_decisions:
            pattern = decision.failure_pattern
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            pattern_costs[pattern] = pattern_costs.get(pattern, 0) + decision.cost_of_mistake

        # Sort by frequency
        patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Plot 1: Frequency
        pattern_names = [p[0].replace('_', ' ').title() for p in patterns]
        counts = [p[1] for p in patterns]
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(patterns)))

        bars1 = ax1.barh(pattern_names, counts, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Number of Occurrences', fontsize=12, fontweight='bold')
        ax1.set_title('Failure Patterns by Frequency', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}',
                    ha='left', va='center', fontsize=10, fontweight='bold')

        # Plot 2: Cost
        costs = [pattern_costs.get(p[0], 0) for p in patterns]
        bars2 = ax2.barh(pattern_names, costs, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Total Cost ($)', fontsize=12, fontweight='bold')
        ax2.set_title('Failure Patterns by Cost', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'${int(width):,}',
                    ha='left', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved failure patterns plot to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_confidence_vs_outcome(
        self,
        decisions: List[TradeDecision],
        output_path: Optional[str] = None
    ):
        """
        Plot model confidence vs actual outcomes.

        This reveals if the model is overconfident in bad decisions,
        which indicates a calibration problem.

        Args:
            decisions: List of TradeDecision objects
            output_path: Optional path to save figure
        """
        # Filter to BUY decisions only (most interesting)
        buy_decisions = [d for d in decisions if d.action == "BUY" and d.returns_5d is not None]

        if not buy_decisions:
            print("No BUY decisions to analyze")
            return

        confidences = [d.confidence for d in buy_decisions]
        returns = [d.returns_5d for d in buy_decisions]
        is_bad = [d.is_bad_decision for d in buy_decisions]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Plot 1: Scatter plot
        colors = ['red' if bad else 'green' for bad in is_bad]
        ax1.scatter(confidences, returns, c=colors, alpha=0.6, s=100, edgecolors='black', linewidths=0.5)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axhline(y=-5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Failure Threshold')
        ax1.set_xlabel('Model Confidence', fontsize=12, fontweight='bold')
        ax1.set_ylabel('5-Day Forward Return (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Confidence vs Outcome (BUY Decisions)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(['Zero Return', 'Failure Threshold', 'Bad Decision', 'Good Decision'])

        # Plot 2: Confidence distribution for good vs bad
        good_confidences = [d.confidence for d in buy_decisions if not d.is_bad_decision]
        bad_confidences = [d.confidence for d in buy_decisions if d.is_bad_decision]

        bins = np.linspace(0, 1, 20)
        ax2.hist(good_confidences, bins=bins, alpha=0.6, label='Good Decisions', color='green', edgecolor='black')
        ax2.hist(bad_confidences, bins=bins, alpha=0.6, label='Bad Decisions', color='red', edgecolor='black')
        ax2.set_xlabel('Model Confidence', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Confidence Distribution: Good vs Bad', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved confidence analysis to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_regime_performance(
        self,
        decisions: List[TradeDecision],
        output_path: Optional[str] = None
    ):
        """
        Plot performance breakdown by market regime.

        Shows if the model struggles in specific market conditions
        (e.g., bear markets, high volatility).

        Args:
            decisions: List of TradeDecision objects
            output_path: Optional path to save figure
        """
        # Aggregate by regime
        regime_stats = {
            'bull': {'total': 0, 'bad': 0, 'cost': 0},
            'bear': {'total': 0, 'bad': 0, 'cost': 0},
            'neutral': {'total': 0, 'bad': 0, 'cost': 0}
        }

        for decision in decisions:
            regime = decision.regime
            regime_stats[regime]['total'] += 1
            if decision.is_bad_decision:
                regime_stats[regime]['bad'] += 1
                regime_stats[regime]['cost'] += decision.cost_of_mistake

        # Calculate error rates
        regimes = list(regime_stats.keys())
        error_rates = [
            regime_stats[r]['bad'] / regime_stats[r]['total'] * 100
            if regime_stats[r]['total'] > 0 else 0
            for r in regimes
        ]
        total_costs = [regime_stats[r]['cost'] for r in regimes]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Plot 1: Error rates
        colors = ['green', 'red', 'gray']
        bars1 = ax1.bar(regimes, error_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Bad Decision Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Error Rate by Market Regime', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Plot 2: Total costs
        bars2 = ax2.bar(regimes, total_costs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Total Cost of Mistakes ($)', fontsize=12, fontweight='bold')
        ax2.set_title('Cost by Market Regime', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height,
                    f'${int(height):,}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved regime performance to {output_path}")
        else:
            plt.show()

        plt.close()

    def create_full_report(
        self,
        decisions: List[TradeDecision],
        market_data: pd.DataFrame,
        output_dir: str = "forensics_output"
    ):
        """
        Generate complete visualization report.

        Creates all visualization types and saves to directory.

        Args:
            decisions: List of TradeDecision objects
            market_data: DataFrame with price data
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nGenerating forensics visualizations in {output_dir}/")
        print("-" * 60)

        # Timeline
        print("Creating timeline plot...")
        self.plot_decision_timeline(
            decisions,
            market_data,
            output_path=str(output_path / "timeline.png")
        )

        # Failure patterns
        print("Creating failure patterns plot...")
        self.plot_failure_patterns(
            decisions,
            output_path=str(output_path / "failure_patterns.png")
        )

        # Confidence analysis
        print("Creating confidence analysis...")
        self.plot_confidence_vs_outcome(
            decisions,
            output_path=str(output_path / "confidence_analysis.png")
        )

        # Regime performance
        print("Creating regime performance plot...")
        self.plot_regime_performance(
            decisions,
            output_path=str(output_path / "regime_performance.png")
        )

        print("-" * 60)
        print(f"✓ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    """
    Test visualizations with sample data.
    """
    print("Testing Forensics Visualizer...")

    # This would normally be called from run_forensics.py with real data
    print("\nNote: Run this through run_forensics.py for actual analysis")
    print("This file provides visualization functions for forensics data")

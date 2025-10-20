#!/usr/bin/env python3
"""
Label Historical Bitcoin Data with Market Regimes

Uses the RegimeDetector to classify all historical data points
and create a labeled dataset for:
1. Training regime-specific agents
2. Validating regime detection accuracy
3. Analyzing regime-specific performance
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from regime_detector import RegimeDetector, RegimePrimary, RegimeVolatility
from backtesting.data_fetcher import DataFetcher


def load_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load Bitcoin price data."""
    print(f"Loading {symbol} data from {start_date} to {end_date}...")
    fetcher = DataFetcher(symbol=symbol)
    df = fetcher.fetch(start_date, end_date)
    df = fetcher.add_technical_indicators()
    print(f"Loaded {len(df)} data points")
    return df


def generate_regime_statistics(labeled_data: pd.DataFrame) -> Dict:
    """Generate statistics about regime distribution."""

    stats = {}

    # Primary regime distribution
    primary_counts = labeled_data['regime_primary'].value_counts()
    stats['primary_distribution'] = primary_counts.to_dict()
    stats['primary_percentages'] = (primary_counts / len(labeled_data) * 100).to_dict()

    # Volatility regime distribution
    vol_counts = labeled_data['regime_volatility'].value_counts()
    stats['volatility_distribution'] = vol_counts.to_dict()

    # Average confidence
    stats['avg_confidence'] = labeled_data['regime_confidence'].mean()
    stats['min_confidence'] = labeled_data['regime_confidence'].min()

    # Regime durations
    regime_changes = labeled_data['regime_primary'] != labeled_data['regime_primary'].shift()
    regime_runs = regime_changes.cumsum()
    regime_durations = labeled_data.groupby(regime_runs).size()
    stats['avg_regime_duration'] = regime_durations.mean()
    stats['median_regime_duration'] = regime_durations.median()
    stats['min_regime_duration'] = regime_durations.min()
    stats['max_regime_duration'] = regime_durations.max()

    # Performance by regime
    for regime in [RegimePrimary.BULL.value, RegimePrimary.BEAR.value, RegimePrimary.SIDEWAYS.value]:
        regime_data = labeled_data[labeled_data['regime_primary'] == regime]
        if len(regime_data) > 0:
            stats[f'{regime.lower()}_avg_return'] = regime_data['returns_1d'].mean()
            stats[f'{regime.lower()}_volatility'] = regime_data['returns_1d'].std()
            stats[f'{regime.lower()}_sharpe'] = (
                regime_data['returns_1d'].mean() / regime_data['returns_1d'].std() * np.sqrt(365)
                if regime_data['returns_1d'].std() > 0 else 0
            )

    return stats


def visualize_regimes(labeled_data: pd.DataFrame, output_path: Path):
    """Create visualizations of regime classifications."""

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # Plot 1: Price with regime backgrounds
    ax1 = axes[0]
    ax1.plot(labeled_data.index, labeled_data['close'], 'k-', linewidth=0.5, label='Price')

    # Color backgrounds by regime
    regime_colors = {
        'BULL': 'lightgreen',
        'BEAR': 'lightcoral',
        'SIDEWAYS': 'lightgray',
        'UNKNOWN': 'white'
    }

    for regime, color in regime_colors.items():
        regime_mask = labeled_data['regime_primary'] == regime
        if regime_mask.any():
            # Find contiguous regions
            regime_changes = regime_mask != regime_mask.shift()
            regime_groups = regime_changes.cumsum()

            for group_id in regime_groups[regime_mask].unique():
                group_data = labeled_data[regime_groups == group_id]
                if len(group_data) > 0:
                    ax1.axvspan(
                        group_data.index[0],
                        group_data.index[-1],
                        alpha=0.3,
                        color=color,
                        label=regime if group_id == regime_groups[regime_mask].iloc[0] else None
                    )

    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.set_title('Bitcoin Price with Market Regimes', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization

    # Plot 2: Volatility with regime classification
    ax2 = axes[1]
    ax2.plot(labeled_data.index, labeled_data['volatility'], 'b-', linewidth=1, label='Volatility')
    ax2.axhline(y=0.30, color='g', linestyle='--', alpha=0.5, label='Low threshold')
    ax2.axhline(y=0.80, color='r', linestyle='--', alpha=0.5, label='High threshold')
    ax2.set_ylabel('Volatility (Annualized)', fontsize=12)
    ax2.set_title('Volatility Regime Classification', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Regime confidence over time
    ax3 = axes[2]
    ax3.plot(labeled_data.index, labeled_data['regime_confidence'], 'purple', linewidth=1)
    ax3.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='High confidence threshold')
    ax3.fill_between(labeled_data.index, 0, labeled_data['regime_confidence'],
                      alpha=0.3, color='purple')
    ax3.set_ylabel('Classification Confidence', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Regime Detection Confidence', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    plt.close()


def main():
    """Main execution function."""

    print("=" * 80)
    print("HISTORICAL DATA REGIME LABELING")
    print("=" * 80)
    print()

    # Configuration
    symbol = "BTC-USD"
    start_date = "2017-01-01"
    end_date = "2024-12-31"
    output_csv = "bitcoin_labeled_regimes.csv"
    output_visualization = "regime_visualization.png"
    output_stats = "regime_statistics.txt"

    # Load data
    market_data = load_market_data(symbol, start_date, end_date)
    print()

    # Initialize regime detector
    detector = RegimeDetector(
        lookback_window=30,
        volatility_window=20,
        trend_threshold=0.05,
        min_regime_duration=5
    )

    # Label data
    print("=" * 80)
    labeled_data = detector.label_historical_data(
        market_data,
        smooth_transitions=True
    )
    print()

    # Generate statistics
    print("=" * 80)
    print("GENERATING STATISTICS")
    print("=" * 80)
    stats = generate_regime_statistics(labeled_data)

    # Print summary statistics
    print()
    print("📊 Regime Distribution:")
    print("-" * 40)
    for regime, count in stats['primary_distribution'].items():
        pct = stats['primary_percentages'][regime]
        print(f"  {regime:<12}: {count:>4} days ({pct:>5.1f}%)")

    print()
    print("📏 Regime Duration Statistics:")
    print("-" * 40)
    print(f"  Average:  {stats['avg_regime_duration']:.1f} days")
    print(f"  Median:   {stats['median_regime_duration']:.1f} days")
    print(f"  Min:      {stats['min_regime_duration']} days")
    print(f"  Max:      {stats['max_regime_duration']} days")

    print()
    print("🎯 Classification Confidence:")
    print("-" * 40)
    print(f"  Average:  {stats['avg_confidence']:.3f}")
    print(f"  Minimum:  {stats['min_confidence']:.3f}")

    print()
    print("📈 Performance by Regime:")
    print("-" * 40)
    for regime in ['bull', 'bear', 'sideways']:
        if f'{regime}_avg_return' in stats:
            avg_ret = stats[f'{regime}_avg_return'] * 100
            sharpe = stats[f'{regime}_sharpe']
            print(f"  {regime.upper():<12}: Avg Return = {avg_ret:>+7.3f}%, Sharpe = {sharpe:>6.3f}")

    # Save labeled data
    print()
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Select columns to save
    output_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'regime_primary', 'regime_volatility', 'regime_strength', 'regime_confidence',
        'returns_1d', 'returns_30d', 'volatility', 'rsi', 'adx',
        'price_vs_sma_50', 'ma_alignment'
    ]

    labeled_data[output_columns].to_csv(output_csv)
    print(f"✅ Labeled data saved to: {output_csv}")

    # Save detailed statistics
    with open(output_stats, 'w') as f:
        f.write("BITCOIN REGIME LABELING STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        f.write(f"Total Days: {len(labeled_data)}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("REGIME DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for regime, count in stats['primary_distribution'].items():
            pct = stats['primary_percentages'][regime]
            f.write(f"{regime:<15}: {count:>5} days ({pct:>6.2f}%)\n")

        f.write("\n")
        f.write("REGIME DURATION STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average:  {stats['avg_regime_duration']:.2f} days\n")
        f.write(f"Median:   {stats['median_regime_duration']:.2f} days\n")
        f.write(f"Min:      {stats['min_regime_duration']} days\n")
        f.write(f"Max:      {stats['max_regime_duration']} days\n")

        f.write("\n")
        f.write("PERFORMANCE BY REGIME\n")
        f.write("-" * 80 + "\n")
        for regime in ['bull', 'bear', 'sideways']:
            if f'{regime}_avg_return' in stats:
                f.write(f"\n{regime.upper()}:\n")
                f.write(f"  Avg Daily Return: {stats[f'{regime}_avg_return']*100:>+8.4f}%\n")
                f.write(f"  Daily Volatility: {stats[f'{regime}_volatility']*100:>8.4f}%\n")
                f.write(f"  Sharpe Ratio:     {stats[f'{regime}_sharpe']:>8.3f}\n")

    print(f"✅ Statistics saved to: {output_stats}")

    # Create visualizations
    print()
    print("Creating visualizations...")
    visualize_regimes(labeled_data, Path(output_visualization))

    print()
    print("=" * 80)
    print("✅ LABELING COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review regime_visualization.png to validate classifications")
    print("  2. Check regime_statistics.txt for distribution analysis")
    print("  3. Use bitcoin_labeled_regimes.csv for regime-specific agent training")
    print()

    return labeled_data, stats


if __name__ == "__main__":
    try:
        import numpy as np
        labeled_data, stats = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nLabeling interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

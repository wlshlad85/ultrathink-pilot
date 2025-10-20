#!/usr/bin/env python3
"""
Compare Simple Reward-Only vs Sharpe-Optimized Training Results
"""

import json
import sys
from pathlib import Path
import numpy as np

def load_metrics(path: Path) -> dict:
    """Load training metrics from JSON."""
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)

def analyze_rewards(rewards: list) -> dict:
    """Analyze reward distribution."""
    rewards_array = np.array(rewards)
    return {
        'mean': float(np.mean(rewards_array)),
        'std': float(np.std(rewards_array)),
        'min': float(np.min(rewards_array)),
        'max': float(np.max(rewards_array)),
        'positive_count': int(np.sum(rewards_array > 0)),
        'negative_count': int(np.sum(rewards_array < 0)),
        'zero_count': int(np.sum(rewards_array == 0))
    }

def analyze_validation(val_sharpes: list) -> dict:
    """Analyze validation Sharpe ratios."""
    sharpes_array = np.array(val_sharpes)
    return {
        'mean': float(np.mean(sharpes_array)),
        'std': float(np.std(sharpes_array)),
        'min': float(np.min(sharpes_array)),
        'max': float(np.max(sharpes_array)),
        'positive_count': int(np.sum(sharpes_array > 0)),
        'zero_count': int(np.sum(sharpes_array == 0)),
        'improving_trend': bool(len(sharpes_array) >= 2 and sharpes_array[-1] > sharpes_array[0])
    }

def print_comparison():
    """Print detailed comparison of both training runs."""

    root_dir = Path(__file__).parent

    # Load metrics
    sharpe_path = root_dir / "rl" / "models" / "sharpe_universal" / "training_metrics.json"
    simple_path = root_dir / "rl" / "models" / "simple_reward" / "training_metrics.json"

    sharpe_metrics = load_metrics(sharpe_path)
    simple_metrics = load_metrics(simple_path)

    print("=" * 80)
    print("TRAINING COMPARISON: Simple Reward-Only vs Sharpe-Optimized")
    print("=" * 80)
    print()

    if not sharpe_metrics:
        print("ERROR: Sharpe-optimized metrics not found!")
        return

    if not simple_metrics:
        print("ERROR: Simple reward-only metrics not found!")
        print("Training may still be in progress...")
        return

    # === REWARD ANALYSIS ===
    print("=" * 80)
    print("EPISODE REWARDS")
    print("=" * 80)
    print()

    sharpe_rewards = analyze_rewards(sharpe_metrics['episode_rewards'])
    simple_rewards = analyze_rewards(simple_metrics['episode_rewards'])

    print(f"{'Metric':<30} {'Sharpe-Optimized':<20} {'Simple Reward-Only':<20}")
    print("-" * 80)
    print(f"{'Mean Reward':<30} {sharpe_rewards['mean']:>+18.2f}   {simple_rewards['mean']:>+18.2f}")
    print(f"{'Std Dev':<30} {sharpe_rewards['std']:>18.2f}   {simple_rewards['std']:>18.2f}")
    print(f"{'Min Reward':<30} {sharpe_rewards['min']:>+18.2f}   {simple_rewards['min']:>+18.2f}")
    print(f"{'Max Reward':<30} {sharpe_rewards['max']:>+18.2f}   {simple_rewards['max']:>+18.2f}")
    print()
    print(f"{'Positive Rewards':<30} {sharpe_rewards['positive_count']:>18d}   {simple_rewards['positive_count']:>18d}")
    print(f"{'Negative Rewards':<30} {sharpe_rewards['negative_count']:>18d}   {simple_rewards['negative_count']:>18d}")
    print(f"{'Zero Rewards':<30} {sharpe_rewards['zero_count']:>18d}   {simple_rewards['zero_count']:>18d}")
    print()

    # === VALIDATION ANALYSIS ===
    print("=" * 80)
    print("VALIDATION SHARPE RATIOS")
    print("=" * 80)
    print()

    sharpe_val = analyze_validation(sharpe_metrics['validation_sharpes'])
    simple_val = analyze_validation(simple_metrics['validation_sharpes'])

    print(f"{'Metric':<30} {'Sharpe-Optimized':<20} {'Simple Reward-Only':<20}")
    print("-" * 80)
    print(f"{'Mean Validation Sharpe':<30} {sharpe_val['mean']:>+18.3f}   {simple_val['mean']:>+18.3f}")
    print(f"{'Std Dev':<30} {sharpe_val['std']:>18.3f}   {simple_val['std']:>18.3f}")
    print(f"{'Min Sharpe':<30} {sharpe_val['min']:>+18.3f}   {simple_val['min']:>+18.3f}")
    print(f"{'Max Sharpe':<30} {sharpe_val['max']:>+18.3f}   {simple_val['max']:>+18.3f}")
    print()
    print(f"{'Best Validation Sharpe':<30} {sharpe_metrics['best_val_sharpe']:>+18.3f}   {simple_metrics['best_val_sharpe']:>+18.3f}")
    print()
    print(f"{'Positive Sharpes':<30} {sharpe_val['positive_count']:>18d}   {simple_val['positive_count']:>18d}")
    print(f"{'Zero Sharpes':<30} {sharpe_val['zero_count']:>18d}   {simple_val['zero_count']:>18d}")
    print(f"{'Improving Trend':<30} {str(sharpe_val['improving_trend']):>18}   {str(simple_val['improving_trend']):>18}")
    print()

    # === TRAINING EFFICIENCY ===
    print("=" * 80)
    print("TRAINING EFFICIENCY")
    print("=" * 80)
    print()

    print(f"{'Metric':<30} {'Sharpe-Optimized':<20} {'Simple Reward-Only':<20}")
    print("-" * 80)
    print(f"{'Total Episodes':<30} {sharpe_metrics['total_episodes']:>18d}   {simple_metrics['total_episodes']:>18d}")
    print(f"{'Early Stopped':<30} {str(sharpe_metrics.get('early_stopped', False)):>18}   {str(simple_metrics.get('early_stopped', False)):>18}")
    print(f"{'Validations Run':<30} {len(sharpe_metrics['validation_sharpes']):>18d}   {len(simple_metrics['validation_sharpes']):>18d}")
    print()

    # === VERDICT ===
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()

    sharpe_best = sharpe_metrics['best_val_sharpe']
    simple_best = simple_metrics['best_val_sharpe']

    improvements = []

    # Check reward sign flip
    if sharpe_rewards['mean'] < 0 and simple_rewards['mean'] > 0:
        improvements.append("✅ REWARD SIGN FLIP: Negative → Positive rewards")

    # Check validation improvement
    if simple_best > sharpe_best:
        improvement_pct = ((simple_best - sharpe_best) / abs(sharpe_best) * 100) if sharpe_best != 0 else float('inf')
        improvements.append(f"✅ VALIDATION IMPROVEMENT: {sharpe_best:+.3f} → {simple_best:+.3f} ({improvement_pct:+.1f}%)")

    # Check zero Sharpe problem
    if sharpe_val['zero_count'] > 0 and simple_val['zero_count'] == 0:
        improvements.append("✅ ZERO SHARPE FIXED: Agent now trades during validation")

    # Check positive validation Sharpes
    if simple_val['positive_count'] > sharpe_val['positive_count']:
        improvements.append(f"✅ MORE POSITIVE VALIDATIONS: {sharpe_val['positive_count']} → {simple_val['positive_count']}")

    if improvements:
        print("IMPROVEMENTS DETECTED:")
        print()
        for improvement in improvements:
            print(f"  {improvement}")
        print()
    else:
        print("⚠️  NO CLEAR IMPROVEMENT")
        print()

    # Overall assessment
    if simple_best > 0:
        print("🎉 SUCCESS: Simple reward-only system achieved POSITIVE validation Sharpe!")
    elif simple_best > sharpe_best:
        print("✅ PROGRESS: Simple reward-only system outperformed Sharpe-optimized version")
    else:
        print("❌ NEEDS WORK: Simple reward-only system did not improve results")

    print()
    print("=" * 80)
    print()

if __name__ == "__main__":
    try:
        print_comparison()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

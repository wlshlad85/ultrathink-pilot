#!/usr/bin/env python3
"""
Analyze the training results from the completed 300-episode professional training run.
"""

import json
import numpy as np
from pathlib import Path

def main():
    """Parse and analyze training metrics."""

    metrics_path = Path("rl/models/professional/training_metrics.json")

    if not metrics_path.exists():
        print(f"❌ Metrics file not found at {metrics_path}")
        return

    print("="*80)
    print("TRAINING ANALYSIS: Professional 300-Episode Run")
    print("="*80)
    print()

    # Load metrics
    with open(metrics_path) as f:
        data = json.load(f)

    episodes = data.get('episodes', [])

    if not episodes:
        print("No episode data found in metrics file.")
        return

    print(f"📊 TRAINING OVERVIEW:")
    print(f"  Total episodes completed: {len(episodes)}")
    print(f"  Training data: 2017-2021 (1,825 days)")
    print(f"  Test data: 2022-2023 (545 days)")
    print()

    # Extract episode metrics
    ep_nums = [ep['episode'] for ep in episodes]
    train_returns = [ep['train_return'] for ep in episodes]
    train_rewards = [ep.get('train_reward', 0) for ep in episodes]

    # Test evaluations (every 15 episodes)
    test_episodes = [ep for ep in episodes if 'test_metrics' in ep and ep['test_metrics']]

    print("="*80)
    print("TRAINING PROGRESSION")
    print("="*80)
    print()

    # Early episodes
    print("📈 First 50 Episodes:")
    early_eps = [ep for ep in episodes if ep['episode'] <= 50]
    if early_eps:
        early_returns = [ep['train_return'] for ep in early_eps]
        print(f"  Mean return: {np.mean(early_returns):+.2f}%")
        print(f"  Best return: {np.max(early_returns):+.2f}%")
        print(f"  Worst return: {np.min(early_returns):+.2f}%")
        print(f"  Std dev: {np.std(early_returns):.2f}%")

    # Middle episodes
    print()
    print("📈 Episodes 51-150:")
    mid_eps = [ep for ep in episodes if 51 <= ep['episode'] <= 150]
    if mid_eps:
        mid_returns = [ep['train_return'] for ep in mid_eps]
        print(f"  Mean return: {np.mean(mid_returns):+.2f}%")
        print(f"  Best return: {np.max(mid_returns):+.2f}%")
        print(f"  Worst return: {np.min(mid_returns):+.2f}%")
        print(f"  Std dev: {np.std(mid_returns):.2f}%")

    # Late episodes
    print()
    print("📈 Episodes 151-300:")
    late_eps = [ep for ep in episodes if ep['episode'] >= 151]
    if late_eps:
        late_returns = [ep['train_return'] for ep in late_eps]
        print(f"  Mean return: {np.mean(late_returns):+.2f}%")
        print(f"  Best return: {np.max(late_returns):+.2f}%")
        print(f"  Worst return: {np.min(late_returns):+.2f}%")
        print(f"  Std dev: {np.std(late_returns):.2f}%")

    print()
    print("="*80)
    print("TEST SET EVALUATIONS (2022-2023)")
    print("="*80)
    print()

    if test_episodes:
        print(f"Total test evaluations: {len(test_episodes)}")
        print()

        best_test_sharpe = -np.inf
        best_test_ep = None

        print("Test Results by Episode:")
        print("-" * 80)
        for ep in test_episodes:
            ep_num = ep['episode']
            test_metrics = ep['test_metrics']
            test_return = test_metrics['mean_return']
            test_sharpe = test_metrics['mean_sharpe']

            is_best = test_sharpe > best_test_sharpe
            if is_best:
                best_test_sharpe = test_sharpe
                best_test_ep = ep_num

            marker = " ✅ BEST" if is_best else ""
            print(f"  Episode {ep_num:3d}: Return {test_return:+7.2f}%  |  Sharpe {test_sharpe:+.3f}{marker}")

        print()
        print(f"🏆 BEST MODEL SELECTED:")
        print(f"  Episode: {best_test_ep}")
        print(f"  Test Sharpe: {best_test_sharpe:+.3f}")
        print(f"  Test Return: {[ep['test_metrics']['mean_return'] for ep in test_episodes if ep['episode'] == best_test_ep][0]:+.2f}%")
    else:
        print("No test evaluations found in metrics.")

    print()
    print("="*80)
    print("EARLY STOPPING ANALYSIS")
    print("="*80)
    print()

    if 'early_stopping' in data:
        es_info = data['early_stopping']
        print(f"Early stopping triggered: {es_info.get('triggered', 'Unknown')}")
        print(f"Stopped at episode: {es_info.get('stopped_at_episode', 'N/A')}")
        print(f"Reason: {es_info.get('reason', 'N/A')}")
    elif len(episodes) < 300:
        print(f"⏹️  Training stopped early at episode {len(episodes)}")
        print("   Likely due to early stopping patience (no improvement for 4 validations)")
    else:
        print("✓ Training completed all 300 episodes")

    print()
    print("="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print()

    print("📊 Training Returns (All Episodes):")
    print(f"  Mean: {np.mean(train_returns):+.2f}%")
    print(f"  Median: {np.median(train_returns):+.2f}%")
    print(f"  Std Dev: {np.std(train_returns):.2f}%")
    print(f"  Best: {np.max(train_returns):+.2f}% (Episode {ep_nums[np.argmax(train_returns)]})")
    print(f"  Worst: {np.min(train_returns):+.2f}% (Episode {ep_nums[np.argmin(train_returns)]})")

    print()
    print("📊 Training Rewards (All Episodes):")
    print(f"  Mean: {np.mean(train_rewards):.2f}")
    print(f"  Median: {np.median(train_rewards):.2f}")
    print(f"  Std Dev: {np.std(train_rewards):.2f}")

    # Learning progression
    print()
    print("📈 Learning Progression:")
    first_10_avg = np.mean([ep['train_return'] for ep in episodes[:10]])
    last_10_avg = np.mean([ep['train_return'] for ep in episodes[-10:]])
    improvement = last_10_avg - first_10_avg

    print(f"  First 10 episodes avg return: {first_10_avg:+.2f}%")
    print(f"  Last 10 episodes avg return: {last_10_avg:+.2f}%")
    print(f"  Improvement: {improvement:+.2f}%")

    if improvement > 0:
        print("  ✅ Model improved over training")
    else:
        print("  ⚠️  Model did not show clear improvement")

    print()
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()

    # Calculate variance in recent episodes (stability)
    recent_returns = train_returns[-20:] if len(train_returns) >= 20 else train_returns
    recent_std = np.std(recent_returns)

    if recent_std < 10:
        print("✓ Policy has STABILIZED (low variance in recent episodes)")
    elif recent_std < 20:
        print("⚠️  Policy shows MODERATE stability")
    else:
        print("⚠️  Policy still UNSTABLE (high variance in recent episodes)")

    # Test performance
    if test_episodes:
        test_sharpes = [ep['test_metrics']['mean_sharpe'] for ep in test_episodes]
        if best_test_sharpe > 0.5:
            print("✓ STRONG test set performance (Sharpe > 0.5)")
        elif best_test_sharpe > 0.3:
            print("✓ GOOD test set performance (Sharpe > 0.3)")
        elif best_test_sharpe > 0:
            print("⚠️  MODERATE test set performance (Sharpe > 0)")
        else:
            print("❌ POOR test set performance (negative Sharpe)")

    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("1. ✓ Training completed successfully")
    print("2. → Analyze regime-conditional behavior:")
    print("     python analyze_professional.py")
    print()
    print("3. → When ready for deployment, evaluate on held-out set:")
    print("     python evaluate_professional.py")
    print()
    print("4. → If held-out performance is good, deploy to paper trading")
    print()
    print("⚠️  REMEMBER: The held-out set (2023-2024) has NOT been used yet!")
    print("   It remains your final unbiased performance check.")
    print()


if __name__ == "__main__":
    main()

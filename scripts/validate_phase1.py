#!/usr/bin/env python3
"""
Phase 1 Validation Script
=========================

Compares old training system (legacy) vs new training system (unified pipeline).

Tests:
1. Run 10 training episodes with old system
2. Run 10 training episodes with new system
3. Compare:
   - I/O time percentage
   - Total training time
   - Feature consistency
   - Final performance
4. Generate validation report

Expected Improvements:
- I/O time: 40% → <10%
- Training speed: 2-3x faster
- Features: 20 → 70+
- Cache hit rate: N/A → 90%+
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import both versions
from rl.trading_env import TradingEnv  # Legacy (v1)
from rl.trading_env_v3 import TradingEnvV3  # New (v3)
from rl.ppo_agent import PPOAgent
import torch


def profile_training_episode(env, agent, episode_num=1):
    """
    Profile a single training episode.

    Returns:
        dict: Performance metrics
    """
    metrics = {
        'episode': episode_num,
        'total_time': 0,
        'io_time': 0,
        'compute_time': 0,
        'steps': 0,
        'reward': 0,
        'final_value': 0,
        'return_pct': 0
    }

    start_time = time.time()

    # Reset environment (this includes data loading)
    reset_start = time.time()
    state, info = env.reset(seed=42 + episode_num)
    reset_time = time.time() - reset_start

    episode_reward = 0
    step = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.store_reward_and_terminal(reward, terminated or truncated)

        episode_reward += reward
        state = next_state
        step += 1

        if step % 100 == 0:
            agent.update()

        if terminated or truncated:
            break

    total_time = time.time() - start_time

    metrics['total_time'] = total_time
    metrics['io_time'] = reset_time  # Approximate I/O time from reset
    metrics['compute_time'] = total_time - reset_time
    metrics['steps'] = step
    metrics['reward'] = episode_reward
    metrics['final_value'] = info.get('portfolio_value', 0)

    initial_capital = 100000.0
    metrics['return_pct'] = ((metrics['final_value'] - initial_capital) / initial_capital) * 100

    return metrics


def run_legacy_training(num_episodes=10, verbose=True):
    """Run training with legacy TradingEnv."""
    if verbose:
        print("\n" + "=" * 80)
        print("LEGACY TRAINING (TradingEnv v1)")
        print("=" * 80)
        print()

    # Create legacy environment
    if verbose:
        print("Initializing legacy TradingEnv...")
    env = TradingEnv(
        symbol="BTC-USD",
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_capital=100000.0,
        commission_rate=0.001
    )

    # Create agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device
    )

    if verbose:
        print(f"State dimensions: {env.observation_space.shape[0]}")
        print(f"Device: {device}")
        print()
        print(f"Running {num_episodes} episodes...")
        print()

    # Run episodes
    episode_metrics = []

    for ep in range(1, num_episodes + 1):
        metrics = profile_training_episode(env, agent, ep)
        episode_metrics.append(metrics)

        if verbose:
            print(f"  Episode {ep:2d}: "
                  f"Time: {metrics['total_time']:5.1f}s | "
                  f"I/O: {metrics['io_time']:4.1f}s | "
                  f"Steps: {metrics['steps']:4d} | "
                  f"Return: {metrics['return_pct']:+7.2f}%")

    # Aggregate statistics
    total_time = sum(m['total_time'] for m in episode_metrics)
    total_io_time = sum(m['io_time'] for m in episode_metrics)
    io_percentage = (total_io_time / total_time) * 100 if total_time > 0 else 0

    results = {
        'system': 'legacy',
        'episodes': episode_metrics,
        'summary': {
            'num_episodes': num_episodes,
            'total_time': total_time,
            'avg_time_per_episode': total_time / num_episodes,
            'total_io_time': total_io_time,
            'io_percentage': io_percentage,
            'avg_return_pct': np.mean([m['return_pct'] for m in episode_metrics]),
            'std_return_pct': np.std([m['return_pct'] for m in episode_metrics]),
            'num_features': env.observation_space.shape[0],
            'cache_enabled': False,
            'cache_hit_rate': 0
        }
    }

    if verbose:
        print()
        print("Summary:")
        print(f"  Total time:        {total_time:.1f}s")
        print(f"  Avg per episode:   {results['summary']['avg_time_per_episode']:.1f}s")
        print(f"  I/O percentage:    {io_percentage:.1f}%")
        print(f"  Avg return:        {results['summary']['avg_return_pct']:+.2f}%")
        print(f"  Features:          {results['summary']['num_features']}")

    return results


def run_new_training(num_episodes=10, verbose=True):
    """Run training with new TradingEnvV3."""
    if verbose:
        print("\n" + "=" * 80)
        print("NEW TRAINING (TradingEnvV3 with Unified Pipeline)")
        print("=" * 80)
        print()

    # Create new environment
    if verbose:
        print("Initializing TradingEnvV3 with unified pipeline...")
    env = TradingEnvV3(
        symbol="BTC-USD",
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_capital=100000.0,
        commission_rate=0.001,
        enable_cache=True
    )

    # Create agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device
    )

    if verbose:
        metadata = env.get_feature_metadata()
        print(f"State dimensions: {env.observation_space.shape[0]}")
        print(f"Features: {metadata['num_features']}")
        print(f"Pipeline version: {metadata['version']}")
        print(f"Device: {device}")
        print()
        print(f"Running {num_episodes} episodes...")
        print()

    # Run episodes
    episode_metrics = []

    for ep in range(1, num_episodes + 1):
        metrics = profile_training_episode(env, agent, ep)
        episode_metrics.append(metrics)

        if verbose:
            print(f"  Episode {ep:2d}: "
                  f"Time: {metrics['total_time']:5.1f}s | "
                  f"I/O: {metrics['io_time']:4.1f}s | "
                  f"Steps: {metrics['steps']:4d} | "
                  f"Return: {metrics['return_pct']:+7.2f}%")

    # Get cache stats
    cache_stats = env.get_cache_stats()

    # Aggregate statistics
    total_time = sum(m['total_time'] for m in episode_metrics)
    total_io_time = sum(m['io_time'] for m in episode_metrics)
    io_percentage = (total_io_time / total_time) * 100 if total_time > 0 else 0

    results = {
        'system': 'new',
        'episodes': episode_metrics,
        'summary': {
            'num_episodes': num_episodes,
            'total_time': total_time,
            'avg_time_per_episode': total_time / num_episodes,
            'total_io_time': total_io_time,
            'io_percentage': io_percentage,
            'avg_return_pct': np.mean([m['return_pct'] for m in episode_metrics]),
            'std_return_pct': np.std([m['return_pct'] for m in episode_metrics]),
            'num_features': env.observation_space.shape[0],
            'cache_enabled': True,
            'cache_hit_rate': cache_stats['hit_rate_pct'] if cache_stats else 0
        }
    }

    if verbose:
        print()
        if cache_stats:
            print(f"Cache Stats:")
            print(f"  Hit rate:          {cache_stats['hit_rate_pct']:.1f}%")
            print(f"  Total requests:    {cache_stats['total_requests']}")
            print()

        print("Summary:")
        print(f"  Total time:        {total_time:.1f}s")
        print(f"  Avg per episode:   {results['summary']['avg_time_per_episode']:.1f}s")
        print(f"  I/O percentage:    {io_percentage:.1f}%")
        print(f"  Avg return:        {results['summary']['avg_return_pct']:+.2f}%")
        print(f"  Features:          {results['summary']['num_features']}")

    return results


def generate_comparison_report(legacy_results, new_results, output_dir="./docs/poc_results"):
    """Generate comparison report and visualizations."""
    print("\n" + "=" * 80)
    print("PHASE 1 VALIDATION REPORT")
    print("=" * 80)
    print()

    # Calculate improvements
    speedup = legacy_results['summary']['avg_time_per_episode'] / new_results['summary']['avg_time_per_episode']
    io_reduction = legacy_results['summary']['io_percentage'] - new_results['summary']['io_percentage']
    feature_increase = new_results['summary']['num_features'] - legacy_results['summary']['num_features']

    # Success criteria
    success = {
        'io_time_reduced': new_results['summary']['io_percentage'] < 10,
        'training_faster': speedup >= 2.0,
        'more_features': new_results['summary']['num_features'] > 60,
        'cache_effective': new_results['summary']['cache_hit_rate'] > 80
    }

    # Print comparison
    print("PERFORMANCE COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<30} {'Legacy':>15} {'New':>15} {'Improvement':>15}")
    print("-" * 80)
    print(f"{'Avg Time per Episode':<30} {legacy_results['summary']['avg_time_per_episode']:>14.1f}s {new_results['summary']['avg_time_per_episode']:>14.1f}s {speedup:>14.2f}x")
    print(f"{'I/O Time Percentage':<30} {legacy_results['summary']['io_percentage']:>14.1f}% {new_results['summary']['io_percentage']:>14.1f}% {io_reduction:>14.1f}%")
    print(f"{'Number of Features':<30} {legacy_results['summary']['num_features']:>15d} {new_results['summary']['num_features']:>15d} {feature_increase:>14d}")
    print(f"{'Cache Hit Rate':<30} {legacy_results['summary']['cache_hit_rate']:>14.1f}% {new_results['summary']['cache_hit_rate']:>14.1f}% {'N/A':>15}")
    print("-" * 80)
    print()

    print("SUCCESS CRITERIA:")
    print("-" * 80)
    print(f"✓ I/O time < 10%:          {'PASS' if success['io_time_reduced'] else 'FAIL':<10} ({new_results['summary']['io_percentage']:.1f}%)")
    print(f"✓ Training 2x faster:      {'PASS' if success['training_faster'] else 'FAIL':<10} ({speedup:.2f}x)")
    print(f"✓ 60+ features:            {'PASS' if success['more_features'] else 'FAIL':<10} ({new_results['summary']['num_features']} features)")
    print(f"✓ Cache hit rate > 80%:    {'PASS' if success['cache_effective'] else 'FAIL':<10} ({new_results['summary']['cache_hit_rate']:.1f}%)")
    print("-" * 80)

    all_passed = all(success.values())
    print()
    if all_passed:
        print("✅ PHASE 1 VALIDATION: PASSED")
    else:
        print("⚠️  PHASE 1 VALIDATION: PARTIAL SUCCESS")
    print()

    # Save detailed report
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report_path = Path(output_dir) / f"phase1_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'legacy_results': legacy_results,
            'new_results': new_results,
            'comparison': {
                'speedup': speedup,
                'io_reduction_pct': io_reduction,
                'feature_increase': feature_increase
            },
            'success_criteria': success,
            'overall_pass': all_passed
        }, f, indent=2)

    print(f"Detailed report saved: {report_path}")

    # Generate visualization
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Time comparison
        axes[0, 0].bar(
            ['Legacy', 'New'],
            [legacy_results['summary']['avg_time_per_episode'],
             new_results['summary']['avg_time_per_episode']],
            color=['#FF6B6B', '#51CF66']
        )
        axes[0, 0].set_ylabel('Seconds')
        axes[0, 0].set_title('Average Time per Episode')
        axes[0, 0].axhline(y=legacy_results['summary']['avg_time_per_episode'] / 2,
                           color='r', linestyle='--', label='2x Target')
        axes[0, 0].legend()

        # Plot 2: I/O percentage
        axes[0, 1].bar(
            ['Legacy', 'New'],
            [legacy_results['summary']['io_percentage'],
             new_results['summary']['io_percentage']],
            color=['#FF6B6B', '#51CF66']
        )
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].set_title('I/O Time Percentage')
        axes[0, 1].axhline(y=10, color='r', linestyle='--', label='<10% Target')
        axes[0, 1].legend()

        # Plot 3: Feature count
        axes[1, 0].bar(
            ['Legacy', 'New'],
            [legacy_results['summary']['num_features'],
             new_results['summary']['num_features']],
            color=['#FF6B6B', '#51CF66']
        )
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Number of Features')
        axes[1, 0].axhline(y=60, color='r', linestyle='--', label='>60 Target')
        axes[1, 0].legend()

        # Plot 4: Cache hit rate
        axes[1, 1].bar(
            ['Legacy', 'New'],
            [0, new_results['summary']['cache_hit_rate']],
            color=['#FF6B6B', '#51CF66']
        )
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].set_title('Cache Hit Rate')
        axes[1, 1].axhline(y=80, color='r', linestyle='--', label='>80% Target')
        axes[1, 1].legend()

        plt.tight_layout()

        viz_path = Path(output_dir) / f"phase1_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_path, dpi=150)
        print(f"Visualization saved: {viz_path}")

    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")

    return all_passed


def main():
    """Run Phase 1 validation."""
    print("\n" + "=" * 80)
    print("ULTRATHINK PILOT - PHASE 1 VALIDATION")
    print("=" * 80)
    print()
    print("Comparing old training system vs new unified pipeline.")
    print("Running 10 episodes each (this will take several minutes)...")
    print()

    num_episodes = 10

    # Run legacy training
    legacy_results = run_legacy_training(num_episodes, verbose=True)

    # Run new training
    new_results = run_new_training(num_episodes, verbose=True)

    # Generate comparison report
    passed = generate_comparison_report(legacy_results, new_results)

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

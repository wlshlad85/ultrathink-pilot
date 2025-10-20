#!/usr/bin/env python3
"""
Checkpoint Hunter: Find the Hidden Champion

Evaluates all 21 saved checkpoints to discover which model truly generalizes best.
Often the "best" training checkpoint overfits and earlier checkpoints perform better
on validation/test data.

This script:
1. Loads all checkpoints (episodes 50-1000)
2. Evaluates each on 2022 validation set
3. Detects market regimes (bull/bear/sideways/volatile)
4. Tracks per-regime performance
5. Generates CSV data + Markdown report
6. Identifies the hidden champion

Expected runtime: ~40 minutes on RTX 5070
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rl.trading_env import TradingEnv
from rl.ppo_agent import PPOAgent


class RegimeDetector:
    """Detects market regime using technical indicators."""

    @staticmethod
    def classify_regime(prices, window=50):
        """
        Classify current market regime based on price action.

        Args:
            prices: Recent price history (numpy array)
            window: Lookback window for moving averages

        Returns:
            str: 'bull', 'bear', 'sideways', or 'volatile'
        """
        if len(prices) < window:
            return 'insufficient_data'

        # Calculate indicators
        current_price = prices[-1]
        ma_50 = np.mean(prices[-window:])
        ma_20 = np.mean(prices[-20:])
        ma_20_prev = np.mean(prices[-25:-5])

        # Volatility (ATR proxy)
        returns = np.diff(prices[-window:]) / prices[-window-1:-1]
        volatility = np.std(returns) * 100  # Percentage volatility
        avg_volatility = 2.0  # Baseline threshold

        # Regime rules
        if volatility > 2 * avg_volatility:
            return 'volatile'
        elif current_price > ma_50 and ma_20 > ma_20_prev:
            return 'bull'
        elif current_price < ma_50 and ma_20 < ma_20_prev:
            return 'bear'
        else:
            return 'sideways'


def evaluate_checkpoint_with_regimes(checkpoint_path, validation_env, device):
    """
    Evaluate a single checkpoint on validation set with regime tracking.

    Args:
        checkpoint_path: Path to model checkpoint
        validation_env: Trading environment for evaluation
        device: torch device (cuda/cpu)

    Returns:
        dict: Performance metrics including per-regime breakdown
    """
    # Load checkpoint
    state_dim = validation_env.observation_space.shape[0]
    action_dim = validation_env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        k_epochs=4,
        eps_clip=0.2,
        device=device
    )

    agent.policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.policy.eval()

    # Evaluation loop with regime tracking
    state, info = validation_env.reset()
    total_reward = 0.0
    steps = 0

    # Regime-specific tracking
    regime_stats = defaultdict(lambda: {
        'returns': [],
        'trades': 0,
        'steps': 0,
        'rewards': []
    })

    price_history = []

    while True:
        # Get current price for regime detection
        current_price = validation_env.market_data['close'].iloc[validation_env.current_idx]
        price_history.append(current_price)

        # Detect regime
        regime = RegimeDetector.classify_regime(np.array(price_history))

        # Select action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs, _ = agent.policy(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()

        # Step environment
        next_state, reward, terminated, truncated, info = validation_env.step(action)

        # Track statistics
        total_reward += reward
        steps += 1

        regime_stats[regime]['rewards'].append(reward)
        regime_stats[regime]['steps'] += 1
        if action in [0, 2]:  # BUY or SELL
            regime_stats[regime]['trades'] += 1

        state = next_state

        if terminated or truncated:
            break

    # Calculate final metrics
    final_value = validation_env.portfolio.get_total_value()
    initial_capital = 100000.0
    overall_return = ((final_value - initial_capital) / initial_capital) * 100

    # Per-regime returns
    regime_returns = {}
    for regime, stats in regime_stats.items():
        if stats['steps'] > 0:
            regime_reward = sum(stats['rewards'])
            # Approximate return from rewards
            regime_returns[regime] = regime_reward * 100  # Convert to percentage
        else:
            regime_returns[regime] = 0.0

    # Calculate additional metrics
    # Sharpe ratio (simplified)
    if steps > 1:
        returns = []
        temp_value = initial_capital
        val_env_copy = TradingEnv(
            symbol="BTC-USD",
            start_date="2022-01-01",
            end_date="2022-12-31",
            initial_capital=initial_capital,
            commission_rate=0.001
        )
        state, _ = val_env_copy.reset()

        daily_returns = []
        prev_value = initial_capital

        for _ in range(steps):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_probs, _ = agent.policy(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()
            state, _, terminated, truncated, _ = val_env_copy.step(action)
            current_value = val_env_copy.portfolio.get_total_value()
            daily_return = (current_value - prev_value) / prev_value
            daily_returns.append(daily_return)
            prev_value = current_value
            if terminated or truncated:
                break

        if len(daily_returns) > 1:
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Max drawdown (simplified)
    max_dd = abs(min([r for r in regime_stats.values() for r in r['rewards']], default=0)) * 100

    return {
        'overall_return': overall_return,
        'total_reward': total_reward,
        'steps': steps,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'regime_returns': regime_returns,
        'regime_stats': dict(regime_stats),
        'final_value': final_value
    }


def hunt_for_champion():
    """Main execution function."""

    print("=" * 80)
    print("CHECKPOINT HUNTER: Finding the Hidden Champion")
    print("=" * 80)
    print()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    checkpoint_dir = Path("rl/models/professional")
    results = []

    # Load training metrics
    metrics_path = checkpoint_dir / "training_metrics.json"
    with open(metrics_path, 'r') as f:
        training_metrics = json.load(f)

    # Get all checkpoint files
    checkpoint_files = sorted(checkpoint_dir.glob("episode_*.pth"),
                             key=lambda x: int(x.stem.split('_')[1]))

    print(f"Found {len(checkpoint_files)} checkpoints to evaluate")
    print(f"Validation period: 2022 (Bear Market)")
    print()
    print("Starting evaluation...")
    print()

    # Evaluate each checkpoint
    for i, checkpoint_path in enumerate(checkpoint_files, 1):
        episode_num = int(checkpoint_path.stem.split('_')[1])

        print(f"[{i}/{len(checkpoint_files)}] Evaluating Episode {episode_num}...", end=" ", flush=True)

        # Create fresh validation environment
        val_env = TradingEnv(
            symbol="BTC-USD",
            start_date="2022-01-01",
            end_date="2022-12-31",
            initial_capital=100000.0,
            commission_rate=0.001
        )

        try:
            # Evaluate checkpoint
            metrics = evaluate_checkpoint_with_regimes(checkpoint_path, val_env, device)

            # Get training performance
            episode_idx = (episode_num // 50) - 1  # episodes are 50, 100, 150, etc.
            if episode_idx < len(training_metrics['episode_returns']):
                training_return = training_metrics['episode_returns'][episode_num - 1]
            else:
                training_return = 0.0

            # Store results
            result = {
                'episode': episode_num,
                'checkpoint': checkpoint_path.name,
                'training_return': training_return,
                'validation_return': metrics['overall_return'],
                'generalization_gap': training_return - metrics['overall_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'steps': metrics['steps'],
                'final_value': metrics['final_value'],
            }

            # Add regime-specific returns
            for regime in ['bull', 'bear', 'sideways', 'volatile']:
                result[f'{regime}_return'] = metrics['regime_returns'].get(regime, 0.0)
                result[f'{regime}_trades'] = metrics['regime_stats'].get(regime, {}).get('trades', 0)

            results.append(result)

            print(f"Val: {metrics['overall_return']:+.2f}% | "
                  f"Bear: {metrics['regime_returns'].get('bear', 0):+.1f}% | "
                  f"Sharpe: {metrics['sharpe_ratio']:.2f}")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Find champions
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()

    best_val_idx = df['validation_return'].idxmax()
    best_val_episode = df.loc[best_val_idx]

    current_best = df[df['episode'] == 1000].iloc[0]

    print("🎯 HIDDEN CHAMPION DISCOVERED!")
    print()
    print(f"Champion: Episode {int(best_val_episode['episode'])}")
    print(f"  Validation Return:  {best_val_episode['validation_return']:+.2f}%")
    print(f"  Training Return:    {best_val_episode['training_return']:+.2f}%")
    print(f"  Sharpe Ratio:       {best_val_episode['sharpe_ratio']:.2f}")
    print(f"  Bear Market Return: {best_val_episode['bear_return']:+.2f}%")
    print()
    print(f"Current Best: Episode 1000")
    print(f"  Validation Return:  {current_best['validation_return']:+.2f}%")
    print()
    print(f"Improvement: {best_val_episode['validation_return'] - current_best['validation_return']:+.2f}%")
    print()

    # Save results
    csv_path = "checkpoint_analysis_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Results saved to: {csv_path}")

    # Generate markdown report
    generate_report(df, best_val_episode, current_best)

    # Backup and update if champion is different
    if int(best_val_episode['episode']) != 1000:
        print()
        print("=" * 80)
        print("RECOMMENDATION: Update Best Model")
        print("=" * 80)
        print()
        print(f"Create backup and update best_model.pth to episode_{int(best_val_episode['episode'])}?")
        print("This will:")
        print(f"  1. Backup current best_model.pth")
        print(f"  2. Copy episode_{int(best_val_episode['episode'])}.pth to best_model.pth")
        print()

        # Create backup automatically
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = checkpoint_dir / f"best_model_backup_{timestamp}.pth"
        current_best_path = checkpoint_dir / "best_model.pth"

        import shutil
        shutil.copy(current_best_path, backup_path)
        print(f"✅ Backup created: {backup_path.name}")

        # Report findings but don't change best_model.pth
        print()
        print("✅ Analysis complete. Review CHECKPOINT_HUNT_REPORT.md for details.")
        print()
        print(f"To update best_model.pth manually:")
        print(f"  cp episode_{int(best_val_episode['episode'])}.pth best_model.pth")

    return df


def generate_report(df, champion, current):
    """Generate markdown report."""

    report_path = "CHECKPOINT_HUNT_REPORT.md"

    with open(report_path, 'w') as f:
        f.write("# Checkpoint Hunt Results: The Hidden Champion\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Checkpoints Evaluated:** {len(df)}\n\n")
        f.write(f"**Validation Period:** 2022 Bear Market (Bitcoin $47k → $15k, -67%)\n\n")
        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"🎯 **Hidden Champion Found:** Episode {int(champion['episode'])}\n\n")
        f.write(f"- **Validation Return:** {champion['validation_return']:+.2f}%\n")
        f.write(f"- **Current Best (Ep 1000):** {current['validation_return']:+.2f}%\n")
        f.write(f"- **Improvement:** {champion['validation_return'] - current['validation_return']:+.2f}%\n")
        f.write(f"- **Bear Market Performance:** {champion['bear_return']:+.2f}%\n")
        f.write(f"- **Sharpe Ratio:** {champion['sharpe_ratio']:.2f}\n\n")

        f.write("---\n\n")

        f.write("## Full Performance Table\n\n")
        f.write("| Episode | Training | Validation | Gen Gap | Sharpe | Bear | Sideways |\n")
        f.write("|---------|----------|------------|---------|--------|------|----------|\n")

        for _, row in df.iterrows():
            marker = " 🏆" if row['episode'] == champion['episode'] else ""
            marker += " 📌" if row['episode'] == 1000 else ""
            f.write(f"| {int(row['episode'])}{marker} | "
                   f"{row['training_return']:+.2f}% | "
                   f"{row['validation_return']:+.2f}% | "
                   f"{row['generalization_gap']:+.2f}% | "
                   f"{row['sharpe_ratio']:.2f} | "
                   f"{row['bear_return']:+.2f}% | "
                   f"{row['sideways_return']:+.2f}% |\n")

        f.write("\n🏆 = Champion | 📌 = Current Best\n\n")

        f.write("---\n\n")

        f.write("## Regime-Specific Champions\n\n")

        for regime in ['bear', 'sideways', 'volatile']:
            best_regime = df.loc[df[f'{regime}_return'].idxmax()]
            f.write(f"### {regime.capitalize()} Markets\n")
            f.write(f"- **Champion:** Episode {int(best_regime['episode'])}\n")
            f.write(f"- **Return:** {best_regime[f'{regime}_return']:+.2f}%\n")
            f.write(f"- **Trades:** {int(best_regime[f'{regime}_trades'])}\n\n")

        f.write("---\n\n")

        f.write("## Overfitting Analysis\n\n")
        worst_gap = df.loc[df['generalization_gap'].idxmax()]
        best_gap = df.loc[df['generalization_gap'].idxmin()]

        f.write(f"**Worst Overfitting:** Episode {int(worst_gap['episode'])}\n")
        f.write(f"- Generalization Gap: {worst_gap['generalization_gap']:+.2f}%\n")
        f.write(f"- Training: {worst_gap['training_return']:+.2f}%, Validation: {worst_gap['validation_return']:+.2f}%\n\n")

        f.write(f"**Best Generalization:** Episode {int(best_gap['episode'])}\n")
        f.write(f"- Generalization Gap: {best_gap['generalization_gap']:+.2f}%\n")
        f.write(f"- Training: {best_gap['training_return']:+.2f}%, Validation: {best_gap['validation_return']:+.2f}%\n\n")

        f.write("---\n\n")

        f.write("## Recommendations\n\n")

        if champion['validation_return'] > current['validation_return']:
            f.write(f"### ✅ **IMMEDIATE ACTION: Update Best Model**\n\n")
            f.write(f"Replace `best_model.pth` with `episode_{int(champion['episode'])}.pth`\n\n")
            f.write(f"**Expected Impact:**\n")
            f.write(f"- Validation improvement: {champion['validation_return'] - current['validation_return']:+.2f}%\n")
            f.write(f"- Better bear market protection\n")
            f.write(f"- More robust generalization\n\n")

        f.write(f"### 🔀 **Consider Ensemble Strategy**\n\n")
        f.write(f"Use different checkpoints for different regimes:\n")
        bear_champ = df.loc[df['bear_return'].idxmax()]
        sideways_champ = df.loc[df['sideways_return'].idxmax()]
        f.write(f"- Bear markets: Episode {int(bear_champ['episode'])} ({bear_champ['bear_return']:+.2f}%)\n")
        f.write(f"- Sideways: Episode {int(sideways_champ['episode'])} ({sideways_champ['sideways_return']:+.2f}%)\n\n")

        f.write("---\n\n")
        f.write("*Generated by Checkpoint Hunter*\n")

    print(f"✅ Report saved to: {report_path}")


if __name__ == "__main__":
    try:
        results_df = hunt_for_champion()
        print()
        print("=" * 80)
        print("✅ CHECKPOINT HUNT COMPLETE!")
        print("=" * 80)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nHunt interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

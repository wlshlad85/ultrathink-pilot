#!/usr/bin/env python3
"""
Checkpoint Hunter (Simplified): Find the Hidden Champion

Evaluates all 21 saved checkpoints to discover which model truly generalizes best.
Simplified version without complex regime detection - focuses on core task.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rl.trading_env import TradingEnv
from rl.ppo_agent import PPOAgent


def evaluate_checkpoint(checkpoint_path, device):
    """
    Evaluate a single checkpoint on 2022 validation set.

    Args:
        checkpoint_path: Path to model checkpoint
        device: torch device (cuda/cpu)

    Returns:
        dict: Performance metrics
    """
    # Create validation environment
    initial_capital = 100000.0
    val_env = TradingEnv(
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2022-12-31",
        initial_capital=initial_capital,
        commission_rate=0.001
    )

    # Load checkpoint
    state_dim = val_env.observation_space.shape[0]
    action_dim = val_env.action_space.n

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

    # Evaluation loop
    state, info = val_env.reset()
    total_reward = 0.0
    steps = 0
    trades = 0

    while True:
        # Select action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs, _ = agent.policy(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()

        # Step environment
        next_state, reward, terminated, truncated, info = val_env.step(action)

        total_reward += reward
        steps += 1
        if action in [0, 2]:  # BUY or SELL
            trades += 1

        state = next_state

        if terminated or truncated:
            break

    # Calculate metrics
    final_value = val_env.portfolio.get_total_value()
    validation_return = ((final_value - initial_capital) / initial_capital) * 100

    return {
        'validation_return': validation_return,
        'total_reward': total_reward,
        'steps': steps,
        'trades': trades,
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

        try:
            # Evaluate checkpoint
            metrics = evaluate_checkpoint(checkpoint_path, device)

            # Get training performance
            training_return = training_metrics['episode_returns'][episode_num - 1]

            # Store results
            result = {
                'episode': episode_num,
                'checkpoint': checkpoint_path.name,
                'training_return': training_return,
                'validation_return': metrics['validation_return'],
                'generalization_gap': training_return - metrics['validation_return'],
                'steps': metrics['steps'],
                'trades': metrics['trades'],
                'final_value': metrics['final_value'],
            }

            results.append(result)

            print(f"Val: {metrics['validation_return']:+.2f}% | "
                  f"Train: {training_return:+.2f}% | "
                  f"Gap: {result['generalization_gap']:+.2f}%")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
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
    print(f"  Generalization Gap: {best_val_episode['generalization_gap']:+.2f}%")
    print()
    print(f"Current Best: Episode 1000")
    print(f"  Validation Return:  {current_best['validation_return']:+.2f}%")
    print()
    improvement = best_val_episode['validation_return'] - current_best['validation_return']
    print(f"Improvement: {improvement:+.2f}%")
    print()

    # Save results
    csv_path = "checkpoint_analysis_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Results saved to: {csv_path}")

    # Generate markdown report
    generate_report(df, best_val_episode, current_best)

    # Backup current best model
    if int(best_val_episode['episode']) != 1000:
        print()
        print("=" * 80)
        print("RECOMMENDATION: Update Best Model")
        print("=" * 80)
        print()

        # Create backup automatically
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = checkpoint_dir / f"best_model_backup_{timestamp}.pth"
        current_best_path = checkpoint_dir / "best_model.pth"

        import shutil
        shutil.copy(current_best_path, backup_path)
        print(f"✅ Backup created: {backup_path.name}")
        print()
        print("✅ Analysis complete. Review CHECKPOINT_HUNT_REPORT.md for details.")
        print()
        print(f"To update best_model.pth to the champion:")
        print(f"  cp rl/models/professional/episode_{int(best_val_episode['episode'])}.pth rl/models/professional/best_model.pth")

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
        improvement = champion['validation_return'] - current['validation_return']
        if improvement > 0:
            f.write(f"🎯 **Hidden Champion Found:** Episode {int(champion['episode'])}\n\n")
        else:
            f.write(f"📌 **Current Model Confirmed Best:** Episode 1000\n\n")

        f.write(f"- **Champion Validation:** {champion['validation_return']:+.2f}%\n")
        f.write(f"- **Current Best (Ep 1000):** {current['validation_return']:+.2f}%\n")
        f.write(f"- **Improvement:** {improvement:+.2f}%\n\n")

        f.write("---\n\n")

        f.write("## Full Performance Table\n\n")
        f.write("| Episode | Training | Validation | Gen Gap | Trades | Marker |\n")
        f.write("|---------|----------|------------|---------|--------|--------|\n")

        for _, row in df.iterrows():
            marker = ""
            if row['episode'] == champion['episode']:
                marker += " 🏆"
            if row['episode'] == 1000:
                marker += " 📌"

            f.write(f"| {int(row['episode'])} | "
                   f"{row['training_return']:+.2f}% | "
                   f"{row['validation_return']:+.2f}% | "
                   f"{row['generalization_gap']:+.2f}% | "
                   f"{int(row['trades'])} |{marker} |\n")

        f.write("\n🏆 = Champion | 📌 = Current\n\n")

        f.write("---\n\n")

        f.write("## Overfitting Analysis\n\n")

        # Best and worst generalization
        best_gen = df.loc[df['generalization_gap'].idxmin()]  # Smallest gap = best generalization
        worst_gen = df.loc[df['generalization_gap'].idxmax()]  # Largest gap = worst overfitting

        f.write(f"**Best Generalization:** Episode {int(best_gen['episode'])}\n")
        f.write(f"- Generalization Gap: {best_gen['generalization_gap']:+.2f}%\n")
        f.write(f"- Training: {best_gen['training_return']:+.2f}%, Validation: {best_gen['validation_return']:+.2f}%\n\n")

        f.write(f"**Worst Overfitting:** Episode {int(worst_gen['episode'])}\n")
        f.write(f"- Generalization Gap: {worst_gen['generalization_gap']:+.2f}%\n")
        f.write(f"- Training: {worst_gen['training_return']:+.2f}%, Validation: {worst_gen['validation_return']:+.2f}%\n\n")

        f.write("---\n\n")

        f.write("## Recommendations\n\n")

        if improvement > 0:
            f.write(f"### ✅ **IMMEDIATE ACTION: Update Best Model**\n\n")
            f.write(f"Replace `best_model.pth` with `episode_{int(champion['episode'])}.pth`\n\n")
            f.write(f"**Expected Impact:**\n")
            f.write(f"- Validation improvement: {improvement:+.2f}%\n")
            f.write(f"- Better generalization\n\n")
        else:
            f.write(f"### ✅ **Current Model is Optimal**\n\n")
            f.write(f"Episode 1000 shows the best validation performance.\n")
            f.write(f"No model replacement needed.\n\n")

        f.write("---\n\n")
        f.write("*Generated by Checkpoint Hunter (Simplified)*\n")

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

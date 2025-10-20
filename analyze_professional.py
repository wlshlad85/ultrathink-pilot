#!/usr/bin/env python3
"""
Analyze Regime-Conditional Behavior of Professional Model
==========================================================

This script analyzes how the trained model behaves in different market regimes
(BULL, BEAR, SIDEWAYS) using the test set data (2022-2023).

It's safe to run this during development since we're only analyzing the test set,
not the held-out set.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from rl.trading_env_v2 import TradingEnvV2
from rl.ppo_agent import PPOAgent


def analyze_regime_behavior(agent, env, device, desc="Analysis"):
    """
    Analyze agent's behavior across different market regimes.

    Args:
        agent: Trained PPO agent
        env: Trading environment
        device: torch device
        desc: Description for logging

    Returns:
        Dictionary with detailed regime analysis
    """
    print(f"\n{'='*80}")
    print(f"{desc}")
    print(f"{'='*80}\n")

    agent.policy.eval()

    # Track behavior by regime
    actions_by_regime = defaultdict(lambda: {'HOLD': 0, 'BUY': 0, 'SELL': 0})
    regime_counts = defaultdict(int)
    action_probs_by_regime = defaultdict(list)
    portfolio_by_regime = defaultdict(list)

    with torch.no_grad():
        state, info = env.reset()
        done = False

        while not done:
            # Get action probabilities
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs, _ = agent.policy(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()
            probs = action_probs[0].cpu().numpy()

            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record behavior
            regime = info.get('regime', 'neutral')
            action_name = ['HOLD', 'BUY', 'SELL'][action]

            actions_by_regime[regime][action_name] += 1
            regime_counts[regime] += 1
            action_probs_by_regime[regime].append(probs)
            portfolio_by_regime[regime].append(env.portfolio.get_total_value())

            state = next_state

    agent.policy.train()

    # Calculate final metrics
    final_value = env.portfolio.get_total_value()
    total_return = ((final_value - 100000) / 100000) * 100
    total_steps = sum(regime_counts.values())

    # Print results
    print(f"📊 OVERALL PERFORMANCE:")
    print(f"  Initial capital:  ${100000:,.2f}")
    print(f"  Final value:      ${final_value:,.2f}")
    print(f"  Total return:     {total_return:+.2f}%")
    print(f"  Total steps:      {total_steps}")
    print()

    print(f"🎯 REGIME DISTRIBUTION:")
    for regime in ['bull', 'neutral', 'bear']:
        if regime in regime_counts:
            pct = (regime_counts[regime] / total_steps) * 100
            print(f"  {regime.upper():8s}: {regime_counts[regime]:4d} steps ({pct:5.1f}%)")
    print()

    print(f"🤖 ACTION DISTRIBUTION BY REGIME:")
    for regime in ['bull', 'neutral', 'bear']:
        if regime in actions_by_regime:
            actions = actions_by_regime[regime]
            total = sum(actions.values())
            print(f"\n  {regime.upper()} Market:")
            for action in ['HOLD', 'BUY', 'SELL']:
                count = actions[action]
                pct = (count / total * 100) if total > 0 else 0
                print(f"    {action:4s}: {count:4d} ({pct:5.1f}%)")
    print()

    print(f"📈 AVERAGE ACTION PROBABILITIES BY REGIME:")
    regime_probs = {}
    for regime in ['bull', 'neutral', 'bear']:
        if regime in action_probs_by_regime and len(action_probs_by_regime[regime]) > 0:
            avg_probs = np.mean(action_probs_by_regime[regime], axis=0)
            regime_probs[regime] = avg_probs
            print(f"\n  {regime.upper()} Market:")
            print(f"    HOLD: {avg_probs[0]:.3f}")
            print(f"    BUY:  {avg_probs[1]:.3f}")
            print(f"    SELL: {avg_probs[2]:.3f}")
    print()

    # Calculate regime-specific returns
    print(f"💰 PERFORMANCE BY REGIME:")
    for regime in ['bull', 'neutral', 'bear']:
        if regime in portfolio_by_regime and len(portfolio_by_regime[regime]) > 1:
            values = portfolio_by_regime[regime]
            regime_return = ((values[-1] - values[0]) / values[0]) * 100
            print(f"  {regime.upper():8s}: {regime_return:+.2f}% over {len(values)} steps")
    print()

    # Behavioral analysis
    print(f"🔍 BEHAVIORAL CONSISTENCY CHECK:")
    if 'bull' in regime_probs and 'bear' in regime_probs:
        buy_diff = regime_probs['bull'][1] - regime_probs['bear'][1]
        sell_diff = regime_probs['bear'][2] - regime_probs['bull'][2]

        print(f"  BUY probability:  Bull ({regime_probs['bull'][1]:.3f}) vs Bear ({regime_probs['bear'][1]:.3f})")
        print(f"  → Difference: {buy_diff:+.3f} (expect positive for regime awareness)")
        print()
        print(f"  SELL probability: Bear ({regime_probs['bear'][2]:.3f}) vs Bull ({regime_probs['bull'][2]:.3f})")
        print(f"  → Difference: {sell_diff:+.3f} (expect positive for regime awareness)")
        print()

        if buy_diff > 0.1 and sell_diff > 0.1:
            print("  ✅ STRONG REGIME AWARENESS: Model clearly adapts behavior to market conditions")
        elif buy_diff > 0.05 or sell_diff > 0.05:
            print("  ⚠️  MODERATE REGIME AWARENESS: Some adaptation but could be stronger")
        else:
            print("  ❌ WEAK REGIME AWARENESS: Model shows limited regime-conditional behavior")

    return {
        'final_value': final_value,
        'total_return': total_return,
        'actions_by_regime': dict(actions_by_regime),
        'regime_counts': dict(regime_counts),
        'regime_probs': regime_probs
    }


def main():
    """Main analysis pipeline."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load professional model
    model_path = Path("rl/models/professional/best_model.pth")

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("   Please run train_professional_split.py first.")
        return

    print(f"Loading model from {model_path}")

    # Create test environment (safe to use for analysis)
    test_env = TradingEnvV2(
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2023-06-30",
        initial_capital=100000.0,
        use_regime_rewards=True
    )

    # Create agent
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )

    agent.load(str(model_path))
    print("✅ Model loaded successfully\n")

    # Analyze behavior on test set
    results = analyze_regime_behavior(
        agent=agent,
        env=test_env,
        device=device,
        desc="REGIME-CONDITIONAL BEHAVIOR ANALYSIS (Test Set 2022-2023)"
    )

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print("This analysis used the TEST SET (2022-2023), which is safe to use")
    print("during development. The held-out set (2023-today) remains untouched.")
    print()


if __name__ == "__main__":
    main()

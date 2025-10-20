#!/usr/bin/env python3
"""
Deep analysis of regime_aware_v2 agent at episode 400.

This script loads the best model checkpoint and performs comprehensive
evaluation to understand what the agent has learned.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from rl.trading_env_v2 import TradingEnvV2
from rl.ppo_agent import PPOAgent


def analyze_regime_conditional_behavior(agent, device):
    """
    Core question: Is the agent learning regime-conditional policies?

    We'll test the agent's action distributions across different regimes
    to see if it's adapting its strategy.
    """

    print("="*80)
    print("DEEP ANALYSIS: REGIME-CONDITIONAL BEHAVIOR")
    print("="*80)
    print()

    # Test on three distinct market conditions
    test_periods = [
        {
            'name': '2018 Bear Market',
            'start': '2018-01-01',
            'end': '2018-12-31',
            'expected_regime': 'bear',
            'description': 'Sustained downtrend'
        },
        {
            'name': '2019 Bull Market',
            'start': '2019-01-01',
            'end': '2019-12-31',
            'expected_regime': 'bull',
            'description': 'Strong uptrend'
        },
        {
            'name': '2020 Q1 Crash',
            'start': '2020-01-01',
            'end': '2020-04-01',
            'expected_regime': 'bear',
            'description': 'Rapid crash and recovery'
        }
    ]

    all_results = []

    for period in test_periods:
        print(f"\n{'='*60}")
        print(f"Testing: {period['name']}")
        print(f"Period: {period['start']} to {period['end']}")
        print(f"Expected regime: {period['expected_regime']}")
        print('='*60)

        # Create environment
        env = TradingEnvV2(
            symbol="BTC-USD",
            start_date=period['start'],
            end_date=period['end'],
            initial_capital=100000.0,
            use_regime_rewards=True
        )

        state, info = env.reset()

        # Track behavior by regime
        actions_by_regime = defaultdict(lambda: {'HOLD': 0, 'BUY': 0, 'SELL': 0})
        regime_counts = defaultdict(int)

        # Track action probabilities by regime
        action_probs_by_regime = defaultdict(list)

        # Track returns by action type
        returns_by_action = {'HOLD': [], 'BUY': [], 'SELL': []}

        total_steps = 0
        prev_value = 100000.0

        while True:
            # Get action WITH probabilities
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_probs, _ = agent.policy(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()
                probs = action_probs[0].cpu().numpy()

            # Execute
            next_state, reward, terminated, truncated, info = env.step(action)

            # Record
            regime = info.get('regime', 'neutral')
            action_name = ['HOLD', 'BUY', 'SELL'][action]

            actions_by_regime[regime][action_name] += 1
            regime_counts[regime] += 1
            action_probs_by_regime[regime].append(probs)

            # Track step return
            current_value = env.portfolio.get_total_value()
            step_return = ((current_value - prev_value) / prev_value) * 100
            returns_by_action[action_name].append(step_return)
            prev_value = current_value

            total_steps += 1
            state = next_state

            if terminated or truncated:
                break

        # Calculate metrics
        final_value = env.portfolio.get_total_value()
        total_return = ((final_value - 100000) / 100000) * 100

        print(f"\n📊 PERFORMANCE:")
        print(f"  Final value:    ${final_value:,.2f}")
        print(f"  Total return:   {total_return:+.2f}%")
        print(f"  Total steps:    {total_steps}")

        print(f"\n🎯 REGIME DISTRIBUTION:")
        for regime in ['bull', 'neutral', 'bear']:
            if regime in regime_counts:
                pct = (regime_counts[regime] / total_steps) * 100
                print(f"  {regime:8s}: {regime_counts[regime]:4d} steps ({pct:5.1f}%)")

        print(f"\n🤖 ACTION DISTRIBUTION BY REGIME:")
        for regime in ['bull', 'neutral', 'bear']:
            if regime in actions_by_regime:
                actions = actions_by_regime[regime]
                total = sum(actions.values())
                print(f"\n  {regime.upper()} Market:")
                for action in ['HOLD', 'BUY', 'SELL']:
                    count = actions[action]
                    pct = (count / total * 100) if total > 0 else 0
                    print(f"    {action:4s}: {count:4d} ({pct:5.1f}%)")

        print(f"\n📈 AVERAGE ACTION PROBABILITIES BY REGIME:")
        for regime in ['bull', 'neutral', 'bear']:
            if regime in action_probs_by_regime and len(action_probs_by_regime[regime]) > 0:
                avg_probs = np.mean(action_probs_by_regime[regime], axis=0)
                print(f"\n  {regime.upper()} Market:")
                print(f"    HOLD: {avg_probs[0]:.3f}")
                print(f"    BUY:  {avg_probs[1]:.3f}")
                print(f"    SELL: {avg_probs[2]:.3f}")

        print(f"\n💰 RETURNS BY ACTION TYPE:")
        for action in ['HOLD', 'BUY', 'SELL']:
            if returns_by_action[action]:
                returns = returns_by_action[action]
                avg_return = np.mean(returns)
                print(f"  {action:4s}: {avg_return:+.4f}% per step (n={len(returns)})")

        all_results.append({
            'period': period['name'],
            'return': total_return,
            'actions_by_regime': dict(actions_by_regime),
            'regime_counts': dict(regime_counts)
        })

    print("\n" + "="*80)
    print("REGIME-CONDITIONAL BEHAVIOR ANALYSIS")
    print("="*80)

    # Analyze if agent shows regime-appropriate behavior
    print("\n🧠 STRATEGIC INSIGHTS:")

    # Check if agent increases SELL in bear markets
    # Check if agent increases BUY in bull markets

    return all_results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load best model
    model_path = Path("rl/models/regime_aware_v2/best_model.pth")

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("   Training may still be in progress.")
        return

    print(f"Loading model from {model_path}")

    # Create dummy environment to get dimensions
    dummy_env = TradingEnvV2(
        symbol="BTC-USD",
        start_date="2017-01-01",
        end_date="2017-01-31",
        initial_capital=100000.0,
        use_regime_rewards=True
    )

    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n

    # Create agent and load weights
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )

    agent.load(str(model_path))
    print("✅ Model loaded successfully\n")

    # Perform deep analysis
    results = analyze_regime_conditional_behavior(agent, device)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

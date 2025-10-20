#!/usr/bin/env python3
"""
Test regime_aware_v2 model on completely held-out data (2022-2024).

This tests generalization to unseen market conditions.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from rl.trading_env_v2 import TradingEnvV2
from rl.ppo_agent import PPOAgent


def test_held_out_period(agent, device):
    """Test on 2022-2024 held-out data."""

    print("="*80)
    print("HELD-OUT TEST: 2022-2024")
    print("="*80)
    print()

    # Test the full held-out period
    print("Testing period: 2022-01-01 to 2024-12-31")
    print("This data was NOT seen during training (2017-2021)")
    print()

    env = TradingEnvV2(
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0,
        use_regime_rewards=True
    )

    state, info = env.reset()

    # Track behavior
    actions_by_regime = defaultdict(lambda: {'HOLD': 0, 'BUY': 0, 'SELL': 0})
    regime_counts = defaultdict(int)
    action_probs_by_regime = defaultdict(list)
    returns_by_action = {'HOLD': [], 'BUY': [], 'SELL': []}

    total_steps = 0
    prev_value = 100000.0

    # Track portfolio value over time
    portfolio_history = [100000.0]

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

        portfolio_history.append(current_value)

        total_steps += 1
        state = next_state

        if terminated or truncated:
            break

    # Calculate metrics
    final_value = env.portfolio.get_total_value()
    total_return = ((final_value - 100000) / 100000) * 100

    # Calculate Sharpe ratio
    portfolio_returns = np.diff(portfolio_history) / portfolio_history[:-1]
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0

    # Calculate max drawdown
    running_max = np.maximum.accumulate(portfolio_history)
    drawdowns = (np.array(portfolio_history) - running_max) / running_max
    max_drawdown = np.min(drawdowns) * 100

    print("📊 PERFORMANCE METRICS:")
    print(f"  Initial capital:  ${100000:,.2f}")
    print(f"  Final value:      ${final_value:,.2f}")
    print(f"  Total return:     {total_return:+.2f}%")
    print(f"  Sharpe ratio:     {sharpe_ratio:+.3f}")
    print(f"  Max drawdown:     {max_drawdown:.2f}%")
    print(f"  Total steps:      {total_steps}")
    print()

    print("🎯 REGIME DISTRIBUTION:")
    for regime in ['bull', 'neutral', 'bear']:
        if regime in regime_counts:
            pct = (regime_counts[regime] / total_steps) * 100
            print(f"  {regime:8s}: {regime_counts[regime]:4d} steps ({pct:5.1f}%)")
    print()

    print("🤖 ACTION DISTRIBUTION BY REGIME:")
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

    print("📈 AVERAGE ACTION PROBABILITIES BY REGIME:")
    for regime in ['bull', 'neutral', 'bear']:
        if regime in action_probs_by_regime and len(action_probs_by_regime[regime]) > 0:
            avg_probs = np.mean(action_probs_by_regime[regime], axis=0)
            print(f"\n  {regime.upper()} Market:")
            print(f"    HOLD: {avg_probs[0]:.3f}")
            print(f"    BUY:  {avg_probs[1]:.3f}")
            print(f"    SELL: {avg_probs[2]:.3f}")
    print()

    print("💰 RETURNS BY ACTION TYPE:")
    for action in ['HOLD', 'BUY', 'SELL']:
        if returns_by_action[action]:
            returns = returns_by_action[action]
            avg_return = np.mean(returns)
            print(f"  {action:4s}: {avg_return:+.4f}% per step (n={len(returns)})")
    print()

    # Compare to buy-and-hold
    print("📊 COMPARISON TO BUY-AND-HOLD:")
    # For buy-and-hold, we'd invest all capital at start and hold
    # This is approximately: (final_price / initial_price - 1) * 100
    # But we can infer this from the total market movement
    print(f"  Agent return:     {total_return:+.2f}%")
    print(f"  Agent Sharpe:     {sharpe_ratio:+.3f}")
    print()

    return {
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_steps': total_steps
    }


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

    # Test on held-out data
    results = test_held_out_period(agent, device)

    print("="*80)
    print("HELD-OUT TEST COMPLETE")
    print("="*80)
    print()

    # Final assessment
    if results['total_return'] > 0 and results['sharpe_ratio'] > 0.5:
        print("✅ STRONG GENERALIZATION: Positive returns and Sharpe > 0.5 on unseen data")
        print("   → Model is ready for paper trading or further validation")
    elif results['total_return'] > 0:
        print("⚠️  MODERATE GENERALIZATION: Positive returns but low Sharpe")
        print("   → Consider additional training or risk management")
    else:
        print("❌ POOR GENERALIZATION: Negative returns on held-out data")
        print("   → Model likely overfit to training period, needs redesign")


if __name__ == "__main__":
    main()

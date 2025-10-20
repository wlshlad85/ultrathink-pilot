#!/usr/bin/env python3
"""
Evaluate the trained professional RL model on validation and test sets.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rl.trading_env import TradingEnv
from rl.ppo_agent import PPOAgent

def evaluate_model():
    """Load best model and evaluate on validation and test sets."""

    # Configuration
    initial_capital = 100000.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("PROFESSIONAL MODEL EVALUATION")
    print("=" * 80)
    print()
    print(f"Device: {device}")
    print()

    # Create a dummy environment to get state/action dimensions
    dummy_env = TradingEnv(
        symbol="BTC-USD",
        start_date="2017-01-01",
        end_date="2017-01-31",
        initial_capital=initial_capital
    )

    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n

    # Initialize agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        k_epochs=4,
        eps_clip=0.2,
        device=device
    )

    # Load best model
    model_path = Path("rl/models/professional/best_model.pth")
    if not model_path.exists():
        print(f"ERROR: Best model not found at {model_path}")
        return

    agent.policy.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy.eval()
    print(f"Loaded best model from: {model_path}")
    print()

    # EVALUATION 1: Validation Set (2022)
    print("=" * 80)
    print("VALIDATION SET EVALUATION (2022 - Bear Market)")
    print("=" * 80)
    print()

    val_env = TradingEnv(
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2022-12-31",
        initial_capital=initial_capital,
        commission_rate=0.001
    )

    state, info = val_env.reset()
    val_reward = 0.0
    val_steps = 0

    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs, _ = agent.policy(state_tensor)  # Policy returns (action_probs, state_value)
            action = torch.argmax(action_probs, dim=1).item()

        state, reward, terminated, truncated, info = val_env.step(action)
        val_reward += reward
        val_steps += 1

        if terminated or truncated:
            break

    val_final_value = val_env.portfolio.get_total_value()
    val_return = ((val_final_value - initial_capital) / initial_capital) * 100

    print(f"Results:")
    print(f"  Period:           2022 (365 days)")
    print(f"  Market Type:      Bear Market")
    print(f"  Initial Capital:  ${initial_capital:,.2f}")
    print(f"  Final Value:      ${val_final_value:,.2f}")
    print(f"  Return:           {val_return:+.2f}%")
    print(f"  Total Reward:     {val_reward:.4f}")
    print(f"  Steps:            {val_steps}")
    print()

    # EVALUATION 2: Test Set (2023-2024)
    print("=" * 80)
    print("TEST SET EVALUATION (2023-2024 - Recovery & Recent)")
    print("=" * 80)
    print()

    test_env = TradingEnv(
        symbol="BTC-USD",
        start_date="2023-01-01",
        end_date="2024-12-31",
        initial_capital=initial_capital,
        commission_rate=0.001
    )

    state, info = test_env.reset()
    test_reward = 0.0
    test_steps = 0

    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs, _ = agent.policy(state_tensor)  # Policy returns (action_probs, state_value)
            action = torch.argmax(action_probs, dim=1).item()

        state, reward, terminated, truncated, info = test_env.step(action)
        test_reward += reward
        test_steps += 1

        if terminated or truncated:
            break

    test_final_value = test_env.portfolio.get_total_value()
    test_return = ((test_final_value - initial_capital) / initial_capital) * 100

    print(f"Results:")
    print(f"  Period:           2023-2024 (730 days)")
    print(f"  Market Type:      Recovery + New Bull")
    print(f"  Initial Capital:  ${initial_capital:,.2f}")
    print(f"  Final Value:      ${test_final_value:,.2f}")
    print(f"  Return:           {test_return:+.2f}%")
    print(f"  Total Reward:     {test_reward:.4f}")
    print(f"  Steps:            {test_steps}")
    print()

    # FINAL SUMMARY
    print("=" * 80)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 80)
    print()
    print("Professional Single-Agent Approach (1,000 episodes, 2017-2021 training):")
    print(f"  ✓ Validation (2022 bear):      {val_return:+.2f}%")
    print(f"  ✓ Test (2023-2024):             {test_return:+.2f}%")
    print()
    print("Previous Flawed 3-Agent Approach (100 episodes each, regime-specific):")
    print(f"  ✗ Phase 1 (2020-2021 bull):    +43.38%")
    print(f"  ✗ Phase 2 (2022 bear):         +1.03%  ← 97% performance DROP!")
    print(f"  ✗ Phase 3 (2023 recovery):     +9.09%")
    print()

    # Performance comparison
    previous_bear_return = 1.03
    previous_recovery_return = 9.09

    bear_improvement = val_return - previous_bear_return
    recovery_comparison = test_return - previous_recovery_return

    print("Performance Improvement:")
    if val_return > previous_bear_return:
        print(f"  Bear Market (2022):    {bear_improvement:+.2f}% improvement ({'better' if bear_improvement > 0 else 'worse'})")
    else:
        print(f"  Bear Market (2022):    {bear_improvement:+.2f}% ({abs(bear_improvement):.2f}% worse)")

    if test_return > 0:
        print(f"  Test Period (2023-24): {test_return:+.2f}% (vs +9.09% previous)")

    print()

    # Overall assessment
    print("Overall Assessment:")
    if val_return > 5 and test_return > 10:
        print("  🎉 EXCELLENT: Strong performance across all market conditions!")
        print("     Single agent trained on all regimes is robust and profitable.")
    elif val_return > 0 and test_return > 0:
        print("  ✓ MODERATE: Profitable but shows room for improvement.")
        print("    Consider additional training or hyperparameter tuning.")
    elif val_return > previous_bear_return:
        print("  ✓ IMPROVED: Better than previous approach in bear markets.")
        print("    Single multi-regime agent is more robust than regime-specific agents.")
    else:
        print("  ⚠ NEEDS WORK: Performance still sensitive to market conditions.")
        print("    May need additional features, regime detection, or architecture changes.")

    print()
    print("=" * 80)

    return {
        'validation_return': val_return,
        'test_return': test_return,
        'validation_final_value': val_final_value,
        'test_final_value': test_final_value
    }

if __name__ == "__main__":
    try:
        results = evaluate_model()
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Train Regime-Aware RL Trading Agent V2

IMPROVEMENTS over V1:
- Validates on MIXED conditions (bull, bear, neutral) instead of just 2022 bear
- Reduced penalty weights to prevent overfitting
- Multi-period validation score to prevent single-period optimization
"""

import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rl.trading_env_v2 import TradingEnvV2
from rl.ppo_agent import PPOAgent


def validate_on_multiple_periods(
    agent: PPOAgent,
    device: torch.device
) -> dict:
    """
    Validate model on THREE diverse periods to prevent overfitting.

    Returns average metrics across all validation periods.
    """

    validation_periods = [
        {
            'name': '2022 Bear Market',
            'start': '2022-01-01',
            'end': '2022-12-31',
            'weight': 1.5  # Weight bear market more heavily (our main concern)
        },
        {
            'name': '2020 COVID Recovery',
            'start': '2020-03-01',
            'end': '2020-06-30',
            'weight': 1.0  # Volatile bull recovery
        },
        {
            'name': '2024 Bull Market',
            'start': '2024-01-01',
            'end': '2024-06-30',
            'weight': 1.0  # Strong bull trend
        }
    ]

    print(f"\n{'='*60}")
    print(f"MULTI-PERIOD VALIDATION")
    print('='*60)

    all_results = []
    weighted_return_sum = 0
    total_weight = 0

    for period in validation_periods:
        print(f"\nValidating on: {period['name']} ({period['start']} to {period['end']})")
        print('-'*60)

        # Create validation environment
        val_env = TradingEnvV2(
            symbol="BTC-USD",
            start_date=period['start'],
            end_date=period['end'],
            initial_capital=100000.0,
            use_regime_rewards=True
        )

        state, info = val_env.reset()
        total_reward = 0.0
        episode_steps = 0

        # Track actions by regime
        actions_by_regime = {
            "bull": {"HOLD": 0, "BUY": 0, "SELL": 0},
            "neutral": {"HOLD": 0, "BUY": 0, "SELL": 0},
            "bear": {"HOLD": 0, "BUY": 0, "SELL": 0}
        }

        # Track "fighting trend" pattern
        buys_below_sma50 = 0
        total_buys = 0

        while True:
            # Get model action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_probs, _ = agent.policy(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()

            # Execute
            state, reward, terminated, truncated, info = val_env.step(action)
            total_reward += reward
            episode_steps += 1

            # Track action by regime
            action_name = ["HOLD", "BUY", "SELL"][action]
            regime = info.get('regime', 'neutral')
            actions_by_regime[regime][action_name] += 1

            # Check for "fighting trend" pattern
            if action == 1:  # BUY
                total_buys += 1
                price = info['price']
                row = val_env.market_data.iloc[val_env.current_idx - 1]
                sma_50 = row.get('sma_50', price)

                if price < sma_50 * 0.95:  # > 5% below SMA50
                    buys_below_sma50 += 1

            if terminated or truncated:
                break

        final_value = val_env.portfolio.get_total_value()
        final_return = ((final_value - 100000) / 100000) * 100

        # Calculate fighting trend rate
        fighting_trend_rate = buys_below_sma50 / total_buys if total_buys > 0 else 0

        # Get BUY rate in bear market
        bear_decisions = sum(actions_by_regime['bear'].values())
        bear_buy_rate = (actions_by_regime['bear']['BUY'] / bear_decisions * 100) if bear_decisions > 0 else 0

        results = {
            'period': period['name'],
            'final_value': final_value,
            'return_pct': final_return,
            'total_reward': total_reward,
            'steps': episode_steps,
            'fighting_trend_rate': fighting_trend_rate,
            'bear_buy_rate': bear_buy_rate,
            'weight': period['weight']
        }

        all_results.append(results)

        # Weighted return
        weighted_return_sum += final_return * period['weight']
        total_weight += period['weight']

        # Print results
        print(f"  Return:               {final_return:+.2f}%")
        print(f"  Fighting Trend Rate:  {fighting_trend_rate*100:.1f}%")
        if bear_decisions > 0:
            print(f"  Bear Market BUY Rate: {bear_buy_rate:.1f}%")

    # Calculate weighted average
    weighted_avg_return = weighted_return_sum / total_weight

    print(f"\n{'='*60}")
    print(f"WEIGHTED AVERAGE RETURN: {weighted_avg_return:+.2f}%")
    print(f"(Bear market weighted {validation_periods[0]['weight']}x)")
    print('='*60)

    return {
        'weighted_avg_return': weighted_avg_return,
        'period_results': all_results
    }


def train_regime_aware_model_v2(
    episodes: int = 1000,
    validate_every: int = 100,
    save_dir: str = "rl/models/regime_aware_v2",
    device: str = None,
    regime_penalty_weight: float = 0.3,  # REDUCED from 0.5
    trend_bonus_weight: float = 0.3,
    sell_bonus_weight: float = 0.2
):
    """
    Train PPO agent with regime-aware environment V2.

    IMPROVEMENTS:
    - Multi-period validation (bear, bull, recovery)
    - Reduced regime penalty weight (0.3 vs 0.5)
    - Weighted validation score prevents overfitting
    """
    print("="*80)
    print("REGIME-AWARE RL TRAINING V2")
    print("="*80)
    print()
    print("IMPROVEMENTS:")
    print("- Validates on 3 diverse periods (not just 2022 bear)")
    print("- Reduced regime_penalty_weight: 0.5 → 0.3")
    print("- Weighted validation score (bear market 1.5x)")
    print()

    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Device: {device}")

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create training environment
    print("\nInitializing training environment...")
    train_env = TradingEnvV2(
        symbol="BTC-USD",
        start_date="2017-01-01",
        end_date="2021-12-31",
        initial_capital=100000.0,
        use_regime_rewards=True,
        regime_penalty_weight=regime_penalty_weight,  # REDUCED
        trend_bonus_weight=trend_bonus_weight,
        sell_bonus_weight=sell_bonus_weight
    )

    state_dim = train_env.observation_space.shape[0]  # 53
    action_dim = train_env.action_space.n  # 3

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"\nHyperparameters:")
    print(f"  regime_penalty_weight: {regime_penalty_weight}")
    print(f"  trend_bonus_weight: {trend_bonus_weight}")
    print(f"  sell_bonus_weight: {sell_bonus_weight}")

    # Create agent
    print("\nInitializing PPO agent...")
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        entropy_coef=0.01,
        value_coef=0.5,
        device=device
    )

    # Training loop
    print(f"\nStarting training for {episodes} episodes...")
    print(f"Multi-period validation every {validate_every} episodes")
    print()

    training_history = []
    validation_history = []
    best_val_return = -float('inf')

    for episode in range(1, episodes + 1):
        state, info = train_env.reset()
        episode_reward = 0
        episode_steps = 0

        while True:
            # Select action
            action = agent.select_action(state)

            # Execute action
            next_state, reward, terminated, truncated, info = train_env.step(action)

            # Store transition
            agent.store_reward_and_terminal(reward, terminated or truncated)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if terminated or truncated:
                break

        # Update policy
        metrics = agent.update()

        # Record training metrics
        final_value = train_env.portfolio.get_total_value()
        final_return = ((final_value - 100000) / 100000) * 100

        training_history.append({
            'episode': episode,
            'reward': episode_reward,
            'return': final_return,
            'steps': episode_steps,
            'loss': metrics['loss'],
            'policy_loss': metrics['policy_loss'],
            'value_loss': metrics['value_loss']
        })

        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Return: {final_return:+7.2f}% | "
                  f"Steps: {episode_steps:4d} | "
                  f"Loss: {metrics['loss']:.4f}")

        # Validate and save checkpoint
        if episode % validate_every == 0:
            val_results = validate_on_multiple_periods(agent, device)

            validation_history.append({
                'episode': episode,
                **val_results
            })

            # Save checkpoint
            checkpoint_path = save_path / f"episode_{episode}.pth"
            agent.save(str(checkpoint_path))
            print(f"\n✓ Checkpoint saved: {checkpoint_path}")

            # Save best model based on WEIGHTED average return
            weighted_return = val_results['weighted_avg_return']
            if weighted_return > best_val_return:
                best_val_return = weighted_return
                best_path = save_path / "best_model.pth"
                agent.save(str(best_path))
                print(f"✓ New best model! Weighted avg return: {best_val_return:+.2f}%")

            # Early stopping check - now based on weighted average
            if episode >= 300:
                recent_vals = [v['weighted_avg_return'] for v in validation_history[-3:]]
                if len(recent_vals) == 3 and all(r < -10 for r in recent_vals):
                    print("\n⚠ Early stopping: Consistent poor validation performance")
                    break

    # Save final model
    final_path = save_path / "final_model.pth"
    agent.save(str(final_path))

    # Save training history
    history_path = save_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'training': training_history,
            'validation': validation_history,
            'hyperparameters': {
                'regime_penalty_weight': regime_penalty_weight,
                'trend_bonus_weight': trend_bonus_weight,
                'sell_bonus_weight': sell_bonus_weight
            }
        }, f, indent=2)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nModels saved to: {save_path}/")
    print(f"  - best_model.pth (best weighted validation: {best_val_return:+.2f}%)")
    print(f"  - final_model.pth")
    print(f"\nTraining history saved to: {history_path}")

    # Print final validation comparison
    if validation_history:
        print("\n" + "="*80)
        print("VALIDATION PERFORMANCE OVER TIME (Weighted Average)")
        print("="*80)
        print(f"\n{'Episode':<10} {'Weighted Return %':<20}")
        print("-"*35)
        for val in validation_history:
            print(f"{val['episode']:<10} {val['weighted_avg_return']:>+18.2f}%")

    return agent, training_history, validation_history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train regime-aware RL trading agent V2")
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--validate-every', type=int, default=100, help='Validate every N episodes')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default='rl/models/regime_aware_v2',
                        help='Directory to save models')
    parser.add_argument('--regime-penalty', type=float, default=0.3,
                        help='Regime penalty weight (default: 0.3, was 0.5 in v1)')
    parser.add_argument('--trend-bonus', type=float, default=0.3,
                        help='Trend bonus weight (default: 0.3)')
    parser.add_argument('--sell-bonus', type=float, default=0.2,
                        help='Sell bonus weight (default: 0.2)')

    args = parser.parse_args()

    try:
        agent, train_history, val_history = train_regime_aware_model_v2(
            episodes=args.episodes,
            validate_every=args.validate_every,
            save_dir=args.save_dir,
            device=args.device,
            regime_penalty_weight=args.regime_penalty,
            trend_bonus_weight=args.trend_bonus,
            sell_bonus_weight=args.sell_bonus
        )

        print("\n✓ Training completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n⚠ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Train Regime-Aware RL Trading Agent

This training script implements fixes based on forensics findings:
1. Uses TradingEnvV2 with regime features (53-dim state space)
2. Validates on 2022 bear market every 100 episodes
3. Early stopping if validation performance degrades
4. Tracks improvement in failure patterns

Based on forensics showing model needs:
- Regime awareness (11.5% error in neutral markets)
- Trend confirmation (42% of failures = fighting trend)
- Better sell signals (only 5% sell rate)
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


def validate_on_critical_period(
    agent: PPOAgent,
    device: torch.device,
    period_name: str = "2022 Bear Market"
) -> dict:
    """
    Validate model on 2022 bear market - the period where previous model failed.

    Returns metrics to check if forensics patterns are improving.
    """
    print(f"\n{'='*60}")
    print(f"VALIDATION: {period_name}")
    print('='*60)

    # Create validation environment
    val_env = TradingEnvV2(
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2022-12-31",
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

    # Track decisions
    buys_below_sma50 = 0  # "Fighting trend" pattern
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
            # Get SMA50 from current row
            row = val_env.market_data.iloc[val_env.current_idx - 1]
            sma_50 = row.get('sma_50', price)

            if price < sma_50 * 0.95:  # > 5% below SMA50
                buys_below_sma50 += 1

        if terminated or truncated:
            break

    final_value = val_env.portfolio.get_total_value()
    final_return = ((final_value - 100000) / 100000) * 100

    # Calculate metrics
    results = {
        "final_value": final_value,
        "return_pct": final_return,
        "total_reward": total_reward,
        "steps": episode_steps,
        "actions_by_regime": actions_by_regime,
        "buys_below_sma50": buys_below_sma50,
        "total_buys": total_buys,
        "fighting_trend_rate": buys_below_sma50 / total_buys if total_buys > 0 else 0
    }

    # Print results
    print(f"\nResults:")
    print(f"  Final Value:     ${final_value:,.2f}")
    print(f"  Return:          {final_return:+.2f}%")
    print(f"  Total Reward:    {total_reward:.2f}")
    print(f"  Steps:           {episode_steps}")
    print(f"\nAction Distribution by Regime:")
    for regime in ["bull", "neutral", "bear"]:
        actions = actions_by_regime[regime]
        total = sum(actions.values())
        if total > 0:
            print(f"  {regime.upper():8s}: HOLD={actions['HOLD']:3d} ({actions['HOLD']/total*100:.1f}%), "
                  f"BUY={actions['BUY']:3d} ({actions['BUY']/total*100:.1f}%), "
                  f"SELL={actions['SELL']:3d} ({actions['SELL']/total*100:.1f}%)")

    print(f"\nFailure Pattern Check:")
    print(f"  'Fighting Trend' Rate: {results['fighting_trend_rate']*100:.1f}% "
          f"({buys_below_sma50}/{total_buys} BUYs below SMA50)")
    print(f"  Target: < 20% (old model was 42%)")

    return results


def train_regime_aware_model(
    episodes: int = 1000,
    validate_every: int = 100,
    save_dir: str = "rl/models/regime_aware",
    device: str = None
):
    """
    Train PPO agent with regime-aware environment.

    Args:
        episodes: Number of training episodes
        validate_every: Validate every N episodes
        save_dir: Directory to save models
        device: Device to use ('cuda' or 'cpu')
    """
    print("="*80)
    print("REGIME-AWARE RL TRAINING")
    print("="*80)
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
        regime_penalty_weight=0.5,
        trend_bonus_weight=0.3,
        sell_bonus_weight=0.2
    )

    state_dim = train_env.observation_space.shape[0]  # 53
    action_dim = train_env.action_space.n  # 3

    print(f"State dimension: {state_dim} (was 43 in old model)")
    print(f"Action dimension: {action_dim}")

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
    print(f"Validation on 2022 bear market every {validate_every} episodes")
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
            val_results = validate_on_critical_period(agent, device)

            validation_history.append({
                'episode': episode,
                **val_results
            })

            # Save checkpoint
            checkpoint_path = save_path / f"episode_{episode}.pth"
            agent.save(str(checkpoint_path))
            print(f"\n✓ Checkpoint saved: {checkpoint_path}")

            # Save best model
            if val_results['return_pct'] > best_val_return:
                best_val_return = val_results['return_pct']
                best_path = save_path / "best_model.pth"
                agent.save(str(best_path))
                print(f"✓ New best model! Return: {best_val_return:+.2f}%")

            # Early stopping check
            if episode >= 300:  # After initial warm-up
                recent_vals = [v['return_pct'] for v in validation_history[-3:]]
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
            'validation': validation_history
        }, f, indent=2)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nModels saved to: {save_path}/")
    print(f"  - best_model.pth (best validation return: {best_val_return:+.2f}%)")
    print(f"  - final_model.pth")
    print(f"  - episode_X00.pth checkpoints")
    print(f"\nTraining history saved to: {history_path}")

    # Print final validation comparison
    if validation_history:
        print("\n" + "="*80)
        print("VALIDATION PERFORMANCE OVER TIME")
        print("="*80)
        print(f"\n{'Episode':<10} {'Return %':<12} {'Fighting Trend %':<20}")
        print("-"*45)
        for val in validation_history:
            print(f"{val['episode']:<10} {val['return_pct']:>+10.2f}% "
                  f"{val['fighting_trend_rate']*100:>17.1f}%")

    return agent, training_history, validation_history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train regime-aware RL trading agent")
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--validate-every', type=int, default=100, help='Validate every N episodes')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default='rl/models/regime_aware',
                        help='Directory to save models')

    args = parser.parse_args()

    try:
        agent, train_history, val_history = train_regime_aware_model(
            episodes=args.episodes,
            validate_every=args.validate_every,
            save_dir=args.save_dir,
            device=args.device
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

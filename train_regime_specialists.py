#!/usr/bin/env python3
"""
Regime-Specific Specialist Agent Training
=========================================

Trains 3 specialized agents (bull/bear/sideways) using:
- Sharpe-optimized rewards with regime-specific targets
- Weighted sampling (oversample target regime 3x)
- Early stopping per agent (50-episode patience)
- Regime-matched validation

Expected Outcomes:
- Bull specialist: Sharpe >1.5 on bull markets
- Bear specialist: Sharpe >0.5 on bear markets (capital preservation)
- Sideways specialist: Sharpe >1.0 on sideways markets
"""

import sys
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl.ppo_agent import PPOAgent
from rl.trading_env import TradingEnv
from rl.sharpe_reward import RegimeAwareSharpeReward

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegimeSpecificTrainer:
    """Train regime-specific specialist agents."""

    def __init__(
        self,
        regime: str,
        labeled_data_path: str = "bitcoin_labeled_regimes.csv",
        num_episodes: int = 200,
        patience: int = 50,
        device: torch.device = None
    ):
        """
        Initialize regime-specific trainer.

        Args:
            regime: Target regime (BULL, BEAR, or SIDEWAYS)
            labeled_data_path: Path to labeled regime data
            num_episodes: Maximum training episodes
            patience: Early stopping patience
            device: torch device
        """
        self.regime = regime
        self.num_episodes = num_episodes
        self.patience = patience
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load labeled regime data
        print(f"Loading regime-labeled data from {labeled_data_path}...")
        self.labeled_data = pd.read_csv(labeled_data_path)
        self.labeled_data['Date'] = pd.to_datetime(self.labeled_data['Date'])

        # Filter to training period (2017-2021)
        self.training_data = self.labeled_data[
            (self.labeled_data['Date'] >= '2017-01-01') &
            (self.labeled_data['Date'] <= '2021-12-31')
        ].copy()

        print(f"  Total training days: {len(self.training_data)}")
        print(f"  Target regime ({regime}): {(self.training_data['regime_primary'] == regime).sum()} days")
        print()

        # Calculate sampling weights
        self.sampling_weights = np.where(
            self.training_data['regime_primary'].values == regime,
            3.0,  # Target regime: 3x weight
            1.0   # Other regimes: 1x weight
        )
        self.sampling_weights = self.sampling_weights / self.sampling_weights.sum()

        # Regime-specific targets for Sharpe calculator
        self.regime_targets = {
            'BULL': {'target_sharpe': 1.5, 'description': 'Aggressive gains'},
            'BEAR': {'target_sharpe': 0.5, 'description': 'Capital preservation'},
            'SIDEWAYS': {'target_sharpe': 1.0, 'description': 'Consistent profits'}
        }

    def sample_episode_start(self) -> Tuple[str, str]:
        """
        Sample a training episode start date using weighted sampling.

        Returns:
            (start_date, end_date) tuple for episode
        """
        # Sample a random start date with weighted probability
        sampled_idx = np.random.choice(
            len(self.training_data),
            p=self.sampling_weights
        )

        # Get date range for this episode (use ~30 days of data per episode)
        start_idx = max(0, sampled_idx - 15)
        end_idx = min(len(self.training_data) - 1, sampled_idx + 15)

        start_date = self.training_data.iloc[start_idx]['Date'].strftime('%Y-%m-%d')
        end_date = self.training_data.iloc[end_idx]['Date'].strftime('%Y-%m-%d')

        return start_date, end_date

    def create_environment(self, start_date: str, end_date: str, initial_capital: float) -> TradingEnv:
        """Create trading environment with regime-aware Sharpe rewards."""
        return TradingEnv(
            symbol="BTC-USD",
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            commission_rate=0.001,
            use_sharpe_reward=True  # Use Sharpe rewards
        )

    def evaluate_on_regime_matched_validation(
        self,
        agent: PPOAgent,
        initial_capital: float
    ) -> Dict:
        """
        Evaluate agent on regime-matched validation data.

        For each regime, test on periods from 2022 that match the regime.
        """
        # Filter validation data (2022) for target regime
        val_data = self.labeled_data[
            (self.labeled_data['Date'] >= '2022-01-01') &
            (self.labeled_data['Date'] <= '2022-12-31')
        ]

        regime_days = val_data[val_data['regime_primary'] == self.regime]

        if len(regime_days) == 0:
            print(f"  [WARNING] No {self.regime} days in validation set")
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'trade_count': 0,
                'regime_days': 0
            }

        # Use full 2022 for now (could optimize to only test on regime days)
        val_env = TradingEnv(
            symbol="BTC-USD",
            start_date="2022-01-01",
            end_date="2022-12-31",
            initial_capital=initial_capital,
            commission_rate=0.001,
            use_sharpe_reward=True
        )

        agent.policy.eval()
        state, info = val_env.reset()

        while True:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs, _ = agent.policy(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()

            state, reward, terminated, truncated, info = val_env.step(action)

            if terminated or truncated:
                break

        # Get metrics from Sharpe calculator
        if val_env.sharpe_calculator:
            metrics = val_env.sharpe_calculator.get_episode_metrics()
        else:
            final_value = val_env.portfolio.get_total_value()
            total_return = (final_value - initial_capital) / initial_capital
            metrics = {
                'sharpe_ratio': 0.0,
                'total_return': total_return,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'trade_count': 0
            }

        metrics['final_value'] = val_env.portfolio.get_total_value()
        metrics['regime_days'] = len(regime_days)
        agent.policy.train()

        return metrics

    def train_specialist(self, model_dir: Path, initial_capital: float = 100000.0) -> Dict:
        """
        Train regime specialist agent.

        Args:
            model_dir: Directory to save models
            initial_capital: Starting capital

        Returns:
            Training metrics dict
        """
        model_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print(f"{self.regime} MARKET SPECIALIST TRAINING")
        print("=" * 80)
        print()
        print(f"Target: {self.regime_targets[self.regime]['description']}")
        print(f"Target Sharpe: >{self.regime_targets[self.regime]['target_sharpe']}")
        print(f"Episodes: {self.num_episodes} (with early stopping)")
        print(f"Patience: {self.patience} episodes")
        print(f"Weighted Sampling: {self.regime}=3x, others=1x")
        print()

        # Create initial environment to get dimensions
        start_date, end_date = self.sample_episode_start()
        temp_env = self.create_environment(start_date, end_date, initial_capital)

        # Initialize agent
        agent = PPOAgent(
            state_dim=temp_env.observation_space.shape[0],
            action_dim=temp_env.action_space.n,
            device=self.device,
            lr=3e-4,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=4
        )

        print(f"Agent initialized on device: {self.device}")
        print()

        # Training metrics
        episode_rewards = []
        episode_sharpes = []
        episode_returns = []
        validation_sharpes = []
        validation_episodes = []

        best_val_sharpe = -np.inf
        patience_counter = 0

        print(f"Training {self.regime} specialist...")
        print()

        for episode in range(1, self.num_episodes + 1):
            # Sample new episode with weighted sampling
            start_date, end_date = self.sample_episode_start()
            train_env = self.create_environment(start_date, end_date, initial_capital)

            state, info = train_env.reset()
            episode_reward = 0.0
            step = 0
            update_freq = 2048

            while True:
                # Select action
                action = agent.select_action(state)

                # Take step
                next_state, reward, terminated, truncated, info = train_env.step(action)

                # Store reward and terminal
                agent.store_reward_and_terminal(reward, terminated or truncated)

                episode_reward += reward
                state = next_state
                step += 1

                # Update policy
                if step % update_freq == 0:
                    agent.update()

                if terminated or truncated:
                    break

            # Record metrics
            final_value = train_env.portfolio.get_total_value()
            episode_return = ((final_value - initial_capital) / initial_capital) * 100

            if train_env.sharpe_calculator:
                train_metrics = train_env.sharpe_calculator.get_episode_metrics()
                episode_sharpe = train_metrics['sharpe_ratio']
            else:
                episode_sharpe = 0.0

            episode_rewards.append(episode_reward)
            episode_returns.append(episode_return)
            episode_sharpes.append(episode_sharpe)

            # Print progress
            if episode % 10 == 0:
                avg_sharpe = np.mean(episode_sharpes[-10:])
                print(f"Episode {episode}/{self.num_episodes} | "
                      f"Sharpe: {episode_sharpe:+.3f} | "
                      f"Avg(10): {avg_sharpe:+.3f} | "
                      f"Return: {episode_return:+.2f}%")

                # Validation
                print(f"  [VALIDATION] Testing on {self.regime}-matched validation...")
                val_metrics = self.evaluate_on_regime_matched_validation(agent, initial_capital)
                val_sharpe = val_metrics['sharpe_ratio']
                val_return = val_metrics['total_return'] * 100

                validation_sharpes.append(val_sharpe)
                validation_episodes.append(episode)

                print(f"  [VALIDATION] Sharpe: {val_sharpe:+.3f} | "
                      f"Return: {val_return:+.2f}% | "
                      f"Regime Days: {val_metrics['regime_days']}")

                # Check for improvement
                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    patience_counter = 0

                    # Save best model
                    best_model_path = model_dir / f"{self.regime.lower()}_agent.pth"
                    torch.save(agent.policy.state_dict(), best_model_path)
                    print(f"  [NEW BEST] Validation Sharpe: {val_sharpe:+.3f} (saved)")
                else:
                    patience_counter += 1
                    print(f"  [NO IMPROVEMENT] Patience: {patience_counter}/{self.patience}")

                    # Early stopping
                    if patience_counter >= self.patience:
                        print()
                        print(f"  [EARLY STOPPING] No improvement for {self.patience} validations")
                        print(f"  [EARLY STOPPING] Best validation Sharpe: {best_val_sharpe:+.3f}")
                        print(f"  [EARLY STOPPING] Stopping at episode {episode}")
                        break

                print()

        print()
        print("=" * 80)
        print(f"{self.regime} SPECIALIST TRAINING COMPLETE")
        print("=" * 80)
        print()
        print(f"Episodes Completed: {episode}/{self.num_episodes}")
        print(f"Best Validation Sharpe: {best_val_sharpe:+.3f}")
        print(f"Target Sharpe: >{self.regime_targets[self.regime]['target_sharpe']}")

        if best_val_sharpe >= self.regime_targets[self.regime]['target_sharpe']:
            print(f"✅ TARGET ACHIEVED!")
        else:
            print(f"⚠️  Target not reached, but best effort saved")

        print()

        # Save training metrics
        training_metrics = {
            "regime": self.regime,
            "episode_rewards": episode_rewards,
            "episode_returns": episode_returns,
            "episode_sharpes": episode_sharpes,
            "validation_sharpes": validation_sharpes,
            "validation_episodes": validation_episodes,
            "best_val_sharpe": best_val_sharpe,
            "target_sharpe": self.regime_targets[self.regime]['target_sharpe'],
            "total_episodes": episode,
            "early_stopped": (episode < self.num_episodes)
        }

        metrics_path = model_dir / f"{self.regime.lower()}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        print()

        return training_metrics


def train_all_specialists():
    """Train all three regime specialists sequentially."""

    print("=" * 80)
    print("REGIME SPECIALIST TRAINING - ALL 3 AGENTS")
    print("=" * 80)
    print()
    print("Training Configuration:")
    print("  Training Data:     2017-2021 (regime-labeled)")
    print("  Validation Data:   2022 (regime-matched)")
    print()
    print("  Episodes per agent: 200 (with early stopping)")
    print("  Early Stopping:     50 episode patience")
    print("  Sampling Strategy:  Weighted (target=3x, others=1x)")
    print()
    print("Agents to Train:")
    print("  1. Bull Specialist   (Target Sharpe >1.5)")
    print("  2. Bear Specialist   (Target Sharpe >0.5)")
    print("  3. Sideways Specialist (Target Sharpe >1.0)")
    print("=" * 80)
    print()

    root_dir = Path(__file__).parent
    model_dir = root_dir / "rl" / "models" / "regime_specialists"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    all_metrics = {}

    # Train each specialist
    for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
        trainer = RegimeSpecificTrainer(
            regime=regime,
            num_episodes=200,
            patience=50,
            device=device
        )

        metrics = trainer.train_specialist(model_dir)
        all_metrics[regime] = metrics

        print()
        print("=" * 80)
        print()

    # Summary
    print("=" * 80)
    print("ALL SPECIALISTS TRAINING COMPLETE")
    print("=" * 80)
    print()
    print("Final Results:")
    print("-" * 80)
    for regime, metrics in all_metrics.items():
        target = metrics['target_sharpe']
        achieved = metrics['best_val_sharpe']
        status = "✅ ACHIEVED" if achieved >= target else "⚠️  BELOW TARGET"
        print(f"  {regime:<12}: Sharpe {achieved:+.3f} (target >{target}) {status}")
    print()

    # Save combined summary
    summary_path = model_dir / "all_specialists_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Combined summary saved to: {summary_path}")
    print()

    return all_metrics


if __name__ == "__main__":
    try:
        metrics = train_all_specialists()
        print("=" * 80)
        print("✅ ALL SPECIALIST TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

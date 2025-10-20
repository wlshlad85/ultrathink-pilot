#!/usr/bin/env python3
"""
Policy Decision Validator: Verify Premature Convergence Hypothesis

Tests whether two checkpoints make identical trading decisions on the same data.
If episode 50 and episode 1000 agree on 100% of actions, it proves the policy
converged early and never truly changed despite continued training.
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rl.trading_env import TradingEnv
from rl.ppo_agent import PPOAgent


class PolicyComparator:
    """Compare decision-making behavior of two policy checkpoints."""

    def __init__(self, checkpoint1: Path, checkpoint2: Path, device: torch.device):
        """
        Initialize comparator with two checkpoints.

        Args:
            checkpoint1: Path to first checkpoint
            checkpoint2: Path to second checkpoint
            device: torch device (cuda/cpu)
        """
        self.checkpoint1_path = checkpoint1
        self.checkpoint2_path = checkpoint2
        self.device = device

        # Will be initialized when environment is created
        self.agent1 = None
        self.agent2 = None

    def load_agents(self, state_dim: int, action_dim: int):
        """Load both agents from checkpoints."""

        # Create agent 1
        self.agent1 = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=3e-4,
            gamma=0.99,
            k_epochs=4,
            eps_clip=0.2,
            device=self.device
        )
        self.agent1.policy.load_state_dict(
            torch.load(self.checkpoint1_path, map_location=self.device)
        )
        self.agent1.policy.eval()

        # Create agent 2
        self.agent2 = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=3e-4,
            gamma=0.99,
            k_epochs=4,
            eps_clip=0.2,
            device=self.device
        )
        self.agent2.policy.load_state_dict(
            torch.load(self.checkpoint2_path, map_location=self.device)
        )
        self.agent2.policy.eval()

    def get_action(self, agent: PPOAgent, state: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Get action and additional info from agent.

        Returns:
            action (int): Chosen action
            action_probs (np.ndarray): Probability distribution over actions
            confidence (float): Probability of chosen action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, _ = agent.policy(state_tensor)

            # Get action probabilities as numpy
            probs = action_probs.cpu().numpy()[0]

            # Get chosen action
            action = torch.argmax(action_probs, dim=1).item()

            # Get confidence (probability of chosen action)
            confidence = probs[action]

            return action, probs, confidence

    def compare_on_dataset(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0
    ) -> Dict:
        """
        Compare both policies on the same dataset.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital

        Returns:
            Dict with comparison results
        """

        # Create environment
        env = TradingEnv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            commission_rate=0.001
        )

        # Initialize agents
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.load_agents(state_dim, action_dim)

        # Track decisions
        decisions = []
        disagreements = []

        state, info = env.reset()
        step = 0

        print(f"\nComparing policies on {symbol} from {start_date} to {end_date}...")
        print(f"Total steps in environment: {len(env.market_data)}")
        print()

        while True:
            # Get decisions from both agents
            action1, probs1, conf1 = self.get_action(self.agent1, state)
            action2, probs2, conf2 = self.get_action(self.agent2, state)

            # Record decision
            current_price = env.market_data['close'].iloc[env.current_idx]
            current_date = env.market_data.index[env.current_idx]

            decision = {
                'step': step,
                'date': current_date,
                'price': current_price,
                'agent1_action': action1,
                'agent2_action': action2,
                'agent1_confidence': conf1,
                'agent2_confidence': conf2,
                'agree': action1 == action2
            }
            decisions.append(decision)

            # Track disagreement details
            if action1 != action2:
                disagreement = {
                    'step': step,
                    'date': current_date,
                    'price': current_price,
                    'agent1_action': action1,
                    'agent2_action': action2,
                    'agent1_probs': probs1,
                    'agent2_probs': probs2,
                    'state': state.copy()
                }
                disagreements.append(disagreement)

                # Print disagreement immediately
                action_names = ['BUY', 'HOLD', 'SELL']
                print(f"❌ DISAGREEMENT at step {step} ({current_date.strftime('%Y-%m-%d')}):")
                print(f"   Agent 1 ({self.checkpoint1_path.stem}): {action_names[action1]} (conf={conf1:.4f})")
                print(f"   Agent 2 ({self.checkpoint2_path.stem}): {action_names[action2]} (conf={conf2:.4f})")
                print(f"   Price: ${current_price:,.2f}")
                print()

            # Step environment (using agent 1's action, but doesn't matter for comparison)
            state, reward, terminated, truncated, info = env.step(action1)
            step += 1

            if terminated or truncated:
                break

        # Calculate statistics
        total_decisions = len(decisions)
        agreements = sum(1 for d in decisions if d['agree'])
        agreement_rate = (agreements / total_decisions) * 100 if total_decisions > 0 else 0

        return {
            'total_decisions': total_decisions,
            'agreements': agreements,
            'disagreements': len(disagreements),
            'agreement_rate': agreement_rate,
            'decisions_df': pd.DataFrame(decisions),
            'disagreement_details': disagreements
        }


def main():
    """Main execution function."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Compare decision-making of two policy checkpoints"
    )
    parser.add_argument(
        '--checkpoint1',
        type=str,
        required=True,
        help='Path to first checkpoint (e.g., episode_50.pth)'
    )
    parser.add_argument(
        '--checkpoint2',
        type=str,
        required=True,
        help='Path to second checkpoint (e.g., episode_1000.pth)'
    )
    parser.add_argument(
        '--test_set',
        type=str,
        default='2022_validation',
        choices=['2022_validation', '2023_test', '2024_test', 'full_test'],
        help='Which dataset to test on'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='policy_comparison.csv',
        help='Output CSV file for decision log'
    )

    args = parser.parse_args()

    # Map test set names to date ranges
    test_sets = {
        '2022_validation': ('BTC-USD', '2022-01-01', '2022-12-31'),
        '2023_test': ('BTC-USD', '2023-01-01', '2023-12-31'),
        '2024_test': ('BTC-USD', '2024-01-01', '2024-12-31'),
        'full_test': ('BTC-USD', '2023-01-01', '2024-12-31')
    }

    symbol, start_date, end_date = test_sets[args.test_set]

    print("=" * 80)
    print("POLICY DECISION VALIDATOR")
    print("=" * 80)
    print()
    print(f"Checkpoint 1: {args.checkpoint1}")
    print(f"Checkpoint 2: {args.checkpoint2}")
    print(f"Test Set: {args.test_set} ({start_date} to {end_date})")
    print()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve checkpoint paths
    models_dir = Path("rl/models/professional")
    checkpoint1_path = models_dir / args.checkpoint1
    checkpoint2_path = models_dir / args.checkpoint2

    if not checkpoint1_path.exists():
        print(f"ERROR: Checkpoint 1 not found: {checkpoint1_path}")
        sys.exit(1)
    if not checkpoint2_path.exists():
        print(f"ERROR: Checkpoint 2 not found: {checkpoint2_path}")
        sys.exit(1)

    # Create comparator
    comparator = PolicyComparator(checkpoint1_path, checkpoint2_path, device)

    # Run comparison
    results = comparator.compare_on_dataset(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )

    # Print results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Total Decisions: {results['total_decisions']}")
    print(f"Agreements: {results['agreements']}")
    print(f"Disagreements: {results['disagreements']}")
    print(f"Agreement Rate: {results['agreement_rate']:.2f}%")
    print()

    if results['agreement_rate'] == 100.0:
        print("✅ HYPOTHESIS CONFIRMED:")
        print("   Policies make IDENTICAL decisions on all inputs!")
        print("   This proves premature convergence - the policy converged early")
        print("   and never changed despite continued training.")
    elif results['agreement_rate'] > 99.0:
        print("⚠️  NEARLY IDENTICAL:")
        print(f"   Policies agree on {results['agreement_rate']:.2f}% of decisions.")
        print("   Differences are extremely rare - likely due to numerical precision")
        print("   or stochastic tie-breaking in near-equal probabilities.")
    else:
        print("❌ HYPOTHESIS REJECTED:")
        print("   Policies make different decisions.")
        print(f"   Agreement rate of {results['agreement_rate']:.2f}% suggests")
        print("   the checkpoints represent genuinely different policies.")
    print()

    # Save detailed decision log
    results['decisions_df'].to_csv(args.output, index=False)
    print(f"✅ Decision log saved to: {args.output}")

    # Save disagreement details if any
    if results['disagreements'] > 0:
        disagreement_output = args.output.replace('.csv', '_disagreements.csv')

        disagreement_rows = []
        for d in results['disagreement_details']:
            row = {
                'step': d['step'],
                'date': d['date'],
                'price': d['price'],
                'agent1_action': d['agent1_action'],
                'agent2_action': d['agent2_action']
            }
            # Add probability distributions
            for i, prob in enumerate(d['agent1_probs']):
                row[f'agent1_prob_action{i}'] = prob
            for i, prob in enumerate(d['agent2_probs']):
                row[f'agent2_prob_action{i}'] = prob

            disagreement_rows.append(row)

        pd.DataFrame(disagreement_rows).to_csv(disagreement_output, index=False)
        print(f"✅ Disagreement details saved to: {disagreement_output}")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    try:
        results = main()

        # Exit code based on hypothesis confirmation
        if results['agreement_rate'] == 100.0:
            sys.exit(0)  # Success: Hypothesis confirmed
        else:
            sys.exit(1)  # Hypothesis rejected

    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

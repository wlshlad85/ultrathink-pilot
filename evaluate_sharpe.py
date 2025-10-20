#!/usr/bin/env python3
"""
Risk-Adjusted Performance Evaluation Framework

Evaluates trading agents using comprehensive risk-adjusted metrics:
- Sharpe Ratio: Return per unit of total volatility
- Sortino Ratio: Return per unit of downside volatility
- Calmar Ratio: Return per unit of maximum drawdown
- Maximum Drawdown: Worst peak-to-trough decline
- Win Rate: Percentage of profitable trades
- Profit Factor: Gross profit / gross loss

This framework replaces simple return-based evaluation with proper risk analysis.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rl.trading_env import TradingEnv
from rl.ppo_agent import PPOAgent


class RiskAdjustedEvaluator:
    """Comprehensive risk-adjusted performance evaluator for trading agents."""

    def __init__(self, checkpoint_path: Path, device: torch.device):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            device: torch device (cuda/cpu)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.agent = None

    def load_agent(self, state_dim: int, action_dim: int):
        """Load agent from checkpoint."""
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=3e-4,
            gamma=0.99,
            k_epochs=4,
            eps_clip=0.2,
            device=self.device
        )
        self.agent.policy.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        self.agent.policy.eval()

    def evaluate_episode(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        risk_free_rate: float = 0.02  # 2% annual risk-free rate (approx T-bill)
    ) -> Dict:
        """
        Run evaluation episode and collect detailed performance data.

        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            commission_rate: Commission per trade
            risk_free_rate: Annual risk-free rate for Sharpe calculation

        Returns:
            Dict with comprehensive metrics
        """

        # Create environment
        env = TradingEnv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            commission_rate=commission_rate
        )

        # Load agent
        if self.agent is None:
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            self.load_agent(state_dim, action_dim)

        # Track detailed metrics
        portfolio_values = []
        daily_returns = []
        trades = []
        actions_taken = []

        state, info = env.reset()
        initial_value = env.portfolio.get_total_value()
        portfolio_values.append(initial_value)

        step = 0
        previous_value = initial_value

        while True:
            # Select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs, _ = self.agent.policy(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()

            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)

            # Record portfolio value and daily return
            current_value = env.portfolio.get_total_value()
            portfolio_values.append(current_value)

            daily_return = (current_value - previous_value) / previous_value
            daily_returns.append(daily_return)

            # Record action
            actions_taken.append(action)

            # Record trades (BUY or SELL actions)
            if action in [0, 2]:  # BUY or SELL
                trade_info = {
                    'step': step,
                    'date': env.market_data.index[env.current_idx],
                    'action': 'BUY' if action == 0 else 'SELL',
                    'price': env.market_data['close'].iloc[env.current_idx],
                    'portfolio_value_before': previous_value,
                    'portfolio_value_after': current_value,
                    'trade_return': daily_return
                }
                trades.append(trade_info)

            previous_value = current_value
            state = next_state
            step += 1

            if terminated or truncated:
                break

        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(
            portfolio_values=portfolio_values,
            daily_returns=daily_returns,
            trades=trades,
            actions_taken=actions_taken,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
            total_days=len(env.market_data)
        )

        return metrics

    def _calculate_metrics(
        self,
        portfolio_values: List[float],
        daily_returns: List[float],
        trades: List[Dict],
        actions_taken: List[int],
        initial_capital: float,
        risk_free_rate: float,
        total_days: int
    ) -> Dict:
        """Calculate comprehensive performance metrics."""

        # Basic return metrics
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_capital) / initial_capital
        total_return_pct = total_return * 100

        # Annualized return (assuming 365 trading days per year)
        years = total_days / 365
        annualized_return = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        annualized_return_pct = annualized_return * 100

        # Volatility metrics
        returns_array = np.array(daily_returns)
        daily_volatility = np.std(returns_array)
        annualized_volatility = daily_volatility * np.sqrt(365)
        annualized_volatility_pct = annualized_volatility * 100

        # Sharpe Ratio (risk-adjusted return)
        # Sharpe = (Return - Risk_Free_Rate) / Volatility
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0

        # Sortino Ratio (downside risk-adjusted return)
        # Only considers downside volatility (negative returns)
        downside_returns = returns_array[returns_array < 0]
        downside_volatility = np.std(downside_returns) if len(downside_returns) > 0 else 0
        annualized_downside_volatility = downside_volatility * np.sqrt(365)
        sortino_ratio = excess_return / annualized_downside_volatility if annualized_downside_volatility > 0 else 0

        # Maximum Drawdown
        portfolio_series = pd.Series(portfolio_values)
        running_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - running_max) / running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = max_drawdown * 100

        # Calmar Ratio (return / max drawdown)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade metrics
        num_trades = len(trades)
        if num_trades > 0:
            trade_returns = [t['trade_return'] for t in trades]
            profitable_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]

            win_rate = len(profitable_trades) / num_trades if num_trades > 0 else 0
            win_rate_pct = win_rate * 100

            avg_win = np.mean(profitable_trades) if len(profitable_trades) > 0 else 0
            avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0

            # Profit Factor: Total wins / Total losses
            total_wins = sum(profitable_trades) if len(profitable_trades) > 0 else 0
            total_losses = abs(sum(losing_trades)) if len(losing_trades) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        else:
            win_rate_pct = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Action distribution
        action_counts = pd.Series(actions_taken).value_counts().to_dict()
        action_names = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}
        action_distribution = {
            action_names[k]: v for k, v in action_counts.items()
        }

        # Compile all metrics
        metrics = {
            # Return metrics
            'total_return_pct': total_return_pct,
            'annualized_return_pct': annualized_return_pct,
            'final_value': final_value,

            # Risk metrics
            'annualized_volatility_pct': annualized_volatility_pct,
            'max_drawdown_pct': max_drawdown_pct,

            # Risk-adjusted metrics (PRIMARY)
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,

            # Trade metrics
            'num_trades': num_trades,
            'win_rate_pct': win_rate_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,

            # Action distribution
            'action_distribution': action_distribution,

            # Metadata
            'total_steps': len(actions_taken),
            'total_days': total_days,
            'initial_capital': initial_capital
        }

        return metrics


def evaluate_checkpoint(
    checkpoint_name: str,
    test_sets: Dict[str, Tuple[str, str, str]],
    models_dir: Path = Path("rl/models/professional"),
    output_dir: Path = Path("."),
    device: torch.device = None
) -> Dict:
    """
    Evaluate a single checkpoint across multiple test sets with risk-adjusted metrics.

    Args:
        checkpoint_name: Name of checkpoint file (e.g., "episode_1000.pth")
        test_sets: Dict mapping test set names to (symbol, start_date, end_date)
        models_dir: Directory containing checkpoints
        output_dir: Directory for output files
        device: torch device (defaults to cuda if available)

    Returns:
        Dict with results for each test set
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = models_dir / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Evaluating: {checkpoint_name}")
    print(f"Device: {device}")
    print()

    evaluator = RiskAdjustedEvaluator(checkpoint_path, device)

    results = {}

    for test_name, (symbol, start_date, end_date) in test_sets.items():
        print(f"Testing on {test_name} ({start_date} to {end_date})...", end=" ", flush=True)

        metrics = evaluator.evaluate_episode(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        results[test_name] = metrics

        # Print key metrics
        print(f"Return: {metrics['total_return_pct']:+.2f}% | "
              f"Sharpe: {metrics['sharpe_ratio']:.3f} | "
              f"MaxDD: {metrics['max_drawdown_pct']:.2f}%")

    return results


def main():
    """Main execution function."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate trading agent with risk-adjusted metrics"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='best_model.pth',
        help='Checkpoint file to evaluate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='risk_adjusted_evaluation.json',
        help='Output JSON file'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("RISK-ADJUSTED PERFORMANCE EVALUATION")
    print("=" * 80)
    print()

    # Define test sets
    test_sets = {
        'validation_2022': ('BTC-USD', '2022-01-01', '2022-12-31'),
        'test_2023': ('BTC-USD', '2023-01-01', '2023-12-31'),
        'test_2024': ('BTC-USD', '2024-01-01', '2024-12-31'),
        'test_full': ('BTC-USD', '2023-01-01', '2024-12-31')
    }

    # Run evaluation
    results = evaluate_checkpoint(
        checkpoint_name=args.checkpoint,
        test_sets=test_sets
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print("SUMMARY: Risk-Adjusted Metrics")
    print("=" * 80)
    print()

    # Print summary table
    print(f"{'Test Set':<20} {'Return':<12} {'Sharpe':<10} {'Sortino':<10} {'Calmar':<10} {'MaxDD':<10}")
    print("-" * 80)

    for test_name, metrics in results.items():
        print(f"{test_name:<20} "
              f"{metrics['total_return_pct']:>+10.2f}%  "
              f"{metrics['sharpe_ratio']:>8.3f}  "
              f"{metrics['sortino_ratio']:>8.3f}  "
              f"{metrics['calmar_ratio']:>8.3f}  "
              f"{metrics['max_drawdown_pct']:>8.2f}%")

    print()
    print(f"✅ Detailed results saved to: {args.output}")
    print()
    print("=" * 80)
    print()

    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

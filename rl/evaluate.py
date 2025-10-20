#!/usr/bin/env python3
"""
Evaluate trained RL agent on test data.
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.trading_env import TradingEnv
from rl.ppo_agent import PPOAgent
from backtesting.metrics import PerformanceMetrics
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_agent(
    model_path: str,
    symbol: str = "BTC-USD",
    start_date: str = "2024-01-01",
    end_date: str = "2024-06-01",
    initial_capital: float = 100000.0,
    deterministic: bool = True,
    render: bool = False
):
    """
    Evaluate trained agent on test data.

    Args:
        model_path: Path to trained model
        symbol: Trading symbol
        start_date: Test period start
        end_date: Test period end
        initial_capital: Initial capital
        deterministic: Use deterministic policy
        render: Render environment steps

    Returns:
        Results dictionary
    """
    logger.info(f"Evaluating agent on {symbol} from {start_date} to {end_date}")

    # Create environment
    env = TradingEnv(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )

    # Load agent
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    agent.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    # Run episode
    state, info = env.reset()
    episode_reward = 0
    steps = 0
    action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL

    while True:
        # Select action (deterministic or stochastic)
        if deterministic:
            # Use greedy policy
            import torch
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action_probs, _ = agent.policy(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()
        else:
            action = agent.select_action(state)

        action_counts[action] += 1

        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        steps += 1
        state = next_state

        if render:
            env.render()

        if terminated or truncated:
            break

    # Get final stats
    portfolio_stats = env.get_portfolio_stats()
    equity_df = env.portfolio.get_equity_dataframe()
    trades_df = env.portfolio.get_trades_dataframe()

    # Calculate metrics
    if len(equity_df) > 1:
        metrics_calc = PerformanceMetrics(equity_df)
        performance_metrics = metrics_calc.get_all_metrics(trades_df)
    else:
        performance_metrics = {}

    results = {
        "portfolio_stats": portfolio_stats,
        "performance_metrics": performance_metrics,
        "episode_reward": episode_reward,
        "steps": steps,
        "action_counts": action_counts,
        "equity_curve": equity_df.to_dict('records') if len(equity_df) > 0 else [],
        "trades": trades_df.to_dict('records') if len(trades_df) > 0 else []
    }

    return results


def print_evaluation_results(results: dict):
    """Print formatted evaluation results."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    print("\n--- Portfolio Performance ---")
    stats = results["portfolio_stats"]
    print(f"Final Value:         ${stats['final_value']:,.2f}")
    print(f"Total P&L:           ${stats['total_pnl']:,.2f}")
    print(f"Total Return:        {stats['total_return_pct']:.2f}%")
    print(f"Total Trades:        {stats['total_trades']}")
    print(f"Win Rate:            {stats['win_rate_pct']:.2f}%")

    if results["performance_metrics"]:
        print("\n--- Risk-Adjusted Metrics ---")
        metrics = results["performance_metrics"]
        print(f"Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"Volatility:          {metrics.get('volatility_pct', 0):.2f}%")

    print("\n--- Agent Behavior ---")
    action_counts = results["action_counts"]
    total_actions = sum(action_counts.values())
    print(f"HOLD actions:        {action_counts[0]} ({action_counts[0]/total_actions*100:.1f}%)")
    print(f"BUY actions:         {action_counts[1]} ({action_counts[1]/total_actions*100:.1f}%)")
    print(f"SELL actions:        {action_counts[2]} ({action_counts[2]/total_actions*100:.1f}%)")

    print("\n" + "="*70)


def plot_evaluation(results: dict, output_path: str = "rl/evaluation.png"):
    """Generate evaluation plots."""
    equity_curve = pd.DataFrame(results["equity_curve"])

    if equity_curve.empty:
        logger.warning("No equity data to plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Portfolio value over time
    axes[0].plot(equity_curve['timestamp'], equity_curve['total_value'])
    axes[0].set_title('Portfolio Value Over Time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].grid(True)
    axes[0].tick_params(axis='x', rotation=45)

    # Returns over time
    axes[1].plot(equity_curve['timestamp'], equity_curve['returns'] * 100)
    axes[1].set_title('Returns Over Time')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Return (%)')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    axes[1].grid(True)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Evaluation plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")

    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--symbol", default="BTC-USD", help="Trading symbol")
    parser.add_argument("--start", default="2024-01-01", help="Test start date")
    parser.add_argument("--end", default="2024-06-01", help="Test end date")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--render", action="store_true", help="Render steps")
    parser.add_argument("--output", default="rl/evaluation.png", help="Output plot path")

    args = parser.parse_args()

    # Need torch for evaluation
    import torch

    results = evaluate_agent(
        model_path=args.model,
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        deterministic=not args.stochastic,
        render=args.render
    )

    print_evaluation_results(results)
    plot_evaluation(results, args.output)


if __name__ == "__main__":
    main()

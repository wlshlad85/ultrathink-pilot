#!/usr/bin/env python3
"""
Evaluate Trained RL Model on 2024 Data
Tests model performance on completely unseen market data
"""
import sys
import os
from pathlib import Path
import torch
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl.ppo_agent import PPOAgent
from rl.trading_env import TradingEnv
from backtesting.data_fetcher import DataFetcher
from backtesting.metrics import PerformanceMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model_on_2024():
    """
    Evaluate the Phase 3 best model on 2024 Bitcoin data.
    This is completely unseen data that's even more recent than the test set.
    """

    print("=" * 80)
    print("EVALUATING TRAINED MODEL ON 2024 DATA")
    print("=" * 80)
    print()
    print("Evaluation Configuration:")
    print("  Symbol:        BTC-USD")
    print("  Date Range:    2024-01-01 to 2024-12-31")
    print("  Model:         phase3_test/best_model.pth (27.38% peak return)")
    print("  Capital:       $100,000")
    print("  Purpose:       Ultimate validation on unseen data")
    print("=" * 80)
    print()

    # Paths
    root_dir = Path(__file__).parent
    model_path = root_dir / "rl" / "models" / "phase3_test" / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from: {model_path}")
    print()

    # Step 1: Fetch 2024 data
    print("Fetching 2024 BTC-USD data...")
    fetcher = DataFetcher()
    data = fetcher.fetch(
        symbol="BTC-USD",
        start_date="2024-01-01",
        end_date="2024-12-31"
    )

    if data.empty:
        raise ValueError("No data fetched for 2024")

    print(f"✓ Fetched {len(data)} days of 2024 data")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    print(f"  Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print()

    # Step 2: Create trading environment
    print("Initializing trading environment...")
    env = TradingEnv(
        data=data,
        initial_capital=100000.0,
        commission_rate=0.001  # 0.1% per trade
    )

    print(f"✓ Environment ready")
    print(f"  State space: {env.observation_space.shape[0]} dimensions")
    print(f"  Action space: {env.action_space.n} actions (BUY/HOLD/SELL)")
    print()

    # Step 3: Load trained agent
    print("Loading trained PPO agent...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device
    )

    # Load the trained model weights
    checkpoint = torch.load(model_path, map_location=device)
    agent.policy.load_state_dict(checkpoint)
    agent.policy.eval()  # Set to evaluation mode

    print(f"✓ Model loaded successfully")
    print(f"  Device: {device}")
    print()

    # Step 4: Run evaluation episode
    print("Running evaluation episode on 2024 data...")
    print("-" * 80)

    state = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    trades_log = []
    equity_curve = []

    initial_capital = env.portfolio.initial_capital

    while not done:
        # Get action from trained policy (no exploration, deterministic)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs = agent.policy(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()

        # Take action in environment
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        step_count += 1

        # Log trade if executed
        if 'trade' in info:
            trades_log.append(info['trade'])
            print(f"  Day {step_count}: {info['trade']['action']} "
                  f"@ ${info['trade']['price']:.2f} - "
                  f"P&L: ${info['trade'].get('pnl', 0):.2f}")

        # Record equity
        equity_curve.append({
            'timestamp': env.current_date,
            'total_value': env.portfolio.get_total_value(env.current_price)
        })

        state = next_state

    # Step 5: Calculate performance metrics
    final_value = env.portfolio.get_total_value(env.current_price)
    total_return_pct = ((final_value - initial_capital) / initial_capital) * 100

    print("-" * 80)
    print()
    print("=" * 80)
    print("EVALUATION RESULTS - 2024 DATA")
    print("=" * 80)
    print()

    # Basic metrics
    print("Performance Summary:")
    print(f"  Initial Capital:     ${initial_capital:,.2f}")
    print(f"  Final Value:         ${final_value:,.2f}")
    print(f"  Total Return:        {total_return_pct:+.2f}%")
    print(f"  Total Reward:        {total_reward:.4f}")
    print(f"  Days Traded:         {step_count}")
    print(f"  Total Trades:        {len(trades_log)}")
    print()

    # Detailed metrics using PerformanceMetrics class
    if equity_curve:
        equity_df = pd.DataFrame(equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])

        metrics_calc = PerformanceMetrics(equity_df, risk_free_rate=0.02)
        detailed_metrics = metrics_calc.get_all_metrics()

        print("Risk-Adjusted Metrics:")
        print(f"  Sharpe Ratio:        {detailed_metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio:       {detailed_metrics['sortino_ratio']:.3f}")
        print(f"  Calmar Ratio:        {detailed_metrics['calmar_ratio']:.3f}")
        print()

        print("Risk Metrics:")
        print(f"  Max Drawdown:        {detailed_metrics['max_drawdown_pct']:.2f}%")
        print(f"  Volatility:          {detailed_metrics['volatility_pct']:.2f}%")
        print(f"  VaR (95%):           {detailed_metrics['var_95_pct']:.2f}%")
        print(f"  CVaR (95%):          {detailed_metrics['cvar_95_pct']:.2f}%")
        print()

    # Trade analysis
    if trades_log:
        trades_df = pd.DataFrame(trades_log)

        # Count profitable trades
        profitable_trades = [t for t in trades_log if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades_log if t.get('pnl', 0) < 0]

        print("Trade Statistics:")
        print(f"  Total Trades:        {len(trades_log)}")
        print(f"  Profitable Trades:   {len(profitable_trades)} ({len(profitable_trades)/len(trades_log)*100:.1f}%)")
        print(f"  Losing Trades:       {len(losing_trades)} ({len(losing_trades)/len(trades_log)*100:.1f}%)")

        if profitable_trades:
            avg_profit = np.mean([t['pnl'] for t in profitable_trades])
            print(f"  Avg Profit:          ${avg_profit:.2f}")

        if losing_trades:
            avg_loss = np.mean([t['pnl'] for t in losing_trades])
            print(f"  Avg Loss:            ${avg_loss:.2f}")
        print()

    # Comparison to training phases
    print("=" * 80)
    print("COMPARISON TO TRAINING PHASES")
    print("=" * 80)
    print()
    print("Historical Performance:")
    print(f"  Phase 1 (Train 2020-2021):     25.63% best return")
    print(f"  Phase 2 (Val 2022):             20.77% best return")
    print(f"  Phase 3 (Test 2023):            27.38% best return")
    print(f"  2024 Evaluation:                {total_return_pct:+.2f}% return")
    print()

    # Assessment
    if total_return_pct > 20:
        print("✅ EXCELLENT: Model maintained strong performance on 2024 data!")
        print("   Agent demonstrates robust generalization to completely unseen market conditions.")
    elif total_return_pct > 10:
        print("✅ GOOD: Model shows positive returns on 2024 data.")
        print("   Some performance degradation but still profitable.")
    elif total_return_pct > 0:
        print("⚠️  CAUTION: Model is profitable but underperforming historical results.")
        print("   Consider retraining with more recent data or adjusting hyperparameters.")
    else:
        print("❌ WARNING: Model lost money on 2024 data.")
        print("   Significant distribution shift or overfitting to training period.")
        print("   DO NOT deploy for live trading without retraining.")

    print()
    print("=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)

    return {
        'total_return_pct': total_return_pct,
        'final_value': final_value,
        'trades': len(trades_log),
        'days': step_count
    }


if __name__ == "__main__":
    try:
        results = evaluate_model_on_2024()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

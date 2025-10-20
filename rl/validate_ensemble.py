#!/usr/bin/env python3
"""
Validate ensemble strategy on held-out data.
Compares ensemble vs single models on unseen 2024 H2 data.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.ensemble_strategy import RegimeAdaptiveEnsemble, EnsembleTradingEnv
from rl.trading_env import TradingEnv
from rl.ppo_agent import PPOAgent
from backtesting.metrics import PerformanceMetrics
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleValidator:
    """Validate ensemble strategy on held-out data."""

    def __init__(
        self,
        symbol: str = "BTC-USD",
        validation_start: str = "2024-07-01",
        validation_end: str = "2024-12-31",
        initial_capital: float = 100000.0
    ):
        self.symbol = symbol
        self.validation_start = validation_start
        self.validation_end = validation_end
        self.initial_capital = initial_capital
        self.results = {}

    def evaluate_ensemble(self) -> Dict:
        """Evaluate ensemble strategy on validation data."""
        logger.info("="*70)
        logger.info("EVALUATING ENSEMBLE STRATEGY")
        logger.info("="*70)

        # Create environment
        env = TradingEnv(
            symbol=self.symbol,
            start_date=self.validation_start,
            end_date=self.validation_end,
            initial_capital=self.initial_capital
        )

        # Create ensemble
        ensemble = RegimeAdaptiveEnsemble(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )

        # Wrap environment
        ensemble_env = EnsembleTradingEnv(ensemble, env)

        # Run episode
        state, info = ensemble_env.reset()
        episode_reward = 0
        steps = 0

        while True:
            # Get ensemble action
            action, regime = ensemble_env.get_ensemble_action(state)

            # Execute step
            next_state, reward, terminated, truncated, info = ensemble_env.step(action)

            episode_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break

        # Get results
        portfolio_stats = env.get_portfolio_stats()
        equity_df = env.portfolio.get_equity_dataframe()
        trades_df = env.portfolio.get_trades_dataframe()

        # Calculate metrics
        if len(equity_df) > 1:
            metrics_calc = PerformanceMetrics(equity_df)
            performance_metrics = metrics_calc.get_all_metrics(trades_df)
        else:
            performance_metrics = {}

        # Get ensemble-specific stats
        ensemble_stats = ensemble.get_performance_summary()
        regime_timeline = ensemble.get_regime_timeline()

        results = {
            'strategy': 'Ensemble',
            'portfolio_stats': portfolio_stats,
            'performance_metrics': performance_metrics,
            'episode_reward': episode_reward,
            'steps': steps,
            'regime_stats': ensemble_stats,
            'regime_timeline': regime_timeline,
            'equity_curve': equity_df,
            'trades': trades_df
        }

        self.results['ensemble'] = results
        return results

    def evaluate_single_model(self, model_path: str, model_name: str) -> Dict:
        """Evaluate single model on validation data."""
        logger.info("="*70)
        logger.info(f"EVALUATING {model_name.upper()}")
        logger.info("="*70)

        # Create environment
        env = TradingEnv(
            symbol=self.symbol,
            start_date=self.validation_start,
            end_date=self.validation_end,
            initial_capital=self.initial_capital
        )

        # Load agent
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )
        agent.load(model_path)

        # Run episode
        state, info = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            # Get action (deterministic)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action_probs, _ = agent.policy(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()

            # Execute step
            next_state, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break

        # Get results
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
            'strategy': model_name,
            'portfolio_stats': portfolio_stats,
            'performance_metrics': performance_metrics,
            'episode_reward': episode_reward,
            'steps': steps,
            'equity_curve': equity_df,
            'trades': trades_df
        }

        self.results[model_name] = results
        return results

    def compare_strategies(self) -> pd.DataFrame:
        """Generate comparison table of all strategies."""
        comparison_data = []

        for strategy_name, results in self.results.items():
            stats = results['portfolio_stats']
            metrics = results['performance_metrics']

            row = {
                'Strategy': strategy_name,
                'Final Value ($)': stats.get('final_value', 0),
                'Total Return (%)': stats.get('total_return_pct', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Max Drawdown (%)': metrics.get('max_drawdown_pct', 0),
                'Volatility (%)': metrics.get('volatility_pct', 0),
                'Total Trades': stats.get('total_trades', 0),
                'Win Rate (%)': stats.get('win_rate_pct', 0),
                'Profit Factor': metrics.get('profit_factor', 0),
                'Avg Win ($)': stats.get('avg_win', 0),
                'Avg Loss ($)': stats.get('avg_loss', 0)
            }

            # Add regime stats for ensemble
            if 'regime_stats' in results:
                regime_dist = results['regime_stats']['regime_distribution']
                row['Bear (%)'] = regime_dist.get('bear', 0)
                row['Bull (%)'] = regime_dist.get('bull', 0)
                row['Neutral (%)'] = regime_dist.get('neutral', 0)

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        return df

    def calculate_market_baseline(self) -> Dict:
        """Calculate buy-and-hold baseline performance."""
        logger.info("="*70)
        logger.info("CALCULATING BUY-AND-HOLD BASELINE")
        logger.info("="*70)

        from backtesting.data_fetcher import DataFetcher

        fetcher = DataFetcher(self.symbol)
        fetcher.fetch(self.validation_start, self.validation_end)
        df = fetcher.data

        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        market_return = ((end_price / start_price) - 1) * 100

        logger.info(f"Start Price: ${start_price:,.2f}")
        logger.info(f"End Price: ${end_price:,.2f}")
        logger.info(f"Market Return: {market_return:+.2f}%")

        return {
            'start_price': start_price,
            'end_price': end_price,
            'market_return_pct': market_return
        }

    def generate_report(self, comparison_df: pd.DataFrame, market_baseline: Dict):
        """Generate formatted validation report."""
        print("\n" + "="*70)
        print("ENSEMBLE VALIDATION REPORT")
        print("="*70)
        print(f"\nValidation Period: {self.validation_start} to {self.validation_end}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Symbol: {self.symbol}")

        print("\n" + "="*70)
        print("MARKET BASELINE (BUY-AND-HOLD)")
        print("="*70)
        print(f"Market Return: {market_baseline['market_return_pct']:+.2f}%")
        print(f"Start Price: ${market_baseline['start_price']:,.2f}")
        print(f"End Price: ${market_baseline['end_price']:,.2f}")

        print("\n" + "="*70)
        print("STRATEGY COMPARISON")
        print("="*70)
        print("\n--- Core Performance Metrics ---")
        print(comparison_df[['Strategy', 'Final Value ($)', 'Total Return (%)',
                           'Sharpe Ratio', 'Max Drawdown (%)']].to_string(index=False))

        print("\n--- Trading Activity ---")
        print(comparison_df[['Strategy', 'Total Trades', 'Win Rate (%)',
                           'Profit Factor']].to_string(index=False))

        # Calculate alpha (vs market)
        print("\n--- Alpha vs Market ---")
        for _, row in comparison_df.iterrows():
            alpha = row['Total Return (%)'] - market_baseline['market_return_pct']
            print(f"{row['Strategy']:20s}: {alpha:+6.2f}%")

        # Regime distribution for ensemble
        ensemble_data = comparison_df[comparison_df['Strategy'] == 'Ensemble']
        if not ensemble_data.empty and 'Bear (%)' in ensemble_data.columns:
            print("\n--- Ensemble Regime Distribution ---")
            row = ensemble_data.iloc[0]
            print(f"BEAR market:    {row['Bear (%)']:5.1f}%")
            print(f"BULL market:    {row['Bull (%)']:5.1f}%")
            print(f"NEUTRAL market: {row['Neutral (%)']:5.1f}%")

        # Determine winner
        print("\n" + "="*70)
        print("VALIDATION RESULT")
        print("="*70)

        best_strategy = comparison_df.loc[comparison_df['Total Return (%)'].idxmax()]
        worst_strategy = comparison_df.loc[comparison_df['Total Return (%)'].idxmin()]

        print(f"\nBest Strategy: {best_strategy['Strategy']}")
        print(f"  Return: {best_strategy['Total Return (%)']:+.2f}%")
        print(f"  Sharpe: {best_strategy['Sharpe Ratio']:.2f}")

        print(f"\nWorst Strategy: {worst_strategy['Strategy']}")
        print(f"  Return: {worst_strategy['Total Return (%)']:+.2f}%")
        print(f"  Sharpe: {worst_strategy['Sharpe Ratio']:.2f}")

        # Check if ensemble won
        ensemble_row = comparison_df[comparison_df['Strategy'] == 'Ensemble']
        if not ensemble_row.empty:
            ensemble_return = ensemble_row.iloc[0]['Total Return (%)']
            is_best = ensemble_return == comparison_df['Total Return (%)'].max()

            if is_best:
                improvement = ensemble_return - worst_strategy['Total Return (%)']
                print(f"\n✓ ENSEMBLE VALIDATED: Outperformed by {improvement:.2f}pp")
            else:
                gap = best_strategy['Total Return (%)'] - ensemble_return
                print(f"\n✗ ENSEMBLE UNDERPERFORMED: Gap of {gap:.2f}pp to best")

        print("\n" + "="*70 + "\n")

    def save_results(self, comparison_df: pd.DataFrame, output_path: str = "ensemble_validation_results.csv"):
        """Save detailed results to CSV."""
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

        # Also save regime timeline for ensemble
        if 'ensemble' in self.results and 'regime_timeline' in self.results['ensemble']:
            timeline = self.results['ensemble']['regime_timeline']
            if not timeline.empty:
                timeline.to_csv("ensemble_regime_timeline.csv", index=False)
                logger.info("Regime timeline saved to ensemble_regime_timeline.csv")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate ensemble on held-out data")
    parser.add_argument("--symbol", default="BTC-USD", help="Trading symbol")
    parser.add_argument("--start", default="2024-07-01", help="Validation start date")
    parser.add_argument("--end", default="2024-12-31", help="Validation end date")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--output", default="ensemble_validation_results.csv", help="Output CSV path")

    args = parser.parse_args()

    # Create validator
    validator = EnsembleValidator(
        symbol=args.symbol,
        validation_start=args.start,
        validation_end=args.end,
        initial_capital=args.capital
    )

    print("\n" + "="*70)
    print("ENSEMBLE VALIDATION ON HELD-OUT DATA")
    print("="*70)
    print(f"Period: {args.start} to {args.end}")
    print(f"Symbol: {args.symbol}")
    print("="*70 + "\n")

    # Calculate market baseline
    market_baseline = validator.calculate_market_baseline()

    # Evaluate ensemble
    validator.evaluate_ensemble()

    # Evaluate competing single models
    single_models = [
        ("rl/models/best_model.pth", "Main Model"),
        ("rl/models/phase1_train/best_model.pth", "Phase 1"),
        ("rl/models/phase2_validation/best_model.pth", "Phase 2 (Bear Specialist)"),
        ("rl/models/phase3_test/best_model.pth", "Phase 3 (Bull Specialist)"),
    ]

    for model_path, model_name in single_models:
        if Path(model_path).exists():
            validator.evaluate_single_model(model_path, model_name)
        else:
            logger.warning(f"Model not found: {model_path}")

    # Generate comparison
    comparison_df = validator.compare_strategies()

    # Generate report
    validator.generate_report(comparison_df, market_baseline)

    # Save results
    validator.save_results(comparison_df, args.output)


if __name__ == "__main__":
    main()

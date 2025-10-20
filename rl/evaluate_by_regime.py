#!/usr/bin/env python3
"""
Evaluate RL models across different market regimes.
Identifies which models specialize in bull/bear/neutral markets.
"""
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.regime_detector import RegimeDetector, MarketRegime
from rl.evaluate import evaluate_agent
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeEvaluator:
    """Evaluate models across different market regimes."""

    def __init__(
        self,
        symbol: str = "BTC-USD",
        initial_capital: float = 100000.0
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.detector = RegimeDetector()
        self.results = []

    def evaluate_model_on_regime(
        self,
        model_path: str,
        regime: MarketRegime
    ) -> Dict:
        """
        Evaluate a single model on a specific market regime.

        Args:
            model_path: Path to model checkpoint
            regime: MarketRegime to test on

        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Evaluating {Path(model_path).name} on {regime.regime_type.upper()} "
                   f"({regime.start_date} to {regime.end_date})")

        try:
            results = evaluate_agent(
                model_path=model_path,
                symbol=self.symbol,
                start_date=regime.start_date,
                end_date=regime.end_date,
                initial_capital=self.initial_capital,
                deterministic=True,
                render=False
            )

            # Extract key metrics
            portfolio_stats = results.get('portfolio_stats', {})
            performance_metrics = results.get('performance_metrics', {})
            action_counts = results.get('action_counts', {})

            # Calculate action distribution
            total_actions = sum(action_counts.values())
            action_dist = {
                'hold_pct': action_counts.get(0, 0) / total_actions * 100 if total_actions > 0 else 0,
                'buy_pct': action_counts.get(1, 0) / total_actions * 100 if total_actions > 0 else 0,
                'sell_pct': action_counts.get(2, 0) / total_actions * 100 if total_actions > 0 else 0
            }

            result = {
                'model': Path(model_path).name,
                'regime': regime.regime_type,
                'start_date': regime.start_date,
                'end_date': regime.end_date,
                'market_return_pct': regime.price_change_pct,
                'portfolio_return_pct': portfolio_stats.get('total_return_pct', 0),
                'final_value': portfolio_stats.get('final_value', self.initial_capital),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown_pct': performance_metrics.get('max_drawdown_pct', 0),
                'volatility_pct': performance_metrics.get('volatility_pct', 0),
                'total_trades': portfolio_stats.get('total_trades', 0),
                'win_rate_pct': portfolio_stats.get('win_rate_pct', 0),
                'hold_pct': action_dist['hold_pct'],
                'buy_pct': action_dist['buy_pct'],
                'sell_pct': action_dist['sell_pct'],
                'alpha': portfolio_stats.get('total_return_pct', 0) - regime.price_change_pct,
                'success': True
            }

        except Exception as e:
            logger.error(f"Error evaluating {model_path}: {e}")
            result = {
                'model': Path(model_path).name,
                'regime': regime.regime_type,
                'start_date': regime.start_date,
                'end_date': regime.end_date,
                'error': str(e),
                'success': False
            }

        self.results.append(result)
        return result

    def evaluate_all_models(
        self,
        model_paths: List[str],
        regimes: List[MarketRegime]
    ) -> pd.DataFrame:
        """
        Evaluate all models across all regimes.

        Args:
            model_paths: List of paths to model checkpoints
            regimes: List of MarketRegime objects to test

        Returns:
            DataFrame with results
        """
        logger.info(f"Evaluating {len(model_paths)} models across {len(regimes)} regimes...")

        self.results = []

        for model_path in model_paths:
            for regime in regimes:
                self.evaluate_model_on_regime(model_path, regime)

        # Convert to DataFrame
        df = pd.DataFrame(self.results)

        # Filter successful evaluations
        df_success = df[df['success'] == True].copy()

        return df_success

    def generate_comparison_matrix(
        self,
        df: pd.DataFrame,
        metric: str = 'portfolio_return_pct'
    ) -> pd.DataFrame:
        """
        Generate pivot table comparing models across regimes.

        Args:
            df: Results DataFrame
            metric: Metric to compare (default: portfolio_return_pct)

        Returns:
            Pivot table with models as rows, regimes as columns
        """
        pivot = df.pivot_table(
            values=metric,
            index='model',
            columns='regime',
            aggfunc='mean'
        )

        # Add average performance
        pivot['average'] = pivot.mean(axis=1)

        # Sort by average
        pivot = pivot.sort_values('average', ascending=False)

        return pivot

    def identify_specialists(
        self,
        df: pd.DataFrame,
        threshold_percentile: float = 0.75
    ) -> Dict[str, str]:
        """
        Identify which models are specialists for each regime.

        Args:
            df: Results DataFrame
            threshold_percentile: Top percentile to consider as specialists

        Returns:
            Dictionary mapping regime -> best specialist model
        """
        specialists = {}

        for regime in df['regime'].unique():
            regime_df = df[df['regime'] == regime].copy()

            # Get top performers by return
            threshold = regime_df['portfolio_return_pct'].quantile(threshold_percentile)
            top_performers = regime_df[
                regime_df['portfolio_return_pct'] >= threshold
            ].sort_values('portfolio_return_pct', ascending=False)

            if len(top_performers) > 0:
                best_model = top_performers.iloc[0]['model']
                best_return = top_performers.iloc[0]['portfolio_return_pct']

                specialists[regime] = {
                    'model': best_model,
                    'return_pct': best_return,
                    'sharpe': top_performers.iloc[0]['sharpe_ratio'],
                    'trades': top_performers.iloc[0]['total_trades'],
                    'win_rate': top_performers.iloc[0]['win_rate_pct']
                }

        return specialists

    def save_results(
        self,
        df: pd.DataFrame,
        output_path: str = "regime_analysis_results.csv"
    ):
        """Save evaluation results to CSV."""
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")


def print_analysis(df: pd.DataFrame, specialists: Dict):
    """Print formatted analysis results."""
    print("\n" + "="*70)
    print("REGIME-BASED MODEL ANALYSIS")
    print("="*70)

    # Overall comparison matrix
    print("\n--- Performance Matrix (Portfolio Return %) ---")
    matrix = df.pivot_table(
        values='portfolio_return_pct',
        index='model',
        columns='regime',
        aggfunc='mean'
    )
    matrix['average'] = matrix.mean(axis=1)
    matrix = matrix.sort_values('average', ascending=False)
    print(matrix.round(2))

    # Sharpe comparison
    print("\n--- Risk-Adjusted Performance (Sharpe Ratio) ---")
    sharpe_matrix = df.pivot_table(
        values='sharpe_ratio',
        index='model',
        columns='regime',
        aggfunc='mean'
    )
    sharpe_matrix['average'] = sharpe_matrix.mean(axis=1)
    sharpe_matrix = sharpe_matrix.sort_values('average', ascending=False)
    print(sharpe_matrix.round(2))

    # Identified specialists
    print("\n--- Identified Specialists by Regime ---")
    for regime, info in specialists.items():
        print(f"\n{regime.upper()} Market:")
        print(f"  Best Model:    {info['model']}")
        print(f"  Return:        {info['return_pct']:.2f}%")
        print(f"  Sharpe Ratio:  {info['sharpe']:.2f}")
        print(f"  Win Rate:      {info['win_rate']:.1f}%")
        print(f"  Total Trades:  {info['trades']}")

    # Market baseline comparison
    print("\n--- Alpha Generation (vs Market) ---")
    alpha_matrix = df.pivot_table(
        values='alpha',
        index='model',
        columns='regime',
        aggfunc='mean'
    )
    alpha_matrix['average'] = alpha_matrix.mean(axis=1)
    alpha_matrix = alpha_matrix.sort_values('average', ascending=False)
    print(alpha_matrix.round(2))

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models across market regimes"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Model paths to evaluate (default: key professional models)"
    )
    parser.add_argument(
        "--symbol",
        default="BTC-USD",
        help="Trading symbol"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital"
    )
    parser.add_argument(
        "--output",
        default="regime_analysis_results.csv",
        help="Output CSV path"
    )

    args = parser.parse_args()

    # Default models if none specified
    if args.models is None:
        base_path = Path("rl/models/professional")
        args.models = [
            str(base_path / "episode_650.pth"),   # Mentioned in user's code
            str(base_path / "best_model.pth"),    # Current best
            str(base_path / "episode_300.pth"),   # High training performance
            str(base_path / "episode_1000.pth"),  # Best generalization
            "rl/models/phase2_validation/best_model.pth",  # Phase 2 mentioned
        ]

    # Verify models exist
    existing_models = [m for m in args.models if Path(m).exists()]
    if not existing_models:
        logger.error("No valid model paths found!")
        return

    logger.info(f"Evaluating {len(existing_models)} models...")

    # Get regimes
    detector = RegimeDetector()
    regimes = detector.get_historical_regimes(symbol=args.symbol)

    logger.info(f"Testing across {len(regimes)} market regimes:")
    for regime in regimes:
        logger.info(f"  - {regime.regime_type.upper()}: {regime.start_date} to {regime.end_date} "
                   f"({regime.price_change_pct:+.1f}%)")

    # Run evaluation
    evaluator = RegimeEvaluator(
        symbol=args.symbol,
        initial_capital=args.capital
    )

    results_df = evaluator.evaluate_all_models(existing_models, regimes)

    # Save results
    evaluator.save_results(results_df, args.output)

    # Identify specialists
    specialists = evaluator.identify_specialists(results_df)

    # Print analysis
    print_analysis(results_df, specialists)

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    # Check if specialization exists
    matrix = results_df.pivot_table(
        values='portfolio_return_pct',
        index='model',
        columns='regime',
        aggfunc='mean'
    )

    # Calculate variance in performance across regimes for each model
    regime_variance = matrix.var(axis=1).mean()

    if regime_variance > 100:  # High variance suggests specialization
        print("\nENSEMBLE APPROACH RECOMMENDED:")
        print("  Models show significant performance variation across regimes.")
        print("  Using regime-specific specialists could improve overall performance.")
        print("\nSuggested ensemble:")
        for regime, info in specialists.items():
            print(f"  {regime.upper()} market → {info['model']} ({info['return_pct']:.1f}%)")
    else:
        print("\nSINGLE MODEL APPROACH RECOMMENDED:")
        print("  Models show consistent performance across regimes.")
        best_overall = matrix.mean(axis=1).idxmax()
        avg_return = matrix.mean(axis=1).max()
        print(f"  Best generalist model: {best_overall}")
        print(f"  Average return: {avg_return:.2f}%")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()

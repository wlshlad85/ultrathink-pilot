"""
Meta-Controller Backtest Script
Validates meta-controller performance on 90 days of historical data

Compares:
- Hierarchical RL meta-controller vs. Naive regime-based routing
- Portfolio disruption metrics (churn rate)
- Strategy weight stability

Target: <5% portfolio disruption (vs 15% baseline)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import sys
import os

# Add module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services', 'meta_controller'))

from meta_controller_v2 import (
    MetaControllerRL,
    RegimeInput,
    StrategyWeights
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_regime_history(days: int = 90) -> pd.DataFrame:
    """
    Generate synthetic regime probability history for backtesting

    Args:
        days: Number of days of history

    Returns:
        DataFrame with regime probabilities over time
    """
    logger.info(f"Generating {days} days of synthetic regime history...")

    timestamps = pd.date_range(end=datetime.now(), periods=days * 24, freq='H')
    data = []

    # Simulate regime transitions
    np.random.seed(42)
    current_regime = 'bull'

    for t in timestamps:
        # Simulate regime persistence with occasional transitions
        if np.random.random() < 0.05:  # 5% chance of regime change per hour
            current_regime = np.random.choice(['bull', 'bear', 'sideways'])

        # Base probabilities based on current regime
        if current_regime == 'bull':
            base_probs = [0.7, 0.15, 0.15]
        elif current_regime == 'bear':
            base_probs = [0.15, 0.7, 0.15]
        else:  # sideways
            base_probs = [0.2, 0.2, 0.6]

        # Add noise for smooth transitions
        noise = np.random.dirichlet([10, 10, 10]) * 0.2
        probs = np.array(base_probs) + noise - 0.1

        # Normalize
        probs = np.maximum(probs, 0.01)
        probs = probs / probs.sum()

        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Confidence is inverse of entropy
        confidence = 1.0 - (entropy / np.log(3))

        # Market features
        volatility = 0.02 + np.random.uniform(-0.01, 0.01)
        trend = probs[0] - probs[1]  # Bull - Bear
        volume_ratio = 1.0 + np.random.uniform(-0.3, 0.3)

        data.append({
            'timestamp': t,
            'prob_bull': probs[0],
            'prob_bear': probs[1],
            'prob_sideways': probs[2],
            'entropy': entropy,
            'confidence': confidence,
            'volatility_20d': volatility,
            'trend_strength': trend,
            'volume_ratio': volume_ratio,
            'true_regime': current_regime
        })

    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} hourly regime samples")
    return df


def naive_regime_router(regime_probs: Dict[str, float]) -> StrategyWeights:
    """
    Naive baseline: Hard switch based on dominant regime

    This is the 15% disruption baseline we're trying to beat.
    """
    dominant = max(regime_probs.items(), key=lambda x: x[1])[0]

    if dominant == 'prob_bull':
        weights = [1.0, 0.0, 0.0, 0.0, 0.0]
    elif dominant == 'prob_bear':
        weights = [0.0, 1.0, 0.0, 0.0, 0.0]
    else:  # sideways
        weights = [0.0, 0.0, 1.0, 0.0, 0.0]

    return StrategyWeights(
        bull_specialist=weights[0],
        bear_specialist=weights[1],
        sideways_specialist=weights[2],
        momentum=weights[3],
        mean_reversion=weights[4],
        timestamp=datetime.utcnow(),
        method='naive_router'
    )


def calculate_portfolio_disruption(
    weight_history: List[StrategyWeights]
) -> Dict[str, float]:
    """
    Calculate portfolio disruption metrics

    Metrics:
    - Churn rate: Average absolute weight change between timesteps
    - Transition count: Number of times weights change significantly (>10%)
    - Max disruption: Maximum single-step weight change

    Args:
        weight_history: List of strategy weights over time

    Returns:
        Dictionary with disruption metrics
    """
    if len(weight_history) < 2:
        return {'churn_rate': 0.0, 'transition_count': 0, 'max_disruption': 0.0}

    churn_rates = []
    transition_count = 0
    max_disruption = 0.0

    for i in range(1, len(weight_history)):
        prev_weights = np.array([
            weight_history[i-1].bull_specialist,
            weight_history[i-1].bear_specialist,
            weight_history[i-1].sideways_specialist,
            weight_history[i-1].momentum,
            weight_history[i-1].mean_reversion
        ])

        curr_weights = np.array([
            weight_history[i].bull_specialist,
            weight_history[i].bear_specialist,
            weight_history[i].sideways_specialist,
            weight_history[i].momentum,
            weight_history[i].mean_reversion
        ])

        # Calculate absolute change
        abs_change = np.abs(curr_weights - prev_weights).sum() / 2.0  # Normalize
        churn_rates.append(abs_change)

        # Count significant transitions (>10% change in any weight)
        if np.max(np.abs(curr_weights - prev_weights)) > 0.1:
            transition_count += 1

        max_disruption = max(max_disruption, abs_change)

    return {
        'churn_rate': np.mean(churn_rates),
        'transition_count': transition_count,
        'max_disruption': max_disruption,
        'mean_disruption': np.mean(churn_rates),
        'std_disruption': np.std(churn_rates)
    }


def run_backtest(
    regime_history: pd.DataFrame,
    use_hierarchical_rl: bool = True
) -> Tuple[List[StrategyWeights], Dict]:
    """
    Run backtest on historical regime data

    Args:
        regime_history: DataFrame with regime probabilities
        use_hierarchical_rl: Use RL controller (True) or naive router (False)

    Returns:
        Tuple of (weight_history, metrics)
    """
    method_name = "Hierarchical RL" if use_hierarchical_rl else "Naive Router"
    logger.info(f"Running backtest with {method_name}...")

    if use_hierarchical_rl:
        controller = MetaControllerRL(device='cpu', epsilon=0.0)  # No exploration in backtest
    else:
        controller = None

    weight_history = []

    for idx, row in regime_history.iterrows():
        # Create regime input
        regime_input = RegimeInput(
            prob_bull=row['prob_bull'],
            prob_bear=row['prob_bear'],
            prob_sideways=row['prob_sideways'],
            entropy=row['entropy'],
            confidence=row['confidence'],
            timestamp=row['timestamp']
        )

        market_features = {
            'recent_pnl': 0.0,
            'volatility_20d': row['volatility_20d'],
            'trend_strength': row['trend_strength'],
            'volume_ratio': row['volume_ratio']
        }

        if use_hierarchical_rl:
            # Use RL controller
            weights = controller.predict_weights(
                regime_input=regime_input,
                market_features=market_features,
                use_epsilon_greedy=False
            )
        else:
            # Use naive router
            weights = naive_regime_router({
                'prob_bull': row['prob_bull'],
                'prob_bear': row['prob_bear'],
                'prob_sideways': row['prob_sideways']
            })

        weight_history.append(weights)

    # Calculate metrics
    metrics = calculate_portfolio_disruption(weight_history)
    metrics['method'] = method_name
    metrics['total_timesteps'] = len(weight_history)

    logger.info(f"{method_name} Results:")
    logger.info(f"  Churn rate: {metrics['churn_rate']:.4f} ({metrics['churn_rate']*100:.2f}%)")
    logger.info(f"  Transition count: {metrics['transition_count']}/{len(weight_history)}")
    logger.info(f"  Max disruption: {metrics['max_disruption']:.4f}")

    return weight_history, metrics


def compare_methods(regime_history: pd.DataFrame) -> Dict:
    """
    Compare hierarchical RL vs naive router

    Args:
        regime_history: DataFrame with regime probabilities

    Returns:
        Comparison metrics
    """
    logger.info("\n" + "="*80)
    logger.info("META-CONTROLLER BACKTEST COMPARISON")
    logger.info("="*80 + "\n")

    # Run hierarchical RL backtest
    rl_weights, rl_metrics = run_backtest(regime_history, use_hierarchical_rl=True)

    logger.info("")

    # Run naive router backtest
    naive_weights, naive_metrics = run_backtest(regime_history, use_hierarchical_rl=False)

    logger.info("\n" + "="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80)

    improvement = ((naive_metrics['churn_rate'] - rl_metrics['churn_rate']) /
                   naive_metrics['churn_rate'] * 100)

    logger.info(f"\nChurn Rate (Portfolio Disruption):")
    logger.info(f"  Naive Router:      {naive_metrics['churn_rate']:.4f} ({naive_metrics['churn_rate']*100:.2f}%)")
    logger.info(f"  Hierarchical RL:   {rl_metrics['churn_rate']:.4f} ({rl_metrics['churn_rate']*100:.2f}%)")
    logger.info(f"  Improvement:       {improvement:+.2f}%")

    logger.info(f"\nTransition Frequency:")
    logger.info(f"  Naive Router:      {naive_metrics['transition_count']} transitions")
    logger.info(f"  Hierarchical RL:   {rl_metrics['transition_count']} transitions")

    logger.info(f"\nMaximum Disruption:")
    logger.info(f"  Naive Router:      {naive_metrics['max_disruption']:.4f}")
    logger.info(f"  Hierarchical RL:   {rl_metrics['max_disruption']:.4f}")

    # Target validation
    logger.info(f"\n" + "-"*80)
    logger.info("TARGET VALIDATION")
    logger.info("-"*80)

    target_churn = 0.05  # 5% target
    baseline_churn = 0.15  # 15% baseline

    logger.info(f"Target:    <{target_churn*100:.0f}% portfolio disruption")
    logger.info(f"Baseline:  {baseline_churn*100:.0f}% (naive router)")
    logger.info(f"Achieved:  {rl_metrics['churn_rate']*100:.2f}%")

    if rl_metrics['churn_rate'] < target_churn:
        logger.info(f"✓ TARGET MET! ({rl_metrics['churn_rate']*100:.2f}% < {target_churn*100:.0f}%)")
    else:
        logger.info(f"✗ Target not met ({rl_metrics['churn_rate']*100:.2f}% >= {target_churn*100:.0f}%)")

    if rl_metrics['churn_rate'] < naive_metrics['churn_rate']:
        logger.info(f"✓ Better than naive baseline")
    else:
        logger.info(f"✗ Worse than naive baseline")

    logger.info("="*80 + "\n")

    return {
        'rl_metrics': rl_metrics,
        'naive_metrics': naive_metrics,
        'improvement_pct': improvement,
        'target_met': rl_metrics['churn_rate'] < target_churn,
        'better_than_baseline': rl_metrics['churn_rate'] < naive_metrics['churn_rate']
    }


def main():
    """Run full backtest validation"""
    # Generate 90 days of regime history
    regime_history = generate_synthetic_regime_history(days=90)

    # Save regime history
    output_path = 'regime_history_90d.csv'
    regime_history.to_csv(output_path, index=False)
    logger.info(f"\nSaved regime history to {output_path}")

    # Run comparison
    results = compare_methods(regime_history)

    # Save results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'test_period_days': 90,
        'total_timesteps': len(regime_history),
        **results
    }

    import json
    with open('meta_controller_backtest_results.json', 'w') as f:
        # Convert non-serializable values
        serializable_results = {}
        for key, value in results_summary.items():
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)

        json.dump(serializable_results, f, indent=2)

    logger.info("\nResults saved to meta_controller_backtest_results.json")

    return results


if __name__ == '__main__':
    results = main()

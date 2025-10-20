#!/usr/bin/env python3
"""
Deep-dive analysis of why Main Model outperformed.
Compares trading behavior, decision timing, and key differentiators.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.trading_env import TradingEnv
from rl.ppo_agent import PPOAgent
from backtesting.data_fetcher import DataFetcher
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelBehaviorAnalyzer:
    """Analyze and compare model trading behaviors."""

    def __init__(
        self,
        symbol: str = "BTC-USD",
        start_date: str = "2024-07-01",
        end_date: str = "2024-12-31",
        initial_capital: float = 100000.0
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        # Fetch market data once
        self.fetcher = DataFetcher(symbol)
        self.fetcher.fetch(start_date, end_date)
        self.fetcher.add_technical_indicators()
        self.market_data = self.fetcher.data

    def run_model_with_tracking(
        self,
        model_path: str,
        model_name: str
    ) -> Dict:
        """Run model and track every single decision."""
        logger.info(f"Analyzing {model_name}...")

        # Create environment
        env = TradingEnv(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital
        )

        # Load agent
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )
        agent.load(model_path)

        # Track every step
        decisions = []
        state, info = env.reset()
        step = 0

        while True:
            # Get action probabilities (not just final action)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action_probs, value_estimate = agent.policy(state_tensor)
                action_probs = action_probs.cpu().numpy()[0]
                value_estimate = value_estimate.cpu().item()
                action = np.argmax(action_probs)

            # Get market context
            market_row = self.market_data.iloc[env.current_idx]

            # Record decision
            decisions.append({
                'step': step,
                'date': str(market_row.name.date()) if hasattr(market_row.name, 'date') else str(market_row.name),
                'price': market_row['close'],
                'action': action,
                'action_name': ['HOLD', 'BUY', 'SELL'][action],
                'prob_hold': action_probs[0],
                'prob_buy': action_probs[1],
                'prob_sell': action_probs[2],
                'confidence': np.max(action_probs),
                'value_estimate': value_estimate,
                'portfolio_value': env.portfolio.get_total_value(),
                'cash': env.portfolio.cash,
                'position_value': env.portfolio.position.get_value(),
                'position_quantity': env.portfolio.position.quantity,
                'rsi_14': market_row.get('rsi_14', np.nan),
                'macd': market_row.get('macd', np.nan),
                'sma_20': market_row.get('sma_20', np.nan),
                'sma_50': market_row.get('sma_50', np.nan),
                'bb_position': (market_row['close'] - market_row.get('bb_lower', market_row['close'])) /
                              (market_row.get('bb_upper', market_row['close']) - market_row.get('bb_lower', market_row['close']) + 1e-8)
            })

            # Execute step
            next_state, reward, terminated, truncated, info = env.step(action)
            step += 1
            state = next_state

            if terminated or truncated:
                break

        # Get final portfolio stats
        portfolio_stats = env.get_portfolio_stats()
        equity_df = env.portfolio.get_equity_dataframe()
        trades_df = env.portfolio.get_trades_dataframe()

        return {
            'model_name': model_name,
            'decisions': pd.DataFrame(decisions),
            'portfolio_stats': portfolio_stats,
            'equity_curve': equity_df,
            'trades': trades_df
        }

    def compare_decision_timing(
        self,
        main_data: Dict,
        other_data: Dict
    ) -> pd.DataFrame:
        """Compare when models made different decisions."""
        main_decisions = main_data['decisions']
        other_decisions = other_data['decisions']

        # Merge on step
        comparison = main_decisions.merge(
            other_decisions,
            on='step',
            suffixes=('_main', '_other')
        )

        # Find disagreements
        comparison['disagreement'] = comparison['action_main'] != comparison['action_other']

        # Annotate with outcome
        comparison['price_change_next_5d'] = comparison['price_main'].shift(-5) / comparison['price_main'] - 1
        comparison['main_was_right'] = (
            ((comparison['action_main'] == 1) & (comparison['price_change_next_5d'] > 0)) |  # BUY before rise
            ((comparison['action_main'] == 2) & (comparison['price_change_next_5d'] < 0)) |  # SELL before fall
            ((comparison['action_main'] == 0))  # HOLD is neutral
        )

        return comparison

    def find_key_differentiators(
        self,
        comparison: pd.DataFrame
    ) -> Dict:
        """Identify what made Main Model's decisions better."""
        disagreements = comparison[comparison['disagreement'] == True].copy()

        if len(disagreements) == 0:
            return {'message': 'Models made identical decisions'}

        # Analyze disagreements
        analysis = {
            'total_disagreements': len(disagreements),
            'main_win_rate': (disagreements['main_was_right'].sum() / len(disagreements) * 100),

            # When did Main Model disagree?
            'disagreement_by_action': disagreements.groupby('action_main').size().to_dict(),

            # Market conditions during disagreements
            'avg_rsi_at_disagreement': disagreements['rsi_14_main'].mean(),
            'avg_price_at_disagreement': disagreements['price_main'].mean(),

            # Key disagreement types
            'main_bought_other_held': len(disagreements[(disagreements['action_main'] == 1) &
                                                        (disagreements['action_other'] == 0)]),
            'main_held_other_bought': len(disagreements[(disagreements['action_main'] == 0) &
                                                        (disagreements['action_other'] == 1)]),
            'main_sold_other_held': len(disagreements[(disagreements['action_main'] == 2) &
                                                       (disagreements['action_other'] == 0)]),
            'main_held_other_sold': len(disagreements[(disagreements['action_main'] == 0) &
                                                      (disagreements['action_other'] == 2)]),
        }

        # Find most impactful disagreements (largest price moves after)
        disagreements['price_change_abs'] = disagreements['price_change_next_5d'].abs()
        top_disagreements = disagreements.nlargest(10, 'price_change_abs')

        analysis['top_disagreements'] = top_disagreements[[
            'date_main', 'price_main', 'action_name_main', 'action_name_other',
            'price_change_next_5d', 'main_was_right'
        ]].to_dict('records')

        return analysis

    def analyze_action_patterns(
        self,
        model_data: Dict
    ) -> Dict:
        """Analyze model's action patterns and timing."""
        decisions = model_data['decisions']

        # Action distribution over time
        action_dist = decisions['action_name'].value_counts().to_dict()

        # Confidence analysis
        avg_confidence_by_action = decisions.groupby('action_name')['confidence'].mean().to_dict()

        # When does model take action?
        buy_conditions = decisions[decisions['action'] == 1]
        sell_conditions = decisions[decisions['action'] == 2]

        patterns = {
            'action_distribution': action_dist,
            'avg_confidence_by_action': avg_confidence_by_action,
            'total_steps': len(decisions),

            # BUY patterns
            'buy_analysis': {
                'count': len(buy_conditions),
                'avg_rsi': buy_conditions['rsi_14'].mean() if len(buy_conditions) > 0 else np.nan,
                'avg_price': buy_conditions['price'].mean() if len(buy_conditions) > 0 else np.nan,
                'avg_bb_position': buy_conditions['bb_position'].mean() if len(buy_conditions) > 0 else np.nan,
                'in_uptrend_pct': (buy_conditions['sma_20'] > buy_conditions['sma_50']).mean() * 100 if len(buy_conditions) > 0 else np.nan
            },

            # SELL patterns
            'sell_analysis': {
                'count': len(sell_conditions),
                'avg_rsi': sell_conditions['rsi_14'].mean() if len(sell_conditions) > 0 else np.nan,
                'avg_price': sell_conditions['price'].mean() if len(sell_conditions) > 0 else np.nan,
                'avg_bb_position': sell_conditions['bb_position'].mean() if len(sell_conditions) > 0 else np.nan,
                'in_downtrend_pct': (sell_conditions['sma_20'] < sell_conditions['sma_50']).mean() * 100 if len(sell_conditions) > 0 else np.nan
            }
        }

        return patterns

    def analyze_position_management(
        self,
        model_data: Dict
    ) -> Dict:
        """Analyze how model manages positions."""
        decisions = model_data['decisions']
        trades = model_data['trades']

        # Position holding patterns
        decisions['has_position'] = decisions['position_quantity'] > 0
        decisions['position_pct'] = decisions['position_value'] / decisions['portfolio_value'] * 100

        # Calculate consecutive holds
        decisions['action_change'] = decisions['action'] != decisions['action'].shift(1)
        decisions['streak_id'] = decisions['action_change'].cumsum()
        streaks = decisions.groupby(['streak_id', 'action_name']).size().reset_index(name='streak_length')

        patterns = {
            'avg_position_size_pct': decisions['position_pct'].mean(),
            'max_position_size_pct': decisions['position_pct'].max(),
            'time_with_position_pct': (decisions['has_position'].sum() / len(decisions) * 100),

            # Streak analysis
            'avg_hold_streak': streaks[streaks['action_name'] == 'HOLD']['streak_length'].mean(),
            'max_hold_streak': streaks[streaks['action_name'] == 'HOLD']['streak_length'].max(),
            'avg_buy_streak': streaks[streaks['action_name'] == 'BUY']['streak_length'].mean(),

            # Trade efficiency (from portfolio stats)
            'total_trades': model_data['portfolio_stats'].get('total_trades', 0),
            'win_rate': model_data['portfolio_stats'].get('win_rate_pct', 0),
            'avg_win': model_data['portfolio_stats'].get('avg_win', 0),
            'avg_loss': model_data['portfolio_stats'].get('avg_loss', 0)
        }

        return patterns

    def generate_report(
        self,
        main_data: Dict,
        comparisons: List[Tuple[str, Dict, Dict]]
    ):
        """Generate comprehensive analysis report."""
        print("\n" + "="*70)
        print("MAIN MODEL OUTPERFORMANCE ANALYSIS")
        print("="*70)
        print(f"\nPeriod: {self.start_date} to {self.end_date}")
        print(f"Symbol: {self.symbol}")

        # Main Model overview
        print("\n" + "="*70)
        print("MAIN MODEL BEHAVIOR")
        print("="*70)

        main_patterns = self.analyze_action_patterns(main_data)
        main_position = self.analyze_position_management(main_data)

        print("\n--- Action Distribution ---")
        for action, count in main_patterns['action_distribution'].items():
            pct = count / main_patterns['total_steps'] * 100
            print(f"{action:6s}: {count:4d} ({pct:5.1f}%)")

        print("\n--- Decision Confidence ---")
        for action, conf in main_patterns['avg_confidence_by_action'].items():
            print(f"{action:6s}: {conf:.3f}")

        print("\n--- Trading Patterns ---")
        buy = main_patterns['buy_analysis']
        print(f"BUY Decisions:  {buy['count']:3d}")
        print(f"  Avg RSI:      {buy['avg_rsi']:.1f}")
        print(f"  Avg Price:    ${buy['avg_price']:,.0f}")
        print(f"  In Uptrend:   {buy['in_uptrend_pct']:.1f}%")

        sell = main_patterns['sell_analysis']
        print(f"\nSELL Decisions: {sell['count']:3d}")
        print(f"  Avg RSI:      {sell['avg_rsi']:.1f}")
        print(f"  Avg Price:    ${sell['avg_price']:,.0f}")
        print(f"  In Downtrend: {sell['in_downtrend_pct']:.1f}%")

        print("\n--- Position Management ---")
        print(f"Avg Position Size:    {main_position['avg_position_size_pct']:.1f}%")
        print(f"Time With Position:   {main_position['time_with_position_pct']:.1f}%")
        print(f"Total Trades:         {main_position['total_trades']}")
        print(f"Win Rate:             {main_position['win_rate']:.1f}%")
        print(f"Avg Win:              ${main_position['avg_win']:,.2f}")
        print(f"Avg Loss:             ${main_position['avg_loss']:,.2f}")

        # Comparisons
        for other_name, other_data, differentiators in comparisons:
            print("\n" + "="*70)
            print(f"MAIN MODEL vs {other_name.upper()}")
            print("="*70)

            if 'message' in differentiators:
                print(f"\n{differentiators['message']}")
                continue

            print(f"\nTotal Disagreements: {differentiators['total_disagreements']}")
            print(f"Main Model Win Rate: {differentiators['main_win_rate']:.1f}%")

            print("\n--- Key Disagreement Types ---")
            print(f"Main BOUGHT, {other_name} HELD: {differentiators['main_bought_other_held']}")
            print(f"Main HELD, {other_name} BOUGHT: {differentiators['main_held_other_bought']}")
            print(f"Main SOLD, {other_name} HELD:   {differentiators['main_sold_other_held']}")
            print(f"Main HELD, {other_name} SOLD:   {differentiators['main_held_other_sold']}")

            print("\n--- Top 5 Most Impactful Disagreements ---")
            for i, disagreement in enumerate(differentiators['top_disagreements'][:5], 1):
                correct = "[CORRECT]" if disagreement['main_was_right'] else "[WRONG]"
                print(f"\n{i}. {disagreement['date_main']} @ ${disagreement['price_main']:,.0f}")
                print(f"   Main: {disagreement['action_name_main']:4s} | "
                      f"{other_name}: {disagreement['action_name_other']:4s}")
                print(f"   Next 5d: {disagreement['price_change_next_5d']*100:+.1f}% {correct}")

            # Compare patterns
            other_patterns = self.analyze_action_patterns(other_data)
            print("\n--- Pattern Comparison ---")
            print(f"{'Metric':<30} {'Main':>12} {other_name:>12}")
            print("-" * 56)

            for action in ['HOLD', 'BUY', 'SELL']:
                main_count = main_patterns['action_distribution'].get(action, 0)
                other_count = other_patterns['action_distribution'].get(action, 0)
                main_pct = main_count / main_patterns['total_steps'] * 100
                other_pct = other_count / other_patterns['total_steps'] * 100
                print(f"{action + ' %':<30} {main_pct:>11.1f}% {other_pct:>11.1f}%")

        print("\n" + "="*70)
        print("KEY INSIGHTS")
        print("="*70)

        # Synthesize insights
        insights = self.synthesize_insights(main_data, comparisons)
        for i, insight in enumerate(insights, 1):
            print(f"\n{i}. {insight}")

        print("\n" + "="*70 + "\n")

    def synthesize_insights(
        self,
        main_data: Dict,
        comparisons: List[Tuple[str, Dict, Dict]]
    ) -> List[str]:
        """Generate key insights from analysis."""
        insights = []

        main_patterns = self.analyze_action_patterns(main_data)
        main_position = self.analyze_position_management(main_data)

        # Insight 1: Action aggressiveness
        buy_pct = main_patterns['action_distribution'].get('BUY', 0) / main_patterns['total_steps'] * 100
        if buy_pct > 15:
            insights.append(f"AGGRESSIVE BUYING: Main Model takes BUY action {buy_pct:.1f}% of the time, "
                          "suggesting it's optimized for capturing upside in bull markets")
        elif buy_pct < 5:
            insights.append(f"CONSERVATIVE APPROACH: Main Model only buys {buy_pct:.1f}% of the time, "
                          "focusing on selective high-conviction trades")

        # Insight 2: Position management
        if main_position['time_with_position_pct'] > 70:
            insights.append(f"STAY INVESTED: Main Model holds positions {main_position['time_with_position_pct']:.1f}% "
                          "of the time, minimizing cash drag in rising markets")

        # Insight 3: Win rate vs trade count
        if main_position['win_rate'] > 60 and main_position['total_trades'] < 50:
            insights.append(f"QUALITY OVER QUANTITY: {main_position['total_trades']} selective trades "
                          f"with {main_position['win_rate']:.1f}% win rate shows patience and precision")

        # Insight 4: Comparison insights
        for other_name, other_data, differentiators in comparisons:
            if 'message' in differentiators:
                continue

            if differentiators['main_win_rate'] > 60:
                insights.append(f"SUPERIOR TIMING vs {other_name}: Main Model's decisions were correct "
                              f"{differentiators['main_win_rate']:.1f}% of the time when disagreeing")

            # Check if Main bought when others held
            if differentiators['main_bought_other_held'] > 10:
                insights.append(f"EARLY BULL RECOGNITION: Main Model bought {differentiators['main_bought_other_held']} times "
                              f"when {other_name} held, capturing early uptrend opportunities")

        # Insight 5: Technical indicator usage
        buy = main_patterns['buy_analysis']
        if buy['avg_rsi'] < 50:
            insights.append(f"CONTRARIAN BUYING: Avg RSI at BUY = {buy['avg_rsi']:.1f}, "
                          "suggesting Main Model buys dips rather than chasing momentum")
        elif buy['avg_rsi'] > 60:
            insights.append(f"MOMENTUM FOLLOWING: Avg RSI at BUY = {buy['avg_rsi']:.1f}, "
                          "suggesting Main Model rides strong trends")

        return insights


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze why Main Model outperformed")
    parser.add_argument("--symbol", default="BTC-USD", help="Trading symbol")
    parser.add_argument("--start", default="2024-07-01", help="Analysis start date")
    parser.add_argument("--end", default="2024-12-31", help="Analysis end date")
    parser.add_argument("--save-decisions", action="store_true", help="Save detailed decision logs")

    args = parser.parse_args()

    # Create analyzer
    analyzer = ModelBehaviorAnalyzer(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end
    )

    # Analyze Main Model
    main_data = analyzer.run_model_with_tracking(
        "rl/models/best_model.pth",
        "Main Model"
    )

    # Analyze competitors
    competitors = [
        ("rl/models/phase3_test/best_model.pth", "Phase 3 (Bull Specialist)"),
        ("rl/models/phase2_validation/best_model.pth", "Phase 2 (Bear Specialist)"),
    ]

    comparisons = []
    for model_path, model_name in competitors:
        if Path(model_path).exists():
            other_data = analyzer.run_model_with_tracking(model_path, model_name)
            comparison = analyzer.compare_decision_timing(main_data, other_data)
            differentiators = analyzer.find_key_differentiators(comparison)
            comparisons.append((model_name, other_data, differentiators))

            # Optionally save detailed comparison
            if args.save_decisions:
                output_file = f"decision_comparison_main_vs_{model_name.replace(' ', '_').lower()}.csv"
                comparison.to_csv(output_file, index=False)
                logger.info(f"Saved decision comparison to {output_file}")

    # Generate report
    analyzer.generate_report(main_data, comparisons)

    # Save main model decisions
    if args.save_decisions:
        main_data['decisions'].to_csv("main_model_decisions.csv", index=False)
        logger.info("Saved Main Model decisions to main_model_decisions.csv")


if __name__ == "__main__":
    main()

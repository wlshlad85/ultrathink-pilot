#!/usr/bin/env python3
"""
Trade Decision Forensics Engine

Analyzes trading decisions made by RL models to identify systematic
failure patterns and understand why specific mistakes were made.

This is crucial for iterative model improvement - you can't fix what
you don't understand.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rl.trading_env import TradingEnv
from rl.trading_env_v2 import TradingEnvV2
from rl.ppo_agent import PPOAgent
from rl.regime_detector import RegimeDetector, RegimeType
from trade_decision import TradeDecision, ForensicsReport, ActionType, FailurePattern
from backtesting.data_fetcher import DataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeForensics:
    """
    Forensic analysis engine for RL trading models.

    Replays model decisions through historical data and captures
    complete context for each decision to enable detailed analysis
    of failure patterns.
    """

    def __init__(
        self,
        model_path: str,
        state_dim: int = None,
        action_dim: int = 3,
        device: str = None
    ):
        """
        Initialize forensics engine.

        Args:
            model_path: Path to trained model checkpoint
            state_dim: State space dimension (auto-detected if None)
            action_dim: Action space dimension (3: HOLD, BUY, SELL)
            device: Compute device ('cuda' or 'cpu')
        """
        # Setup device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Load checkpoint to detect state_dim if not provided
        checkpoint = torch.load(model_path, map_location=self.device)

        # Auto-detect state_dim from checkpoint if not specified
        if state_dim is None:
            if 'policy_state_dict' in checkpoint:
                first_layer = checkpoint['policy_state_dict']['feature_extractor.0.weight']
                state_dim = first_layer.shape[1]
                logger.info(f"Auto-detected state_dim: {state_dim}")
            else:
                # Try to detect from raw state_dict
                for key in checkpoint.keys():
                    if 'feature_extractor.0.weight' in key:
                        state_dim = checkpoint[key].shape[1]
                        logger.info(f"Auto-detected state_dim: {state_dim}")
                        break
                if state_dim is None:
                    state_dim = 43  # Fallback to old default
                    logger.warning(f"Could not auto-detect state_dim, using default: {state_dim}")

        # Store for environment creation
        self.state_dim = state_dim
        self.use_regime_env = (state_dim == 53)  # TradingEnvV2 has 53 features

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )

        # Load weights
        if 'policy_state_dict' in checkpoint:
            self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        else:
            self.agent.policy.load_state_dict(checkpoint)

        self.agent.policy.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully")

        # Regime detector for market classification
        self.regime_detector = RegimeDetector(
            bull_threshold=0.10,
            bear_threshold=-0.10,
            lookback_days=60
        )

        # Storage for decisions
        self.decisions: List[TradeDecision] = []

    def analyze_period(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        failure_threshold_pct: float = -5.0
    ) -> List[TradeDecision]:
        """
        Analyze model decisions over a specific time period.

        This is the main analysis function that replays the model
        through historical data and captures all decision context.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            start_date: Analysis start date
            end_date: Analysis end date
            initial_capital: Starting capital
            failure_threshold_pct: Return threshold for bad decision

        Returns:
            List of TradeDecision objects
        """
        logger.info(f"Analyzing period: {start_date} to {end_date}")

        # Create environment (use TradingEnvV2 for regime-aware models)
        if self.use_regime_env:
            logger.info("Using TradingEnvV2 (regime-aware) for forensics")
            env = TradingEnvV2(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                commission_rate=0.001,
                use_regime_rewards=False  # Don't need rewards for forensics
            )
        else:
            logger.info("Using TradingEnv (standard) for forensics")
            env = TradingEnv(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                commission_rate=0.001
            )

        # Get market data for regime detection and forward returns
        market_data = env.market_data.copy()

        # Reset environment
        state, info = env.reset()
        decisions = []

        # Track portfolio for cost calculations
        portfolio_values = [initial_capital]

        # Step through environment
        step = 0
        while True:
            # Get model prediction with probabilities
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs, state_value = self.agent.policy(state_tensor)
                action_probs = action_probs.cpu().numpy()[0]
                action = np.argmax(action_probs)

            # Convert action to string
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            action_str = action_map[action]

            # Get current market row
            current_idx = env.current_idx
            current_row = market_data.iloc[current_idx]

            # Extract market indicators
            indicators = {
                'rsi_14': current_row.get('rsi_14', 50),
                'macd': current_row.get('macd', 0),
                'macd_signal': current_row.get('macd_signal', 0),
                'atr_14': current_row.get('atr_14', 0),
                'sma_20': current_row.get('sma_20', 0),
                'sma_50': current_row.get('sma_50', 0),
                'volume_ratio': current_row.get('volume_ratio', 1.0),
                'volatility': current_row.get('returns_1d', 0),
            }

            # Detect market regime
            regime = self.regime_detector.detect_regime(market_data, current_idx)

            # Calculate forward returns
            returns_1d = self._calculate_forward_return(market_data, current_idx, 1)
            returns_5d = self._calculate_forward_return(market_data, current_idx, 5)
            returns_20d = self._calculate_forward_return(market_data, current_idx, 20)

            # Create decision record
            decision = TradeDecision(
                timestamp=info['date'],
                price=info['price'],
                action=action_str,
                action_probs=action_probs,
                confidence=action_probs[action],
                state=state.copy(),
                portfolio_value=info['portfolio_value'],
                cash_ratio=env.portfolio.cash / info['portfolio_value'] if info['portfolio_value'] > 0 else 1.0,
                position_ratio=info['position_value'] / info['portfolio_value'] if info['portfolio_value'] > 0 else 0.0,
                total_return=info['total_return'],
                indicators=indicators,
                regime=regime,
                returns_1d=returns_1d,
                returns_5d=returns_5d,
                returns_20d=returns_20d
            )

            # Classify decision as good or bad
            self._classify_decision(decision, failure_threshold_pct)

            # Calculate cost of mistake if applicable
            if decision.is_bad_decision:
                decision.cost_of_mistake = self._calculate_mistake_cost(
                    decision, env.portfolio, portfolio_values[-1]
                )

            decisions.append(decision)

            # Execute action in environment
            state, reward, terminated, truncated, info = env.step(action)
            portfolio_values.append(info['portfolio_value'])

            step += 1
            if step % 50 == 0:
                logger.info(f"  Processed {step} decisions...")

            if terminated or truncated:
                break

        logger.info(f"Analysis complete. Captured {len(decisions)} decisions")
        self.decisions = decisions
        return decisions

    def _calculate_forward_return(
        self,
        market_data: pd.DataFrame,
        current_idx: int,
        days_forward: int
    ) -> Optional[float]:
        """
        Calculate forward return from current position.

        Args:
            market_data: Full market data DataFrame
            current_idx: Current position in data
            days_forward: Number of days to look forward

        Returns:
            Forward return as percentage, or None if data unavailable
        """
        future_idx = current_idx + days_forward

        if future_idx >= len(market_data):
            return None

        current_price = market_data.iloc[current_idx]['close']
        future_price = market_data.iloc[future_idx]['close']

        return ((future_price / current_price) - 1) * 100

    def _classify_decision(
        self,
        decision: TradeDecision,
        failure_threshold_pct: float
    ):
        """
        Classify decision as good/bad and identify failure pattern.

        This is where we identify systematic errors in the model's
        decision-making process.

        Args:
            decision: TradeDecision object to classify
            failure_threshold_pct: Threshold for bad decision
        """
        # Check if it was a bad decision
        if decision.action == "BUY":
            if decision.returns_5d is not None and decision.returns_5d < failure_threshold_pct:
                decision.is_bad_decision = True
            elif decision.returns_20d is not None and decision.returns_20d < failure_threshold_pct * 2:
                decision.is_bad_decision = True

        elif decision.action in ["HOLD", "BUY"]:
            # Missed sell opportunity
            if decision.position_ratio > 0.1:  # We have a position
                if decision.returns_5d is not None and decision.returns_5d < failure_threshold_pct:
                    decision.is_bad_decision = True

        # If not bad, mark as success
        if not decision.is_bad_decision:
            decision.failure_pattern = "success"
            return

        # Identify specific failure pattern
        decision.failure_pattern = self._identify_failure_pattern(decision)

    def _identify_failure_pattern(self, decision: TradeDecision) -> FailurePattern:
        """
        Identify what TYPE of mistake this was.

        This categorization helps identify which patterns the model
        needs to learn to avoid.

        Args:
            decision: TradeDecision marked as bad

        Returns:
            FailurePattern classification
        """
        indicators = decision.indicators
        regime = decision.regime

        # Pattern 1: Caught falling knife
        # Bought "oversold" (RSI < 30) but price continued down
        if decision.action == "BUY" and indicators.get('rsi_14', 50) < 30:
            if decision.returns_5d is not None and decision.returns_5d < -5:
                return "caught_falling_knife"

        # Pattern 2: Fought the trend
        # Bought below major moving averages (counter-trend)
        price = decision.price
        sma_50 = indicators.get('sma_50', price)
        if decision.action == "BUY" and price < sma_50 * 0.95:  # > 5% below SMA50
            return "fought_the_trend"

        # Pattern 3: Bear market denial
        # Bought during confirmed bear regime
        if decision.action == "BUY" and regime == "bear":
            return "bear_market_denial"

        # Pattern 4: High volatility mistake
        # Traded during extreme volatility
        volatility = abs(indicators.get('volatility', 0))
        if volatility > 0.05:  # > 5% daily volatility
            return "high_volatility_mistake"

        # Pattern 5: Momentum misread
        # Bought on false positive momentum signals
        if decision.action == "BUY":
            rsi = indicators.get('rsi_14', 50)
            if 40 < rsi < 60:  # Neutral RSI, not oversold
                return "momentum_misread"

        return "other"

    def _calculate_mistake_cost(
        self,
        decision: TradeDecision,
        portfolio,
        prev_portfolio_value: float
    ) -> float:
        """
        Estimate the dollar cost of a bad decision.

        For BUY decisions: How much we lost by buying
        For HOLD/missed SELL: Opportunity cost of not selling

        Args:
            decision: TradeDecision object
            portfolio: Portfolio object
            prev_portfolio_value: Portfolio value before this decision

        Returns:
            Estimated cost in dollars
        """
        if decision.action == "BUY":
            # Cost is the loss on this trade
            if decision.returns_5d is not None:
                trade_size = prev_portfolio_value * 0.20  # Assume 20% position
                cost = abs(trade_size * (decision.returns_5d / 100))
                return cost

        elif decision.action in ["HOLD", "BUY"]:
            # Opportunity cost of not selling
            position_value = decision.position_ratio * decision.portfolio_value
            if decision.returns_5d is not None and position_value > 0:
                cost = abs(position_value * (decision.returns_5d / 100))
                return cost

        return 0.0

    def generate_report(
        self,
        decisions: Optional[List[TradeDecision]] = None
    ) -> ForensicsReport:
        """
        Generate comprehensive forensics report from decisions.

        Args:
            decisions: List of TradeDecision objects (uses self.decisions if None)

        Returns:
            ForensicsReport object
        """
        if decisions is None:
            decisions = self.decisions

        if not decisions:
            raise ValueError("No decisions to analyze")

        # Basic counts
        total_decisions = len(decisions)
        num_holds = sum(1 for d in decisions if d.action == "HOLD")
        num_buys = sum(1 for d in decisions if d.action == "BUY")
        num_sells = sum(1 for d in decisions if d.action == "SELL")

        # Failure analysis
        bad_decisions = [d for d in decisions if d.is_bad_decision]
        num_bad = len(bad_decisions)
        bad_rate = num_bad / total_decisions if total_decisions > 0 else 0
        total_cost = sum(d.cost_of_mistake for d in bad_decisions)

        # Pattern breakdown
        pattern_counts = defaultdict(int)
        pattern_costs = defaultdict(float)

        for decision in bad_decisions:
            pattern_counts[decision.failure_pattern] += 1
            pattern_costs[decision.failure_pattern] += decision.cost_of_mistake

        # Regime performance
        regime_stats = defaultdict(lambda: {'total': 0, 'bad': 0, 'cost': 0.0})

        for decision in decisions:
            regime_stats[decision.regime]['total'] += 1
            if decision.is_bad_decision:
                regime_stats[decision.regime]['bad'] += 1
                regime_stats[decision.regime]['cost'] += decision.cost_of_mistake

        # Calculate bad rates by regime
        for regime in regime_stats:
            total = regime_stats[regime]['total']
            bad = regime_stats[regime]['bad']
            regime_stats[regime]['bad_rate'] = bad / total if total > 0 else 0

        # Top failures (most costly)
        top_failures = sorted(
            bad_decisions,
            key=lambda d: d.cost_of_mistake,
            reverse=True
        )

        # Create report
        report = ForensicsReport(
            period_start=decisions[0].timestamp,
            period_end=decisions[-1].timestamp,
            total_decisions=total_decisions,
            num_holds=num_holds,
            num_buys=num_buys,
            num_sells=num_sells,
            num_bad_decisions=num_bad,
            bad_decision_rate=bad_rate,
            total_cost_of_mistakes=total_cost,
            pattern_counts=dict(pattern_counts),
            pattern_costs=dict(pattern_costs),
            regime_performance=dict(regime_stats),
            top_failures=top_failures
        )

        return report

    def export_decisions(
        self,
        filepath: str,
        decisions: Optional[List[TradeDecision]] = None
    ):
        """
        Export decisions to CSV for further analysis.

        Args:
            filepath: Output file path
            decisions: List of decisions (uses self.decisions if None)
        """
        if decisions is None:
            decisions = self.decisions

        if not decisions:
            logger.warning("No decisions to export")
            return

        # Convert to DataFrame
        df = pd.DataFrame([d.to_dict() for d in decisions])

        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(decisions)} decisions to {filepath}")


if __name__ == "__main__":
    """
    Test forensics on a small period.
    """
    print("Testing Trade Forensics Engine...")

    model_path = "rl/models/professional/best_model.pth"

    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    # Create forensics engine
    forensics = TradeForensics(model_path=model_path)

    # Analyze a short test period
    decisions = forensics.analyze_period(
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2022-03-31",  # Q1 2022 - early bear market
        initial_capital=100000.0,
        failure_threshold_pct=-5.0
    )

    # Generate report
    report = forensics.generate_report()
    report.print_summary()

    # Export to CSV
    forensics.export_decisions("forensics_test_output.csv")

    print("\nTest complete!")

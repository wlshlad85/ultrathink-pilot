#!/usr/bin/env python3
"""
Regime-Adaptive Ensemble Strategy for trading.
Routes predictions to specialist models based on detected market regime.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.regime_detector import RegimeDetector, RegimeType
from rl.ppo_agent import PPOAgent
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeAdaptiveEnsemble:
    """
    Ensemble of specialist models for different market regimes.

    Based on empirical analysis showing:
    - Phase 2 model: Best bear market performance (-1.13% vs -65% market)
    - Phase 3 model: Best bull market performance (+9.72% vs +83% market)
    - Phase 2 model: Best neutral market performance (-0.33%)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        bear_model_path: str = "rl/models/phase2_validation/best_model.pth",
        bull_model_path: str = "rl/models/phase3_test/best_model.pth",
        neutral_model_path: str = "rl/models/phase2_validation/best_model.pth",
        device: str = None
    ):
        """
        Initialize ensemble with specialist models.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            bear_model_path: Path to bear market specialist
            bull_model_path: Path to bull market specialist
            neutral_model_path: Path to neutral market specialist
            device: Device to run on ('cuda' or 'cpu')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        logger.info(f"Initializing Regime-Adaptive Ensemble on {self.device}")

        # Load specialist models
        self.models = {}

        logger.info(f"Loading BEAR specialist: {bear_model_path}")
        self.models['bear'] = PPOAgent(state_dim, action_dim, device=str(self.device))
        self.models['bear'].load(bear_model_path)

        logger.info(f"Loading BULL specialist: {bull_model_path}")
        self.models['bull'] = PPOAgent(state_dim, action_dim, device=str(self.device))
        self.models['bull'].load(bull_model_path)

        # Note: neutral and bear might be same model
        if neutral_model_path != bear_model_path:
            logger.info(f"Loading NEUTRAL specialist: {neutral_model_path}")
            self.models['neutral'] = PPOAgent(state_dim, action_dim, device=str(self.device))
            self.models['neutral'].load(neutral_model_path)
        else:
            logger.info("NEUTRAL specialist shares model with BEAR")
            self.models['neutral'] = self.models['bear']

        # Regime detector
        self.regime_detector = RegimeDetector()

        # Performance tracking
        self.regime_history = []
        self.predictions_by_regime = {'bear': 0, 'bull': 0, 'neutral': 0}
        self.current_regime = None

    def detect_regime(
        self,
        df: pd.DataFrame,
        current_idx: int
    ) -> RegimeType:
        """
        Detect current market regime.

        Args:
            df: Historical market data with indicators
            current_idx: Current position in dataframe

        Returns:
            Detected regime (bear/bull/neutral)
        """
        regime = self.regime_detector.detect_regime(df, current_idx)
        self.current_regime = regime
        self.regime_history.append({
            'idx': current_idx,
            'regime': regime,
            'price': df.iloc[current_idx]['close']
        })
        return regime

    def select_action(
        self,
        state: np.ndarray,
        regime: Optional[RegimeType] = None,
        deterministic: bool = False
    ) -> int:
        """
        Select action using appropriate specialist model.

        Args:
            state: Current state observation
            regime: Market regime (if already detected, else uses current_regime)
            deterministic: Use greedy policy instead of sampling

        Returns:
            Selected action (0=HOLD, 1=BUY, 2=SELL)
        """
        # Use provided regime or current detected regime
        active_regime = regime if regime is not None else self.current_regime

        if active_regime is None:
            logger.warning("No regime detected, defaulting to neutral")
            active_regime = 'neutral'

        # Select appropriate specialist
        specialist = self.models[active_regime]
        self.predictions_by_regime[active_regime] += 1

        # Get action from specialist
        if deterministic:
            # Greedy policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs, _ = specialist.policy(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()
        else:
            # Stochastic policy (for training)
            action = specialist.select_action(state)

        return action

    def predict(
        self,
        state: np.ndarray,
        df: pd.DataFrame,
        current_idx: int,
        deterministic: bool = True
    ) -> Tuple[int, RegimeType]:
        """
        Full prediction pipeline: detect regime + select action.

        Args:
            state: Current state observation
            df: Market data for regime detection
            current_idx: Current index in dataframe
            deterministic: Use greedy policy

        Returns:
            (action, detected_regime)
        """
        # Detect regime
        regime = self.detect_regime(df, current_idx)

        # Select action
        action = self.select_action(state, regime, deterministic)

        return action, regime

    def get_performance_summary(self) -> Dict:
        """Get performance summary showing regime distribution."""
        total_predictions = sum(self.predictions_by_regime.values())

        summary = {
            'total_predictions': total_predictions,
            'regime_distribution': {
                regime: (count / total_predictions * 100) if total_predictions > 0 else 0
                for regime, count in self.predictions_by_regime.items()
            },
            'regime_transitions': len(set(r['regime'] for r in self.regime_history)),
            'current_regime': self.current_regime
        }

        return summary

    def get_regime_timeline(self) -> pd.DataFrame:
        """Get timeline of regime detections."""
        if not self.regime_history:
            return pd.DataFrame()

        return pd.DataFrame(self.regime_history)

    def reset_tracking(self):
        """Reset performance tracking (for new episode)."""
        self.regime_history = []
        self.predictions_by_regime = {'bear': 0, 'bull': 0, 'neutral': 0}
        self.current_regime = None


class EnsembleTradingEnv:
    """
    Wrapper for TradingEnv that uses ensemble predictions.

    This allows evaluating the ensemble using the same interface as single models.
    """

    def __init__(
        self,
        ensemble: RegimeAdaptiveEnsemble,
        base_env
    ):
        """
        Initialize ensemble environment wrapper.

        Args:
            ensemble: RegimeAdaptiveEnsemble instance
            base_env: Base TradingEnv instance
        """
        self.ensemble = ensemble
        self.env = base_env

    def reset(self, **kwargs):
        """Reset environment and ensemble tracking."""
        self.ensemble.reset_tracking()
        return self.env.reset(**kwargs)

    def step(self, action: int):
        """Execute step in base environment."""
        return self.env.step(action)

    def get_ensemble_action(self, state: np.ndarray) -> Tuple[int, RegimeType]:
        """
        Get action from ensemble.

        Args:
            state: Current state

        Returns:
            (action, regime)
        """
        current_idx = self.env.current_idx
        market_data = self.env.market_data

        action, regime = self.ensemble.predict(
            state=state,
            df=market_data,
            current_idx=current_idx,
            deterministic=True
        )

        return action, regime

    def get_performance_summary(self):
        """Get combined portfolio and ensemble performance."""
        portfolio_stats = self.env.get_portfolio_stats()
        ensemble_stats = self.ensemble.get_performance_summary()

        return {
            **portfolio_stats,
            **ensemble_stats
        }


def print_ensemble_summary(summary: Dict):
    """Print formatted ensemble performance summary."""
    print("\n" + "="*70)
    print("ENSEMBLE PERFORMANCE SUMMARY")
    print("="*70)

    print(f"\nTotal Predictions: {summary['total_predictions']}")
    print("\nRegime Distribution:")
    for regime, pct in summary['regime_distribution'].items():
        print(f"  {regime.upper():8s}: {pct:5.1f}%")

    print(f"\nCurrent Regime: {summary.get('current_regime', 'N/A').upper()}")
    print(f"Regime Transitions: {summary['regime_transitions']}")

    print("\n" + "="*70)


if __name__ == "__main__":
    # Test ensemble
    print("Testing Regime-Adaptive Ensemble...")

    from rl.trading_env import TradingEnv

    # Create environment
    env = TradingEnv(
        symbol="BTC-USD",
        start_date="2024-01-01",
        end_date="2024-06-01",
        initial_capital=100000.0
    )

    # Create ensemble
    ensemble = RegimeAdaptiveEnsemble(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    print(f"\nEnsemble loaded with 3 specialists:")
    print(f"  - BEAR market: phase2_validation/best_model.pth")
    print(f"  - BULL market: phase3_test/best_model.pth")
    print(f"  - NEUTRAL market: phase2_validation/best_model.pth")

    # Wrap environment
    ensemble_env = EnsembleTradingEnv(ensemble, env)

    # Run episode
    state, info = ensemble_env.reset()
    total_reward = 0
    steps = 0

    print("\nRunning test episode...")
    while steps < 50:  # Just test first 50 steps
        # Get ensemble action
        action, regime = ensemble_env.get_ensemble_action(state)

        # Execute action
        next_state, reward, terminated, truncated, info = ensemble_env.step(action)

        total_reward += reward
        steps += 1
        state = next_state

        if steps % 10 == 0:
            print(f"Step {steps}: Regime={regime.upper()}, "
                  f"Action={['HOLD','BUY','SELL'][action]}, "
                  f"Value=${info['portfolio_value']:.2f}")

        if terminated or truncated:
            break

    # Get summary
    summary = ensemble_env.get_performance_summary()
    print_ensemble_summary(summary)

    # Show regime timeline
    timeline = ensemble.get_regime_timeline()
    if not timeline.empty:
        print("\nRegime Timeline (first 10):")
        print(timeline.head(10))

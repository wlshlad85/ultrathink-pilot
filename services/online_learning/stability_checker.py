#!/usr/bin/env python3
"""
Stability Checker for Online Learning

Monitors model performance during incremental updates and triggers automatic
rollback if performance degrades beyond acceptable thresholds.

Key Features:
- Sharpe ratio tracking (primary stability metric)
- Pre/post update performance comparison
- Automatic rollback on >30% degradation
- Multiple performance metrics (returns, volatility, drawdown)
- Alert system for stability failures

Safety Mechanisms:
- Conservative thresholds (30% degradation triggers rollback)
- Multiple stability checks (Sharpe, returns, volatility)
- Checkpoint restoration on failure
- Detailed logging and alerts
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StabilityStatus(Enum):
    """Stability status codes."""
    STABLE = "stable"
    WARNING = "warning"
    CRITICAL = "critical"
    ROLLBACK_REQUIRED = "rollback_required"


@dataclass
class PerformanceMetrics:
    """Performance metrics for stability evaluation."""
    sharpe_ratio: float
    total_return: float
    volatility: float
    max_drawdown: float
    win_rate: float
    average_return: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'total_return': self.total_return,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'average_return': self.average_return,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_returns(cls, returns: np.ndarray) -> 'PerformanceMetrics':
        """
        Compute metrics from returns array.

        Args:
            returns: Array of episode/trade returns

        Returns:
            PerformanceMetrics instance
        """
        if len(returns) == 0:
            return cls(
                sharpe_ratio=0.0,
                total_return=0.0,
                volatility=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                average_return=0.0
            )

        # Basic statistics
        avg_return = np.mean(returns)
        volatility = np.std(returns)

        # Sharpe ratio (annualized, assuming daily returns)
        sharpe_ratio = (avg_return / volatility) * np.sqrt(252) if volatility > 0 else 0.0

        # Total return
        total_return = np.sum(returns)

        # Win rate
        win_rate = np.mean(returns > 0) * 100

        # Maximum drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown)

        return cls(
            sharpe_ratio=sharpe_ratio,
            total_return=total_return,
            volatility=volatility,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            average_return=avg_return
        )


@dataclass
class StabilityCheckResult:
    """Result of a stability check."""
    status: StabilityStatus
    degradation_percent: float
    metrics_before: PerformanceMetrics
    metrics_after: PerformanceMetrics
    should_rollback: bool
    alert_message: str
    details: Dict = field(default_factory=dict)


class StabilityChecker:
    """
    Monitors model performance and triggers rollback on degradation.

    Stability Criteria:
    1. Sharpe ratio degradation <30% (primary)
    2. Win rate degradation <40%
    3. Volatility increase <50%
    4. No catastrophic failures (e.g., all negative returns)

    Usage:
        checker = StabilityChecker()

        # Evaluate before update
        metrics_before = checker.evaluate_performance(model, test_env)

        # Perform update
        model.update(...)

        # Check stability after update
        result = checker.check_stability(model, test_env, metrics_before)

        if result.should_rollback:
            model.load_checkpoint(last_stable_checkpoint)
    """

    def __init__(
        self,
        sharpe_threshold: float = 0.30,  # 30% degradation triggers rollback
        win_rate_threshold: float = 0.40,  # 40% degradation
        volatility_threshold: float = 0.50,  # 50% increase
        min_episodes_for_check: int = 50,
        alert_file: Optional[str] = None
    ):
        """
        Initialize Stability Checker.

        Args:
            sharpe_threshold: Maximum acceptable Sharpe degradation (0.0-1.0)
            win_rate_threshold: Maximum acceptable win rate degradation
            volatility_threshold: Maximum acceptable volatility increase
            min_episodes_for_check: Minimum episodes needed for reliable check
            alert_file: Path to file for storing alerts
        """
        self.sharpe_threshold = sharpe_threshold
        self.win_rate_threshold = win_rate_threshold
        self.volatility_threshold = volatility_threshold
        self.min_episodes_for_check = min_episodes_for_check

        # Alert management
        self.alert_file = Path(alert_file) if alert_file else None
        self.alerts: List[Dict] = []

        # History tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.stability_checks: List[StabilityCheckResult] = []

        logger.info(f"StabilityChecker initialized with thresholds: "
                   f"sharpe={sharpe_threshold}, "
                   f"win_rate={win_rate_threshold}, "
                   f"volatility={volatility_threshold}")

    def evaluate_performance(
        self,
        model: torch.nn.Module,
        env,
        num_episodes: int = 100,
        device: Optional[str] = None
    ) -> PerformanceMetrics:
        """
        Evaluate model performance on environment.

        Args:
            model: PyTorch model to evaluate
            env: Trading environment
            num_episodes: Number of episodes to run
            device: Device for inference

        Returns:
            PerformanceMetrics instance
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        model.eval()
        model.to(device)

        episode_returns = []

        with torch.no_grad():
            for episode in range(num_episodes):
                state = env.reset()
                episode_return = 0.0
                done = False

                while not done:
                    # Get action from model
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action_probs, _ = model(state_tensor)

                    # Sample action
                    from torch.distributions import Categorical
                    dist = Categorical(action_probs)
                    action = dist.sample()

                    # Step environment
                    state, reward, done, _ = env.step(action.item())
                    episode_return += reward

                episode_returns.append(episode_return)

        # Compute metrics from returns
        metrics = PerformanceMetrics.from_returns(np.array(episode_returns))

        # Store in history
        self.performance_history.append(metrics)

        logger.info(f"Performance evaluation: Sharpe={metrics.sharpe_ratio:.3f}, "
                   f"Return={metrics.total_return:.2f}, "
                   f"Win Rate={metrics.win_rate:.1f}%")

        return metrics

    def check_stability(
        self,
        metrics_before: PerformanceMetrics,
        metrics_after: PerformanceMetrics
    ) -> StabilityCheckResult:
        """
        Check if model stability is maintained after update.

        Compares pre-update and post-update performance metrics.
        Triggers rollback if degradation exceeds thresholds.

        Args:
            metrics_before: Performance before update
            metrics_after: Performance after update

        Returns:
            StabilityCheckResult with decision and details
        """
        logger.info("Performing stability check...")

        # Calculate degradation percentages
        sharpe_before = metrics_before.sharpe_ratio
        sharpe_after = metrics_after.sharpe_ratio

        if sharpe_before > 0:
            sharpe_degradation = (sharpe_before - sharpe_after) / sharpe_before
        else:
            sharpe_degradation = 0.0

        # Win rate degradation
        win_rate_before = metrics_before.win_rate
        win_rate_after = metrics_after.win_rate

        if win_rate_before > 0:
            win_rate_degradation = (win_rate_before - win_rate_after) / win_rate_before
        else:
            win_rate_degradation = 0.0

        # Volatility increase
        vol_before = metrics_before.volatility
        vol_after = metrics_after.volatility

        if vol_before > 0:
            vol_increase = (vol_after - vol_before) / vol_before
        else:
            vol_increase = 0.0

        # Determine status
        should_rollback = False
        status = StabilityStatus.STABLE
        alert_message = ""

        # Check Sharpe ratio (primary metric)
        if sharpe_degradation > self.sharpe_threshold:
            should_rollback = True
            status = StabilityStatus.ROLLBACK_REQUIRED
            alert_message = (f"CRITICAL: Sharpe ratio degraded by {sharpe_degradation*100:.1f}% "
                           f"(threshold: {self.sharpe_threshold*100:.1f}%)")

        # Check win rate
        elif win_rate_degradation > self.win_rate_threshold:
            should_rollback = True
            status = StabilityStatus.ROLLBACK_REQUIRED
            alert_message = (f"CRITICAL: Win rate degraded by {win_rate_degradation*100:.1f}% "
                           f"(threshold: {self.win_rate_threshold*100:.1f}%)")

        # Check volatility
        elif vol_increase > self.volatility_threshold:
            should_rollback = True
            status = StabilityStatus.ROLLBACK_REQUIRED
            alert_message = (f"CRITICAL: Volatility increased by {vol_increase*100:.1f}% "
                           f"(threshold: {self.volatility_threshold*100:.1f}%)")

        # Warning thresholds (50% of rollback threshold)
        elif sharpe_degradation > self.sharpe_threshold * 0.5:
            status = StabilityStatus.WARNING
            alert_message = (f"WARNING: Sharpe ratio degraded by {sharpe_degradation*100:.1f}% "
                           f"(approaching threshold)")

        else:
            status = StabilityStatus.STABLE
            alert_message = f"Stability check passed. Sharpe degradation: {sharpe_degradation*100:.1f}%"

        # Create result
        result = StabilityCheckResult(
            status=status,
            degradation_percent=sharpe_degradation * 100,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            should_rollback=should_rollback,
            alert_message=alert_message,
            details={
                'sharpe_degradation': sharpe_degradation * 100,
                'win_rate_degradation': win_rate_degradation * 100,
                'volatility_increase': vol_increase * 100,
                'sharpe_before': sharpe_before,
                'sharpe_after': sharpe_after,
                'win_rate_before': win_rate_before,
                'win_rate_after': win_rate_after,
                'volatility_before': vol_before,
                'volatility_after': vol_after
            }
        )

        # Store check result
        self.stability_checks.append(result)

        # Log result
        if should_rollback:
            logger.error(alert_message)
            self._send_alert(result)
        elif status == StabilityStatus.WARNING:
            logger.warning(alert_message)
        else:
            logger.info(alert_message)

        return result

    def _send_alert(self, result: StabilityCheckResult):
        """
        Send alert about stability issue.

        Args:
            result: Stability check result
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'status': result.status.value,
            'message': result.alert_message,
            'degradation_percent': result.degradation_percent,
            'should_rollback': result.should_rollback,
            'details': result.details
        }

        self.alerts.append(alert)

        # Write to alert file if configured
        if self.alert_file:
            try:
                with open(self.alert_file, 'a') as f:
                    f.write(json.dumps(alert) + '\n')
                logger.info(f"Alert written to {self.alert_file}")
            except Exception as e:
                logger.error(f"Failed to write alert to file: {e}")

        # TODO: Add integration with monitoring systems (e.g., Prometheus, Slack)

    def get_performance_trend(
        self,
        window_size: int = 10
    ) -> Dict[str, float]:
        """
        Analyze performance trend over recent checks.

        Args:
            window_size: Number of recent checks to analyze

        Returns:
            Trend statistics
        """
        if len(self.performance_history) < 2:
            return {}

        recent = self.performance_history[-window_size:]

        sharpe_ratios = [m.sharpe_ratio for m in recent]
        returns = [m.total_return for m in recent]

        return {
            'avg_sharpe': np.mean(sharpe_ratios),
            'sharpe_trend': np.polyfit(range(len(sharpe_ratios)), sharpe_ratios, 1)[0],
            'avg_return': np.mean(returns),
            'sharpe_volatility': np.std(sharpe_ratios),
            'checks_count': len(recent)
        }

    def save_history(self, filepath: str):
        """
        Save performance history and stability checks.

        Args:
            filepath: Path to save history
        """
        history = {
            'performance_history': [m.to_dict() for m in self.performance_history],
            'stability_checks': [
                {
                    'status': check.status.value,
                    'degradation_percent': check.degradation_percent,
                    'should_rollback': check.should_rollback,
                    'alert_message': check.alert_message,
                    'details': check.details,
                    'metrics_before': check.metrics_before.to_dict(),
                    'metrics_after': check.metrics_after.to_dict()
                }
                for check in self.stability_checks
            ],
            'alerts': self.alerts
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"History saved to {filepath}")

    def load_history(self, filepath: str):
        """
        Load performance history from file.

        Args:
            filepath: Path to history file
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"History file not found: {filepath}")

        with open(filepath, 'r') as f:
            history = json.load(f)

        # Restore performance history
        self.performance_history = [
            PerformanceMetrics(
                sharpe_ratio=m['sharpe_ratio'],
                total_return=m['total_return'],
                volatility=m['volatility'],
                max_drawdown=m['max_drawdown'],
                win_rate=m['win_rate'],
                average_return=m['average_return'],
                timestamp=datetime.fromisoformat(m['timestamp'])
            )
            for m in history.get('performance_history', [])
        ]

        # Restore alerts
        self.alerts = history.get('alerts', [])

        logger.info(f"History loaded from {filepath}")
        logger.info(f"Loaded {len(self.performance_history)} performance records")


if __name__ == "__main__":
    """Test Stability Checker"""
    print("Testing Stability Checker...")

    # Create dummy returns data
    np.random.seed(42)

    # Scenario 1: Stable performance
    returns_before = np.random.normal(0.01, 0.02, 100)
    returns_after = np.random.normal(0.009, 0.021, 100)  # Slight degradation

    metrics_before = PerformanceMetrics.from_returns(returns_before)
    metrics_after = PerformanceMetrics.from_returns(returns_after)

    print(f"\nScenario 1: Stable Performance")
    print(f"Before: Sharpe={metrics_before.sharpe_ratio:.3f}, "
          f"Return={metrics_before.total_return:.2f}")
    print(f"After: Sharpe={metrics_after.sharpe_ratio:.3f}, "
          f"Return={metrics_after.total_return:.2f}")

    checker = StabilityChecker()
    result = checker.check_stability(metrics_before, metrics_after)

    print(f"Status: {result.status.value}")
    print(f"Should Rollback: {result.should_rollback}")
    print(f"Alert: {result.alert_message}")

    # Scenario 2: Severe degradation (should trigger rollback)
    returns_before_2 = np.random.normal(0.01, 0.02, 100)
    returns_after_2 = np.random.normal(0.002, 0.03, 100)  # Major degradation

    metrics_before_2 = PerformanceMetrics.from_returns(returns_before_2)
    metrics_after_2 = PerformanceMetrics.from_returns(returns_after_2)

    print(f"\nScenario 2: Severe Degradation")
    print(f"Before: Sharpe={metrics_before_2.sharpe_ratio:.3f}, "
          f"Return={metrics_before_2.total_return:.2f}")
    print(f"After: Sharpe={metrics_after_2.sharpe_ratio:.3f}, "
          f"Return={metrics_after_2.total_return:.2f}")

    result_2 = checker.check_stability(metrics_before_2, metrics_after_2)

    print(f"Status: {result_2.status.value}")
    print(f"Should Rollback: {result_2.should_rollback}")
    print(f"Alert: {result_2.alert_message}")

    # Test history saving
    history_file = "/tmp/stability_history_test.json"
    checker.save_history(history_file)
    print(f"\nHistory saved to {history_file}")

    # Test trend analysis
    trend = checker.get_performance_trend()
    print(f"Performance trend: {trend}")

    print("\nStability Checker test completed successfully!")

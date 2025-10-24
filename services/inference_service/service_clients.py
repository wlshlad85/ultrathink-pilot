"""
Service clients for integrating with other microservices.
Provides both real and mock implementations.
"""
import asyncio
import aiohttp
import numpy as np
from typing import Dict, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class DataServiceClient:
    """Client for Data Service (feature engineering)."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv('DATA_SERVICE_URL', 'http://data-service:8000')
        self.use_mock = os.getenv('MOCK_DATA_SERVICE', 'true').lower() == 'true'

    async def get_features(self, symbol: str, timestamp: str) -> np.ndarray:
        """
        Get features for a symbol at a given timestamp.

        Args:
            symbol: Trading symbol
            timestamp: ISO timestamp

        Returns:
            Feature vector (43-dim numpy array)
        """
        if self.use_mock:
            return self._mock_features()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/v1/features/{symbol}",
                    params={'timestamp': timestamp},
                    timeout=aiohttp.ClientTimeout(total=0.1)  # 100ms timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return np.array(data['features'])
                    else:
                        logger.warning(f"Data service returned {response.status}, using mock")
                        return self._mock_features()
        except Exception as e:
            logger.warning(f"Data service error: {e}, using mock")
            return self._mock_features()

    def _mock_features(self) -> np.ndarray:
        """Generate mock features (43-dim)."""
        # Simulate realistic trading features
        features = np.random.randn(43)
        # Normalize to reasonable range
        features = np.clip(features, -3, 3)
        return features


class RegimeDetectionClient:
    """Client for Regime Detection Service."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv('REGIME_DETECTION_URL', 'http://regime-detection:8001')
        self.use_mock = os.getenv('MOCK_REGIME_DETECTION', 'true').lower() == 'true'

    async def get_regime_probabilities(self, symbol: str) -> Dict[str, float]:
        """
        Get regime probabilities for a symbol.

        Returns:
            Dict with 'bull', 'bear', 'sideways', 'entropy'
        """
        if self.use_mock:
            return self._mock_regime_probs()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/regime/probabilities",
                    params={'symbol': symbol},
                    timeout=aiohttp.ClientTimeout(total=0.05)  # 50ms timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Regime detection returned {response.status}, using mock")
                        return self._mock_regime_probs()
        except Exception as e:
            logger.warning(f"Regime detection error: {e}, using mock")
            return self._mock_regime_probs()

    def _mock_regime_probs(self) -> Dict[str, float]:
        """Generate mock regime probabilities."""
        # Generate random probabilities that sum to 1.0
        probs = np.random.dirichlet([1, 1, 1])
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return {
            'bull': float(probs[0]),
            'bear': float(probs[1]),
            'sideways': float(probs[2]),
            'entropy': float(entropy)
        }


class MetaControllerClient:
    """Client for Meta-Controller Service (strategy blending)."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv('META_CONTROLLER_URL', 'http://meta-controller:8002')
        self.use_mock = os.getenv('MOCK_META_CONTROLLER', 'true').lower() == 'true'

    async def get_strategy_weights(
        self,
        regime_probs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Get strategy blending weights based on regime.

        Args:
            regime_probs: Regime probabilities

        Returns:
            Dict with 'bull_specialist', 'bear_specialist', 'sideways_specialist'
        """
        if self.use_mock:
            return self._mock_strategy_weights(regime_probs)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/strategy/weights",
                    json=regime_probs,
                    timeout=aiohttp.ClientTimeout(total=0.05)  # 50ms timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Meta-controller returned {response.status}, using mock")
                        return self._mock_strategy_weights(regime_probs)
        except Exception as e:
            logger.warning(f"Meta-controller error: {e}, using mock")
            return self._mock_strategy_weights(regime_probs)

    def _mock_strategy_weights(self, regime_probs: Dict[str, float]) -> Dict[str, float]:
        """Generate mock strategy weights based on regime."""
        # Simple heuristic: weight specialists by regime probability
        bull = regime_probs['bull']
        bear = regime_probs['bear']
        sideways = regime_probs['sideways']

        # Add some smoothing
        weights = np.array([bull, bear, sideways]) + 0.1
        weights = weights / weights.sum()

        return {
            'bull_specialist': float(weights[0]),
            'bear_specialist': float(weights[1]),
            'sideways_specialist': float(weights[2])
        }


class RiskManagerClient:
    """Client for Risk Manager Service."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv('RISK_MANAGER_URL', 'http://risk-manager:8003')
        self.use_mock = os.getenv('MOCK_RISK_MANAGER', 'true').lower() == 'true'

    async def check_risk(
        self,
        symbol: str,
        action: str,
        quantity: int
    ) -> Dict:
        """
        Check if a proposed trade passes risk validation.

        Args:
            symbol: Trading symbol
            action: Trading action (BUY/SELL/HOLD)
            quantity: Proposed quantity

        Returns:
            Risk validation result dict
        """
        if self.use_mock:
            return self._mock_risk_check()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/risk/check",
                    json={'symbol': symbol, 'action': action, 'quantity': quantity},
                    timeout=aiohttp.ClientTimeout(total=0.01)  # 10ms timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 403:
                        # Trade rejected
                        data = await response.json()
                        return data
                    else:
                        logger.warning(f"Risk manager returned {response.status}, using mock")
                        return self._mock_risk_check()
        except Exception as e:
            logger.warning(f"Risk manager error: {e}, using mock")
            return self._mock_risk_check()

    def _mock_risk_check(self) -> Dict:
        """Generate mock risk check (always passes)."""
        return {
            'approved': True,
            'warnings': [],
            'checks': {
                'position_limit': 'pass',
                'concentration': 'pass',
                'daily_loss_limit': 'pass'
            }
        }


class ServiceClients:
    """Aggregated service clients."""

    def __init__(self):
        self.data_service = DataServiceClient()
        self.regime_detection = RegimeDetectionClient()
        self.meta_controller = MetaControllerClient()
        self.risk_manager = RiskManagerClient()

    async def get_all_context(
        self,
        symbol: str,
        timestamp: str
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
        """
        Get all context needed for prediction in parallel.

        Returns:
            (features, regime_probs, strategy_weights)
        """
        # Fetch features and regime in parallel
        features_task = self.data_service.get_features(symbol, timestamp)
        regime_task = self.regime_detection.get_regime_probabilities(symbol)

        features, regime_probs = await asyncio.gather(
            features_task,
            regime_task
        )

        # Get strategy weights based on regime
        strategy_weights = await self.meta_controller.get_strategy_weights(regime_probs)

        return features, regime_probs, strategy_weights

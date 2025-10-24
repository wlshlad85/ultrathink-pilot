"""
Probabilistic Regime Detection Service
Agent: regime-detection-specialist
Mission: Eliminate 15% portfolio disruption through continuous probability distributions

Implements Dirichlet Process Gaussian Mixture Model (DPGMM) for market regime classification.
Outputs continuous probability distributions for bull/bear/sideways regimes instead of discrete labels.

Key Innovation: Smooth regime transitions eliminate portfolio discontinuities during ambiguous markets.
Target: <5% position disruption (vs 15% baseline with hard classification)
"""

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import entropy as scipy_entropy
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime
import pickle
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeType(str, Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


@dataclass
class RegimeProbabilities:
    """
    Regime probability distribution with uncertainty quantification

    Attributes:
        prob_bull: Probability of bullish regime [0, 1]
        prob_bear: Probability of bearish regime [0, 1]
        prob_sideways: Probability of sideways regime [0, 1]
        entropy: Shannon entropy measuring regime uncertainty [0, log(3)]
        timestamp: Detection timestamp
        dominant_regime: Regime with highest probability
        confidence: Probability of dominant regime
    """
    prob_bull: float
    prob_bear: float
    prob_sideways: float
    entropy: float
    timestamp: datetime
    dominant_regime: str
    confidence: float

    def __post_init__(self):
        """Validate probability distribution"""
        prob_sum = self.prob_bull + self.prob_bear + self.prob_sideways
        if abs(prob_sum - 1.0) > 0.001:
            raise ValueError(f"Probabilities must sum to 1.0 (got {prob_sum:.6f})")

        if not (0 <= self.prob_bull <= 1 and 0 <= self.prob_bear <= 1 and 0 <= self.prob_sideways <= 1):
            raise ValueError("All probabilities must be in [0, 1]")

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            "prob_bull": float(self.prob_bull),
            "prob_bear": float(self.prob_bear),
            "prob_sideways": float(self.prob_sideways),
            "entropy": float(self.entropy),
            "timestamp": self.timestamp.isoformat(),
            "dominant_regime": self.dominant_regime,
            "confidence": float(self.confidence)
        }


class ProbabilisticRegimeDetector:
    """
    Dirichlet Process GMM for probabilistic regime classification

    Architecture:
    - Uses Bayesian GMM with Dirichlet Process prior
    - Automatically discovers number of market states (up to max_components)
    - Maps discovered states to bull/bear/sideways via feature centroids
    - Outputs continuous probability distributions (NOT discrete labels)

    Feature Engineering:
    - returns_5d: 5-day cumulative returns (directional signal)
    - volatility_20d: 20-day rolling volatility (risk measure)
    - trend_strength: Linear regression slope over 10 days
    - volume_ratio: Current volume / 20-day average

    Key Hyperparameters:
    - n_components=5: Allow model to discover 3-5 distinct states
    - weight_concentration_prior=0.1: Moderate flexibility in state discovery
    - covariance_type='full': Capture feature correlations
    """

    def __init__(
        self,
        n_components: int = 5,
        weight_concentration_prior: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize probabilistic regime detector

        Args:
            n_components: Maximum number of mixture components (states)
            weight_concentration_prior: Dirichlet concentration (lower = more flexible)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.weight_concentration_prior = weight_concentration_prior
        self.random_state = random_state

        # Initialize Bayesian GMM with Dirichlet Process prior
        self.model = BayesianGaussianMixture(
            n_components=n_components,
            covariance_type='full',  # Capture feature correlations
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=weight_concentration_prior,
            max_iter=300,
            n_init=10,  # Multiple initializations for robustness
            random_state=random_state,
            warm_start=True  # Enable incremental learning
        )

        self.is_fitted = False
        self.feature_buffer = []
        self.buffer_size = 2000  # Rolling window for online learning

        # Regime mapping: cluster_id -> regime_type (learned from data)
        self.regime_mapping = {}

        logger.info(f"Initialized DPGMM with {n_components} components, "
                   f"concentration prior {weight_concentration_prior}")

    def extract_features(self, market_data: Dict) -> np.ndarray:
        """
        Extract feature vector for regime classification

        Features:
        1. returns_5d: 5-day cumulative returns (trend direction)
        2. volatility_20d: 20-day rolling volatility (risk level)
        3. trend_strength: 10-day linear regression slope (trend persistence)
        4. volume_ratio: Volume / 20-day average (momentum indicator)

        Args:
            market_data: Dictionary with market indicators

        Returns:
            Feature vector [returns_5d, volatility_20d, trend_strength, volume_ratio]
        """
        try:
            # Extract features with safe defaults
            returns_5d = market_data.get('returns_5d', 0.0)
            volatility_20d = market_data.get('volatility_20d', 0.02)
            trend_strength = market_data.get('trend_strength', 0.0)
            volume_ratio = market_data.get('volume_ratio', 1.0)

            # Clip outliers for stability (3 sigma rule)
            returns_5d = np.clip(returns_5d, -0.15, 0.15)
            volatility_20d = np.clip(volatility_20d, 0.001, 0.10)
            trend_strength = np.clip(trend_strength, -1.0, 1.0)
            volume_ratio = np.clip(volume_ratio, 0.1, 5.0)

            features = np.array([returns_5d, volatility_20d, trend_strength, volume_ratio])

            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.warning(f"Invalid features detected: {features}, using defaults")
                return np.array([0.0, 0.02, 0.0, 1.0])

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.array([0.0, 0.02, 0.0, 1.0])

    def fit(self, market_data_history: list) -> None:
        """
        Fit DPGMM model on historical market data

        Args:
            market_data_history: List of market data dictionaries
        """
        logger.info(f"Fitting DPGMM on {len(market_data_history)} samples")

        # Extract features
        features = np.array([self.extract_features(data) for data in market_data_history])

        # Fit model
        self.model.fit(features)
        self.is_fitted = True

        # Learn regime mapping from cluster centroids
        self._learn_regime_mapping()

        # Initialize buffer with recent data
        self.feature_buffer = features[-self.buffer_size:].tolist()

        logger.info(f"Model fitted. Active components: {self._count_active_components()}")
        logger.info(f"Regime mapping: {self.regime_mapping}")

    def _learn_regime_mapping(self) -> None:
        """
        Map discovered clusters to bull/bear/sideways regimes

        Strategy:
        - Bull: Positive returns + high trend strength
        - Bear: Negative returns + high trend strength
        - Sideways: Low trend strength OR high volatility with mixed returns
        """
        if not self.is_fitted:
            return

        centroids = self.model.means_
        self.regime_mapping = {}

        for cluster_id, centroid in enumerate(centroids):
            returns_5d, volatility_20d, trend_strength, volume_ratio = centroid

            # Classification rules based on centroid characteristics
            if abs(trend_strength) < 0.3:
                # Low trend strength = sideways
                regime = RegimeType.SIDEWAYS
            elif returns_5d > 0.02 and trend_strength > 0.2:
                # Positive returns + uptrend = bull
                regime = RegimeType.BULL
            elif returns_5d < -0.02 and trend_strength < -0.2:
                # Negative returns + downtrend = bear
                regime = RegimeType.BEAR
            elif volatility_20d > 0.04:
                # High volatility = sideways (choppy market)
                regime = RegimeType.SIDEWAYS
            else:
                # Default to sideways for ambiguous cases
                regime = RegimeType.SIDEWAYS

            self.regime_mapping[cluster_id] = regime.value

    def _count_active_components(self) -> int:
        """Count components with significant weight (>1%)"""
        if not self.is_fitted:
            return 0
        return np.sum(self.model.weights_ > 0.01)

    def update_online(self, features: np.ndarray) -> None:
        """
        Online learning: incrementally update model with new data

        Args:
            features: New feature vector
        """
        self.feature_buffer.append(features)

        # Maintain sliding window
        if len(self.feature_buffer) > self.buffer_size:
            self.feature_buffer.pop(0)

        # Refit every 50 samples for efficiency
        if len(self.feature_buffer) >= 100 and len(self.feature_buffer) % 50 == 0:
            logger.info(f"Online update: refitting with {len(self.feature_buffer)} samples")
            X = np.array(self.feature_buffer)
            self.model.fit(X)
            self._learn_regime_mapping()

    def predict_probabilities(self, market_data: Dict) -> RegimeProbabilities:
        """
        Predict regime probability distribution for current market state

        Returns continuous probability distribution over [bull, bear, sideways].
        This eliminates hard switches and enables weighted ensemble decisions.

        Args:
            market_data: Current market indicators

        Returns:
            RegimeProbabilities with full probability distribution
        """
        if not self.is_fitted:
            return self._bootstrap_probabilities(market_data)

        try:
            # Extract features and predict cluster probabilities
            features = self.extract_features(market_data)
            features_2d = features.reshape(1, -1)

            # Get cluster probability distribution
            cluster_probs = self.model.predict_proba(features_2d)[0]

            # Aggregate cluster probabilities into regime probabilities
            regime_probs = self._aggregate_regime_probabilities(cluster_probs)

            # Calculate Shannon entropy (uncertainty measure)
            regime_entropy = self._calculate_entropy(regime_probs)

            # Determine dominant regime
            dominant_regime = max(regime_probs.items(), key=lambda x: x[1])[0]
            confidence = regime_probs[dominant_regime]

            result = RegimeProbabilities(
                prob_bull=regime_probs[RegimeType.BULL.value],
                prob_bear=regime_probs[RegimeType.BEAR.value],
                prob_sideways=regime_probs[RegimeType.SIDEWAYS.value],
                entropy=regime_entropy,
                timestamp=datetime.utcnow(),
                dominant_regime=dominant_regime,
                confidence=confidence
            )

            # Online learning update
            self.update_online(features)

            return result

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return self._bootstrap_probabilities(market_data)

    def _aggregate_regime_probabilities(self, cluster_probs: np.ndarray) -> Dict[str, float]:
        """
        Aggregate cluster probabilities into regime probabilities

        Multiple clusters can map to the same regime, so we sum their probabilities.

        Args:
            cluster_probs: Probability distribution over clusters

        Returns:
            Dictionary mapping regime -> aggregated probability
        """
        regime_probs = {
            RegimeType.BULL.value: 0.0,
            RegimeType.BEAR.value: 0.0,
            RegimeType.SIDEWAYS.value: 0.0
        }

        for cluster_id, prob in enumerate(cluster_probs):
            regime = self.regime_mapping.get(cluster_id, RegimeType.SIDEWAYS.value)
            regime_probs[regime] += prob

        # Normalize to ensure sum=1.0 (handle floating point errors)
        total = sum(regime_probs.values())
        if total > 0:
            regime_probs = {k: v / total for k, v in regime_probs.items()}
        else:
            # Fallback: uniform distribution
            regime_probs = {k: 1.0 / 3.0 for k in regime_probs.keys()}

        return regime_probs

    def _calculate_entropy(self, probs: Dict[str, float]) -> float:
        """
        Calculate Shannon entropy of probability distribution

        Entropy quantifies regime uncertainty:
        - Low entropy (near 0): High confidence in single regime
        - High entropy (near log(3)≈1.1): Maximum uncertainty, mixed regime

        Args:
            probs: Probability distribution

        Returns:
            Shannon entropy in nats
        """
        prob_array = np.array(list(probs.values()))
        # Filter out zero probabilities to avoid log(0)
        prob_array = prob_array[prob_array > 1e-10]
        return float(scipy_entropy(prob_array))

    def _bootstrap_probabilities(self, market_data: Dict) -> RegimeProbabilities:
        """
        Bootstrap regime classification using heuristics (cold start)

        Used when model is not yet fitted. Provides reasonable default
        based on simple rules.

        Args:
            market_data: Market indicators

        Returns:
            RegimeProbabilities with rule-based distribution
        """
        features = self.extract_features(market_data)
        returns_5d, volatility_20d, trend_strength, volume_ratio = features

        # Initialize with uniform distribution
        prob_bull = 0.33
        prob_bear = 0.33
        prob_sideways = 0.34

        # Adjust based on returns and trend
        if returns_5d > 0.03 and trend_strength > 0.3:
            prob_bull = 0.60
            prob_bear = 0.10
            prob_sideways = 0.30
        elif returns_5d < -0.03 and trend_strength < -0.3:
            prob_bull = 0.10
            prob_bear = 0.60
            prob_sideways = 0.30
        elif abs(trend_strength) < 0.2 or volatility_20d > 0.04:
            prob_bull = 0.25
            prob_bear = 0.25
            prob_sideways = 0.50

        # Normalize to ensure sum=1.0
        total = prob_bull + prob_bear + prob_sideways
        prob_bull /= total
        prob_bear /= total
        prob_sideways /= total

        regime_probs = {
            RegimeType.BULL.value: prob_bull,
            RegimeType.BEAR.value: prob_bear,
            RegimeType.SIDEWAYS.value: prob_sideways
        }

        entropy_val = self._calculate_entropy(regime_probs)
        dominant = max(regime_probs.items(), key=lambda x: x[1])[0]

        return RegimeProbabilities(
            prob_bull=prob_bull,
            prob_bear=prob_bear,
            prob_sideways=prob_sideways,
            entropy=entropy_val,
            timestamp=datetime.utcnow(),
            dominant_regime=dominant,
            confidence=regime_probs[dominant]
        )

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_state = {
            'model': self.model,
            'regime_mapping': self.regime_mapping,
            'feature_buffer': self.feature_buffer[-100:],  # Save recent history
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)

        self.model = model_state['model']
        self.regime_mapping = model_state['regime_mapping']
        self.feature_buffer = model_state['feature_buffer']
        self.is_fitted = model_state['is_fitted']

        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Regime mapping: {self.regime_mapping}")


def demo_detector():
    """Demonstration of probabilistic regime detector"""
    detector = ProbabilisticRegimeDetector()

    # Generate synthetic market data for demonstration
    np.random.seed(42)
    bull_market = [
        {
            'returns_5d': np.random.uniform(0.02, 0.08),
            'volatility_20d': np.random.uniform(0.01, 0.03),
            'trend_strength': np.random.uniform(0.4, 0.8),
            'volume_ratio': np.random.uniform(1.0, 2.0)
        }
        for _ in range(200)
    ]

    bear_market = [
        {
            'returns_5d': np.random.uniform(-0.08, -0.02),
            'volatility_20d': np.random.uniform(0.02, 0.05),
            'trend_strength': np.random.uniform(-0.8, -0.4),
            'volume_ratio': np.random.uniform(1.2, 2.5)
        }
        for _ in range(200)
    ]

    sideways_market = [
        {
            'returns_5d': np.random.uniform(-0.02, 0.02),
            'volatility_20d': np.random.uniform(0.015, 0.025),
            'trend_strength': np.random.uniform(-0.2, 0.2),
            'volume_ratio': np.random.uniform(0.8, 1.2)
        }
        for _ in range(200)
    ]

    all_data = bull_market + bear_market + sideways_market
    np.random.shuffle(all_data)

    # Fit model
    detector.fit(all_data)

    # Test predictions
    print("\n=== Probabilistic Regime Detection Demo ===\n")

    # Bull market test
    bull_test = {'returns_5d': 0.05, 'volatility_20d': 0.02, 'trend_strength': 0.6, 'volume_ratio': 1.5}
    bull_probs = detector.predict_probabilities(bull_test)
    print(f"Bull Market Test:")
    print(f"  Probabilities: Bull={bull_probs.prob_bull:.3f}, "
          f"Bear={bull_probs.prob_bear:.3f}, Sideways={bull_probs.prob_sideways:.3f}")
    print(f"  Entropy: {bull_probs.entropy:.3f} (uncertainty)")
    print(f"  Dominant: {bull_probs.dominant_regime} ({bull_probs.confidence:.1%} confidence)\n")

    # Bear market test
    bear_test = {'returns_5d': -0.05, 'volatility_20d': 0.03, 'trend_strength': -0.6, 'volume_ratio': 2.0}
    bear_probs = detector.predict_probabilities(bear_test)
    print(f"Bear Market Test:")
    print(f"  Probabilities: Bull={bear_probs.prob_bull:.3f}, "
          f"Bear={bear_probs.prob_bear:.3f}, Sideways={bear_probs.prob_sideways:.3f}")
    print(f"  Entropy: {bear_probs.entropy:.3f}\n")

    # Sideways market test
    sideways_test = {'returns_5d': 0.0, 'volatility_20d': 0.02, 'trend_strength': 0.0, 'volume_ratio': 1.0}
    sideways_probs = detector.predict_probabilities(sideways_test)
    print(f"Sideways Market Test:")
    print(f"  Probabilities: Bull={sideways_probs.prob_bull:.3f}, "
          f"Bear={sideways_probs.prob_bear:.3f}, Sideways={sideways_probs.prob_sideways:.3f}")
    print(f"  Entropy: {sideways_probs.entropy:.3f}\n")

    # Mixed regime test (ambiguous case)
    mixed_test = {'returns_5d': 0.01, 'volatility_20d': 0.035, 'trend_strength': 0.15, 'volume_ratio': 1.8}
    mixed_probs = detector.predict_probabilities(mixed_test)
    print(f"Mixed/Uncertain Market Test:")
    print(f"  Probabilities: Bull={mixed_probs.prob_bull:.3f}, "
          f"Bear={mixed_probs.prob_bear:.3f}, Sideways={mixed_probs.prob_sideways:.3f}")
    print(f"  Entropy: {mixed_probs.entropy:.3f} (HIGH = uncertain)")
    print(f"  This is where probabilistic approach shines - no hard switch!\n")


if __name__ == '__main__':
    demo_detector()

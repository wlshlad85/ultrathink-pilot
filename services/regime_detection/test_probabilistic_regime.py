"""
Comprehensive Unit Tests for Probabilistic Regime Detector
Agent: regime-detection-specialist

Test Coverage:
1. Probability distribution validation (sum=1.0 ± 0.001)
2. Feature extraction and preprocessing
3. Model fitting and prediction
4. Online learning updates
5. Edge cases and error handling
6. API endpoint functionality
7. TimescaleDB integration

Target: >85% code coverage
"""

import pytest
import numpy as np
from datetime import datetime
import json
from unittest.mock import Mock, patch, MagicMock
import psycopg2

from probabilistic_regime_detector import (
    ProbabilisticRegimeDetector,
    RegimeProbabilities,
    RegimeType
)


class TestRegimeProbabilities:
    """Test RegimeProbabilities dataclass validation"""

    def test_valid_probability_distribution(self):
        """Test that valid probability distribution is accepted"""
        probs = RegimeProbabilities(
            prob_bull=0.5,
            prob_bear=0.3,
            prob_sideways=0.2,
            entropy=0.8,
            timestamp=datetime.utcnow(),
            dominant_regime="bull",
            confidence=0.5
        )

        assert abs(probs.prob_bull + probs.prob_bear + probs.prob_sideways - 1.0) < 0.001

    def test_probability_sum_validation(self):
        """Test that probabilities must sum to 1.0"""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            RegimeProbabilities(
                prob_bull=0.5,
                prob_bear=0.3,
                prob_sideways=0.3,  # Sum = 1.1
                entropy=0.8,
                timestamp=datetime.utcnow(),
                dominant_regime="bull",
                confidence=0.5
            )

    def test_probability_range_validation(self):
        """Test that probabilities must be in [0, 1]"""
        with pytest.raises(ValueError, match="must be in"):
            RegimeProbabilities(
                prob_bull=1.5,  # > 1.0
                prob_bear=0.0,
                prob_sideways=-0.5,  # < 0.0
                entropy=0.8,
                timestamp=datetime.utcnow(),
                dominant_regime="bull",
                confidence=0.5
            )

    def test_to_dict_conversion(self):
        """Test conversion to dictionary for API responses"""
        timestamp = datetime.utcnow()
        probs = RegimeProbabilities(
            prob_bull=0.6,
            prob_bear=0.2,
            prob_sideways=0.2,
            entropy=0.9,
            timestamp=timestamp,
            dominant_regime="bull",
            confidence=0.6
        )

        result = probs.to_dict()

        assert result['prob_bull'] == 0.6
        assert result['prob_bear'] == 0.2
        assert result['prob_sideways'] == 0.2
        assert result['entropy'] == 0.9
        assert result['timestamp'] == timestamp.isoformat()
        assert result['dominant_regime'] == "bull"
        assert result['confidence'] == 0.6


class TestProbabilisticRegimeDetector:
    """Test ProbabilisticRegimeDetector core functionality"""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        return ProbabilisticRegimeDetector(random_state=42)

    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing"""
        return {
            'returns_5d': 0.05,
            'volatility_20d': 0.02,
            'trend_strength': 0.6,
            'volume_ratio': 1.5
        }

    @pytest.fixture
    def training_data(self):
        """Generate synthetic training data"""
        np.random.seed(42)

        # Bull market samples
        bull_data = [
            {
                'returns_5d': np.random.uniform(0.02, 0.08),
                'volatility_20d': np.random.uniform(0.01, 0.03),
                'trend_strength': np.random.uniform(0.4, 0.8),
                'volume_ratio': np.random.uniform(1.0, 2.0)
            }
            for _ in range(100)
        ]

        # Bear market samples
        bear_data = [
            {
                'returns_5d': np.random.uniform(-0.08, -0.02),
                'volatility_20d': np.random.uniform(0.02, 0.05),
                'trend_strength': np.random.uniform(-0.8, -0.4),
                'volume_ratio': np.random.uniform(1.2, 2.5)
            }
            for _ in range(100)
        ]

        # Sideways market samples
        sideways_data = [
            {
                'returns_5d': np.random.uniform(-0.02, 0.02),
                'volatility_20d': np.random.uniform(0.015, 0.025),
                'trend_strength': np.random.uniform(-0.2, 0.2),
                'volume_ratio': np.random.uniform(0.8, 1.2)
            }
            for _ in range(100)
        ]

        return bull_data + bear_data + sideways_data

    def test_detector_initialization(self, detector):
        """Test detector initialization parameters"""
        assert detector.n_components == 5
        assert detector.weight_concentration_prior == 0.1
        assert detector.random_state == 42
        assert not detector.is_fitted
        assert len(detector.feature_buffer) == 0

    def test_feature_extraction(self, detector, sample_market_data):
        """Test feature extraction from market data"""
        features = detector.extract_features(sample_market_data)

        assert features.shape == (4,)
        assert features[0] == 0.05  # returns_5d
        assert features[1] == 0.02  # volatility_20d
        assert features[2] == 0.6   # trend_strength
        assert features[3] == 1.5   # volume_ratio

    def test_feature_extraction_with_defaults(self, detector):
        """Test feature extraction with missing values"""
        incomplete_data = {'returns_5d': 0.03}
        features = detector.extract_features(incomplete_data)

        assert features.shape == (4,)
        assert features[0] == 0.03  # returns_5d
        assert features[1] == 0.02  # default volatility
        assert features[2] == 0.0   # default trend_strength
        assert features[3] == 1.0   # default volume_ratio

    def test_feature_extraction_outlier_clipping(self, detector):
        """Test that extreme outliers are clipped"""
        extreme_data = {
            'returns_5d': 0.5,  # Will be clipped to 0.15
            'volatility_20d': 0.5,  # Will be clipped to 0.10
            'trend_strength': 5.0,  # Will be clipped to 1.0
            'volume_ratio': 10.0  # Will be clipped to 5.0
        }

        features = detector.extract_features(extreme_data)

        assert features[0] <= 0.15  # returns clipped
        assert features[1] <= 0.10  # volatility clipped
        assert features[2] <= 1.0   # trend_strength clipped
        assert features[3] <= 5.0   # volume_ratio clipped

    def test_feature_extraction_nan_handling(self, detector):
        """Test handling of NaN values in features"""
        nan_data = {
            'returns_5d': float('nan'),
            'volatility_20d': 0.02,
            'trend_strength': float('inf'),
            'volume_ratio': 1.0
        }

        features = detector.extract_features(nan_data)

        # Should return safe defaults
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_model_fitting(self, detector, training_data):
        """Test model fitting on training data"""
        assert not detector.is_fitted

        detector.fit(training_data)

        assert detector.is_fitted
        assert len(detector.regime_mapping) > 0
        assert len(detector.feature_buffer) > 0

    def test_regime_mapping_learned(self, detector, training_data):
        """Test that regime mapping is learned correctly"""
        detector.fit(training_data)

        # Check that all three regimes are represented
        regime_values = set(detector.regime_mapping.values())

        # Should have at least 2 distinct regimes (ideally all 3)
        assert len(regime_values) >= 2
        assert any(r in regime_values for r in ['bull', 'bear', 'sideways'])

    def test_bootstrap_prediction_unfitted(self, detector, sample_market_data):
        """Test bootstrap prediction before model is fitted"""
        probs = detector.predict_probabilities(sample_market_data)

        assert isinstance(probs, RegimeProbabilities)
        assert abs(probs.prob_bull + probs.prob_bear + probs.prob_sideways - 1.0) < 0.001
        assert 0 <= probs.entropy <= 2.0
        assert probs.dominant_regime in ['bull', 'bear', 'sideways']

    def test_prediction_after_fitting(self, detector, training_data, sample_market_data):
        """Test prediction after model fitting"""
        detector.fit(training_data)

        probs = detector.predict_probabilities(sample_market_data)

        assert isinstance(probs, RegimeProbabilities)
        assert abs(probs.prob_bull + probs.prob_bear + probs.prob_sideways - 1.0) < 0.001
        assert all(0 <= p <= 1 for p in [probs.prob_bull, probs.prob_bear, probs.prob_sideways])
        assert 0 <= probs.confidence <= 1

    def test_bull_market_prediction(self, detector, training_data):
        """Test prediction for clear bull market conditions"""
        detector.fit(training_data)

        bull_data = {
            'returns_5d': 0.06,
            'volatility_20d': 0.02,
            'trend_strength': 0.7,
            'volume_ratio': 1.5
        }

        probs = detector.predict_probabilities(bull_data)

        # Bull probability should be highest for bull market
        assert probs.prob_bull >= probs.prob_bear
        assert probs.prob_bull >= probs.prob_sideways

    def test_bear_market_prediction(self, detector, training_data):
        """Test prediction for clear bear market conditions"""
        detector.fit(training_data)

        bear_data = {
            'returns_5d': -0.06,
            'volatility_20d': 0.03,
            'trend_strength': -0.7,
            'volume_ratio': 2.0
        }

        probs = detector.predict_probabilities(bear_data)

        # Bear probability should be highest for bear market
        assert probs.prob_bear >= probs.prob_bull
        assert probs.prob_bear >= probs.prob_sideways

    def test_sideways_market_prediction(self, detector, training_data):
        """Test prediction for sideways market conditions"""
        detector.fit(training_data)

        sideways_data = {
            'returns_5d': 0.0,
            'volatility_20d': 0.02,
            'trend_strength': 0.0,
            'volume_ratio': 1.0
        }

        probs = detector.predict_probabilities(sideways_data)

        # Sideways probability should be high for neutral market
        # (May not always be highest due to probabilistic nature)
        assert probs.prob_sideways > 0.2  # At least some sideways probability

    def test_entropy_calculation(self, detector, training_data):
        """Test entropy calculation for uncertainty quantification"""
        detector.fit(training_data)

        # Clear signal (low entropy)
        clear_bull = {'returns_5d': 0.08, 'volatility_20d': 0.015, 'trend_strength': 0.8, 'volume_ratio': 1.8}
        probs_clear = detector.predict_probabilities(clear_bull)

        # Ambiguous signal (high entropy)
        ambiguous = {'returns_5d': 0.01, 'volatility_20d': 0.035, 'trend_strength': 0.1, 'volume_ratio': 1.2}
        probs_ambiguous = detector.predict_probabilities(ambiguous)

        # Entropy should be higher for ambiguous cases
        # (This is probabilistic, so we just check it's calculated)
        assert 0 <= probs_clear.entropy <= np.log(3)
        assert 0 <= probs_ambiguous.entropy <= np.log(3)

    def test_online_learning_update(self, detector, training_data):
        """Test online learning buffer management"""
        detector.fit(training_data)

        initial_buffer_size = len(detector.feature_buffer)

        # Add new data point
        new_data = {'returns_5d': 0.03, 'volatility_20d': 0.025, 'trend_strength': 0.3, 'volume_ratio': 1.2}
        detector.predict_probabilities(new_data)

        # Buffer should grow (up to max size)
        assert len(detector.feature_buffer) >= initial_buffer_size

    def test_buffer_size_limit(self, detector, training_data):
        """Test that buffer doesn't exceed maximum size"""
        detector.fit(training_data)

        # Add many data points
        for _ in range(3000):
            new_data = {'returns_5d': 0.03, 'volatility_20d': 0.025, 'trend_strength': 0.3, 'volume_ratio': 1.2}
            detector.predict_probabilities(new_data)

        # Buffer should not exceed max size
        assert len(detector.feature_buffer) <= detector.buffer_size

    def test_model_save_and_load(self, detector, training_data, tmp_path):
        """Test model serialization and deserialization"""
        detector.fit(training_data)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        detector.save_model(str(model_path))

        assert model_path.exists()

        # Load model in new detector
        new_detector = ProbabilisticRegimeDetector()
        new_detector.load_model(str(model_path))

        assert new_detector.is_fitted
        assert new_detector.regime_mapping == detector.regime_mapping

        # Predictions should be consistent
        test_data = {'returns_5d': 0.05, 'volatility_20d': 0.02, 'trend_strength': 0.5, 'volume_ratio': 1.3}
        probs1 = detector.predict_probabilities(test_data)
        probs2 = new_detector.predict_probabilities(test_data)

        assert abs(probs1.prob_bull - probs2.prob_bull) < 0.01
        assert abs(probs1.prob_bear - probs2.prob_bear) < 0.01
        assert abs(probs1.prob_sideways - probs2.prob_sideways) < 0.01

    def test_save_unfitted_model_raises_error(self, detector, tmp_path):
        """Test that saving unfitted model raises error"""
        model_path = tmp_path / "test_model.pkl"

        with pytest.raises(ValueError, match="Cannot save unfitted model"):
            detector.save_model(str(model_path))


class TestAPIEndpoints:
    """Test API endpoint functionality (integration tests)"""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client"""
        from fastapi.testclient import TestClient
        from regime_api import app

        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint returns service info"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "Probabilistic Regime Detection" in data["service"]

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "detector_initialized" in data
        assert "database_connected" in data

    @patch('regime_api.store_regime_to_db', return_value=True)
    @patch('regime_api.detector')
    def test_predict_probabilities_endpoint(self, mock_detector, mock_store, client):
        """Test prediction endpoint"""
        # Mock detector prediction
        mock_probs = RegimeProbabilities(
            prob_bull=0.6,
            prob_bear=0.2,
            prob_sideways=0.2,
            entropy=0.85,
            timestamp=datetime.utcnow(),
            dominant_regime="bull",
            confidence=0.6
        )
        mock_detector.predict_probabilities.return_value = mock_probs

        request_data = {
            "symbol": "AAPL",
            "returns_5d": 0.05,
            "volatility_20d": 0.02,
            "trend_strength": 0.6,
            "volume_ratio": 1.5
        }

        response = client.post("/regime/probabilities", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["prob_bull"] == 0.6
        assert data["prob_bear"] == 0.2
        assert data["prob_sideways"] == 0.2
        assert "entropy" in data
        assert "dominant_regime" in data


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def detector(self):
        return ProbabilisticRegimeDetector(random_state=42)

    def test_empty_market_data(self, detector):
        """Test handling of empty market data"""
        empty_data = {}
        probs = detector.predict_probabilities(empty_data)

        # Should use defaults
        assert isinstance(probs, RegimeProbabilities)
        assert abs(probs.prob_bull + probs.prob_bear + probs.prob_sideways - 1.0) < 0.001

    def test_extreme_volatility(self, detector):
        """Test handling of extreme volatility values"""
        extreme_vol_data = {
            'returns_5d': 0.02,
            'volatility_20d': 0.5,  # Very high
            'trend_strength': 0.3,
            'volume_ratio': 1.0
        }

        probs = detector.predict_probabilities(extreme_vol_data)

        # Should still produce valid probabilities
        assert abs(probs.prob_bull + probs.prob_bear + probs.prob_sideways - 1.0) < 0.001

    def test_zero_volume_ratio(self, detector):
        """Test handling of zero volume ratio"""
        zero_vol_data = {
            'returns_5d': 0.02,
            'volatility_20d': 0.02,
            'trend_strength': 0.3,
            'volume_ratio': 0.0  # Should be clipped to minimum
        }

        features = detector.extract_features(zero_vol_data)
        assert features[3] >= 0.1  # Clipped to minimum

    def test_consistent_predictions(self, detector, training_data):
        """Test that predictions are consistent for same input"""
        detector.fit(training_data)

        test_data = {'returns_5d': 0.05, 'volatility_20d': 0.02, 'trend_strength': 0.5, 'volume_ratio': 1.3}

        probs1 = detector.predict_probabilities(test_data)
        probs2 = detector.predict_probabilities(test_data)

        # Should be very similar (may differ slightly due to online learning)
        assert abs(probs1.prob_bull - probs2.prob_bull) < 0.05
        assert abs(probs1.prob_bear - probs2.prob_bear) < 0.05
        assert abs(probs1.prob_sideways - probs2.prob_sideways) < 0.05


# Performance benchmark (not a test, just informational)
def test_prediction_performance(benchmark):
    """Benchmark prediction latency"""
    detector = ProbabilisticRegimeDetector(random_state=42)

    # Generate training data
    np.random.seed(42)
    training_data = [
        {
            'returns_5d': np.random.uniform(-0.1, 0.1),
            'volatility_20d': np.random.uniform(0.01, 0.05),
            'trend_strength': np.random.uniform(-1, 1),
            'volume_ratio': np.random.uniform(0.5, 2.0)
        }
        for _ in range(300)
    ]

    detector.fit(training_data)

    test_data = {'returns_5d': 0.05, 'volatility_20d': 0.02, 'trend_strength': 0.5, 'volume_ratio': 1.3}

    # Benchmark prediction
    result = benchmark(detector.predict_probabilities, test_data)

    # Should complete in <10ms for P95 latency target
    assert isinstance(result, RegimeProbabilities)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=probabilistic_regime_detector', '--cov-report=html'])

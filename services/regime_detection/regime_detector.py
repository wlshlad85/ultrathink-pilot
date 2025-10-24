"""
Regime Detection Service - DPGMM Implementation
Agent: regime-detection-specialist

Classifies market regimes in real-time using Dirichlet Process Gaussian Mixture Model.
Publishes regime changes to Kafka for downstream strategy selection.
"""
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from kafka import KafkaProducer, KafkaConsumer
import json
import logging
from datetime import datetime
import redis
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Market regime detection using DPGMM with 4 regime types:
    - Trending (strong directional movement)
    - Mean-Reverting (oscillating around mean)
    - Volatile (high variance, unpredictable)
    - Stable (low variance, range-bound)
    """

    def __init__(self, kafka_bootstrap='kafka-1:9092', redis_host='redis', redis_port=6379):
        self.kafka_bootstrap = kafka_bootstrap
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=1)

        # Initialize DPGMM with Dirichlet Process prior
        self.model = BayesianGaussianMixture(
            n_components=4,  # Maximum 4 regimes
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.01,  # Low prior = flexible regime discovery
            max_iter=200,
            n_init=5,
            random_state=42
        )

        self.regime_labels = {
            0: 'trending',
            1: 'mean_reverting',
            2: 'volatile',
            3: 'stable'
        }

        self.is_fitted = False
        self.feature_buffer = []
        self.buffer_size = 1000  # Window for online learning

    def extract_features(self, market_data):
        """Extract features for regime classification"""
        try:
            # Feature vector: [returns, volatility, volume_ratio, trend_strength]
            price = market_data.get('close', 0)
            prev_price = market_data.get('prev_close', price)

            returns = (price - prev_price) / prev_price if prev_price > 0 else 0
            volatility = market_data.get('volatility', 0.01)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            trend_strength = market_data.get('trend_strength', 0.0)

            return np.array([returns, volatility, volume_ratio, trend_strength])
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return np.array([0, 0.01, 1.0, 0])

    def update_model(self, features):
        """Online learning: update model with new data"""
        self.feature_buffer.append(features)

        if len(self.feature_buffer) > self.buffer_size:
            self.feature_buffer.pop(0)

        # Refit every 100 samples
        if len(self.feature_buffer) >= 100 and len(self.feature_buffer) % 100 == 0:
            X = np.array(self.feature_buffer)
            self.model.fit(X)
            self.is_fitted = True
            self._cache_model()
            logger.info(f"Model updated with {len(self.feature_buffer)} samples")

    def predict_regime(self, features):
        """Predict current market regime"""
        if not self.is_fitted:
            # Bootstrap: use simple heuristics until model is trained
            return self._bootstrap_regime(features)

        try:
            features_2d = features.reshape(1, -1)
            regime_id = self.model.predict(features_2d)[0]
            confidence = self.model.predict_proba(features_2d)[0][regime_id]

            return {
                'regime': self.regime_labels[regime_id],
                'regime_id': int(regime_id),
                'confidence': float(confidence),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._bootstrap_regime(features)

    def _bootstrap_regime(self, features):
        """Simple rule-based regime classification for cold start"""
        returns, volatility, volume_ratio, trend_strength = features

        if abs(trend_strength) > 0.7:
            regime = 'trending'
            regime_id = 0
        elif volatility > 0.03:
            regime = 'volatile'
            regime_id = 2
        elif abs(returns) < 0.005 and volatility < 0.01:
            regime = 'stable'
            regime_id = 3
        else:
            regime = 'mean_reverting'
            regime_id = 1

        return {
            'regime': regime,
            'regime_id': regime_id,
            'confidence': 0.5,  # Lower confidence for bootstrap
            'timestamp': datetime.utcnow().isoformat(),
            'bootstrap': True
        }

    def _cache_model(self):
        """Cache trained model to Redis"""
        try:
            model_bytes = pickle.dumps(self.model)
            self.redis_client.setex('regime_model', 3600, model_bytes)  # 1 hour TTL
        except Exception as e:
            logger.warning(f"Model caching failed: {e}")

    def _load_cached_model(self):
        """Load model from Redis cache"""
        try:
            model_bytes = self.redis_client.get('regime_model')
            if model_bytes:
                self.model = pickle.loads(model_bytes)
                self.is_fitted = True
                logger.info("Loaded cached model from Redis")
                return True
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")
        return False

def main():
    detector = RegimeDetector()
    detector._load_cached_model()

    # Kafka consumer for market data
    consumer = KafkaConsumer(
        'market_data',
        bootstrap_servers=detector.kafka_bootstrap,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        group_id='regime_detection_group'
    )

    # Kafka producer for regime events
    producer = KafkaProducer(
        bootstrap_servers=detector.kafka_bootstrap,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    logger.info("Regime Detection Service started")

    for message in consumer:
        try:
            market_data = message.value
            features = detector.extract_features(market_data)

            # Update model (online learning)
            detector.update_model(features)

            # Predict regime
            regime_info = detector.predict_regime(features)
            regime_info['symbol'] = market_data.get('symbol', 'UNKNOWN')

            # Publish to Kafka
            producer.send('regime_events', value=regime_info)

            logger.info(f"Regime detected: {regime_info['regime']} "
                       f"(confidence: {regime_info['confidence']:.2f})")

        except Exception as e:
            logger.error(f"Processing error: {e}")

if __name__ == '__main__':
    main()

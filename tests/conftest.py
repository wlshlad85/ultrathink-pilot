"""
Test Fixtures and Configuration for UltraThink Trading System
Wave 1 QA Testing Engineer - Agent 4

This module provides shared fixtures for all test suites including:
- Mock services (Kafka, Redis, TimescaleDB)
- Sample market data
- Regime detection fixtures
- Risk management fixtures
- Inference API fixtures
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import json


# ============================================================================
# MARKET DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        'symbol': 'BTCUSDT',
        'close': 50000.0,
        'prev_close': 49500.0,
        'open': 49800.0,
        'high': 50200.0,
        'low': 49400.0,
        'volume': 1000000.0,
        'volatility': 0.02,
        'volume_ratio': 1.2,
        'trend_strength': 0.5,
        'timestamp': datetime.utcnow().isoformat()
    }


@pytest.fixture
def trending_market_data():
    """Market data representing trending regime"""
    return {
        'symbol': 'BTCUSDT',
        'close': 52000.0,
        'prev_close': 50000.0,
        'volatility': 0.015,
        'volume_ratio': 1.5,
        'trend_strength': 0.85,
        'timestamp': datetime.utcnow().isoformat()
    }


@pytest.fixture
def volatile_market_data():
    """Market data representing volatile regime"""
    return {
        'symbol': 'BTCUSDT',
        'close': 50500.0,
        'prev_close': 50000.0,
        'volatility': 0.05,
        'volume_ratio': 2.0,
        'trend_strength': 0.2,
        'timestamp': datetime.utcnow().isoformat()
    }


@pytest.fixture
def stable_market_data():
    """Market data representing stable regime"""
    return {
        'symbol': 'BTCUSDT',
        'close': 50010.0,
        'prev_close': 50000.0,
        'volatility': 0.005,
        'volume_ratio': 0.9,
        'trend_strength': 0.1,
        'timestamp': datetime.utcnow().isoformat()
    }


@pytest.fixture
def mean_reverting_market_data():
    """Market data representing mean-reverting regime"""
    return {
        'symbol': 'BTCUSDT',
        'close': 49990.0,
        'prev_close': 50000.0,
        'volatility': 0.01,
        'volume_ratio': 1.0,
        'trend_strength': 0.3,
        'timestamp': datetime.utcnow().isoformat()
    }


@pytest.fixture
def market_data_sequence():
    """Sequence of market data for time-series testing"""
    base_price = 50000.0
    data_points = []

    for i in range(100):
        price = base_price * (1 + np.random.normal(0, 0.01))
        data_points.append({
            'symbol': 'BTCUSDT',
            'close': price,
            'prev_close': base_price,
            'volatility': abs(np.random.normal(0.02, 0.005)),
            'volume_ratio': abs(np.random.normal(1.0, 0.2)),
            'trend_strength': np.random.uniform(-1, 1),
            'timestamp': (datetime.utcnow() + timedelta(minutes=i)).isoformat()
        })
        base_price = price

    return data_points


# ============================================================================
# MOCK SERVICE FIXTURES
# ============================================================================

@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis_mock = MagicMock()
    redis_mock.get.return_value = None
    redis_mock.setex.return_value = True
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = False
    return redis_mock


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer"""
    producer_mock = MagicMock()
    producer_mock.send.return_value = MagicMock(get=lambda: None)
    producer_mock.flush.return_value = None
    return producer_mock


@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer"""
    consumer_mock = MagicMock()
    consumer_mock.__iter__ = Mock(return_value=iter([]))
    return consumer_mock


@pytest.fixture
def mock_timescaledb():
    """Mock TimescaleDB connection"""
    db_mock = MagicMock()
    cursor_mock = MagicMock()
    cursor_mock.fetchall.return_value = []
    cursor_mock.fetchone.return_value = None
    db_mock.cursor.return_value = cursor_mock
    return db_mock


# ============================================================================
# REGIME DETECTION FIXTURES
# ============================================================================

@pytest.fixture
def regime_probabilities():
    """Sample regime probability distribution"""
    return {
        'bull': 0.60,
        'bear': 0.15,
        'sideways': 0.25,
        'entropy': 0.85
    }


@pytest.fixture
def regime_history():
    """Historical regime transitions for testing"""
    return [
        {'regime': 'trending', 'confidence': 0.8, 'timestamp': '2024-10-01T10:00:00'},
        {'regime': 'volatile', 'confidence': 0.6, 'timestamp': '2024-10-01T11:00:00'},
        {'regime': 'stable', 'confidence': 0.9, 'timestamp': '2024-10-01T12:00:00'},
        {'regime': 'mean_reverting', 'confidence': 0.7, 'timestamp': '2024-10-01T13:00:00'},
    ]


@pytest.fixture
def trained_regime_features():
    """Features for training regime detection model"""
    np.random.seed(42)
    n_samples = 1000

    # Generate features for 4 regimes
    features = []

    # Trending regime: high trend_strength
    trending = np.column_stack([
        np.random.normal(0.02, 0.01, n_samples // 4),  # returns
        np.random.normal(0.015, 0.005, n_samples // 4),  # volatility
        np.random.normal(1.5, 0.3, n_samples // 4),  # volume_ratio
        np.random.normal(0.8, 0.1, n_samples // 4),  # trend_strength
    ])

    # Volatile regime: high volatility
    volatile = np.column_stack([
        np.random.normal(0.01, 0.02, n_samples // 4),
        np.random.normal(0.05, 0.01, n_samples // 4),
        np.random.normal(2.0, 0.5, n_samples // 4),
        np.random.normal(0.3, 0.2, n_samples // 4),
    ])

    # Stable regime: low volatility, low returns
    stable = np.column_stack([
        np.random.normal(0.001, 0.002, n_samples // 4),
        np.random.normal(0.005, 0.002, n_samples // 4),
        np.random.normal(0.9, 0.1, n_samples // 4),
        np.random.normal(0.1, 0.05, n_samples // 4),
    ])

    # Mean-reverting regime: oscillating returns
    mean_reverting = np.column_stack([
        np.random.normal(0.0, 0.005, n_samples // 4),
        np.random.normal(0.01, 0.003, n_samples // 4),
        np.random.normal(1.0, 0.15, n_samples // 4),
        np.random.normal(0.3, 0.1, n_samples // 4),
    ])

    return np.vstack([trending, volatile, stable, mean_reverting])


# ============================================================================
# RISK MANAGEMENT FIXTURES
# ============================================================================

@pytest.fixture
def portfolio_state():
    """Sample portfolio state"""
    return {
        'cash': 100000.0,
        'positions': {
            'BTCUSDT': {'quantity': 1.0, 'avg_price': 50000.0, 'value': 50000.0},
            'ETHUSDT': {'quantity': 10.0, 'avg_price': 3000.0, 'value': 30000.0},
        },
        'total_value': 180000.0,
        'timestamp': datetime.utcnow().isoformat()
    }


@pytest.fixture
def trade_request():
    """Sample trade request"""
    return {
        'symbol': 'BTCUSDT',
        'action': 'BUY',
        'quantity': 0.5,
        'price': 50000.0,
        'timestamp': datetime.utcnow().isoformat()
    }


@pytest.fixture
def risk_limits():
    """Risk management limits"""
    return {
        'max_position_size': 0.25,  # 25% max per asset
        'max_leverage': 1.0,  # No leverage
        'max_daily_loss': 0.05,  # 5% max daily loss
        'var_confidence': 0.95,  # 95% VaR
        'var_horizon': 1,  # 1-day VaR
    }


# ============================================================================
# INFERENCE API FIXTURES
# ============================================================================

@pytest.fixture
def prediction_request():
    """Sample prediction request"""
    return {
        'symbol': 'BTCUSDT',
        'features': {
            'price': 50000.0,
            'volume': 1000000.0,
            'volatility': 0.02,
            'trend_strength': 0.5,
        },
        'regime_probabilities': {
            'bull': 0.60,
            'bear': 0.15,
            'sideways': 0.25,
        },
        'timestamp': datetime.utcnow().isoformat()
    }


@pytest.fixture
def prediction_response():
    """Sample prediction response"""
    return {
        'action': 'BUY',
        'confidence': 0.85,
        'decision_id': 'dec_123456',
        'risk_approved': True,
        'timestamp': datetime.utcnow().isoformat()
    }


@pytest.fixture
def mock_model():
    """Mock ML model for inference testing"""
    model_mock = MagicMock()
    model_mock.predict.return_value = np.array([1])  # BUY action
    model_mock.predict_proba.return_value = np.array([[0.05, 0.85, 0.10]])  # [SELL, BUY, HOLD]
    return model_mock


# ============================================================================
# INTEGRATION TEST FIXTURES
# ============================================================================

@pytest.fixture
def end_to_end_scenario():
    """Complete end-to-end trading scenario"""
    return {
        'market_data': {
            'symbol': 'BTCUSDT',
            'close': 50000.0,
            'prev_close': 49500.0,
            'volatility': 0.02,
            'volume_ratio': 1.2,
            'trend_strength': 0.5,
        },
        'expected_regime': 'trending',
        'expected_action': 'BUY',
        'expected_risk_approval': True,
        'expected_confidence': 0.75,
    }


@pytest.fixture
def database_connection_params():
    """Database connection parameters for integration tests"""
    return {
        'host': 'localhost',
        'port': 5432,
        'database': 'ultrathink_test',
        'user': 'test_user',
        'password': 'test_password',
    }


# ============================================================================
# PARAMETRIZE HELPERS
# ============================================================================

# Test data for parametrized tests
EXTREME_MARKET_CONDITIONS = [
    {'volatility': 0.10, 'expected_regime': 'volatile'},
    {'volatility': 0.001, 'expected_regime': 'stable'},
    {'trend_strength': 0.95, 'expected_regime': 'trending'},
    {'trend_strength': -0.95, 'expected_regime': 'trending'},
]

EDGE_CASE_PRICES = [
    0.01,  # Very low price
    1000000.0,  # Very high price
    49999.99,  # Just below threshold
    50000.01,  # Just above threshold
]

CONCENTRATION_VIOLATIONS = [
    {'position_size': 0.30, 'expected_approval': False},  # Over 25% limit
    {'position_size': 0.24, 'expected_approval': True},   # Under limit
    {'position_size': 0.25, 'expected_approval': True},   # At limit
]


# ============================================================================
# PYTEST HOOKS
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings"""
    config.addinivalue_line(
        "markers", "wave1: marks tests as part of Wave 1 implementation"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add wave1 marker to all tests in tests/ directory
        if "tests/" in str(item.fspath):
            item.add_marker(pytest.mark.wave1)


# ============================================================================
# ANYIO CONFIGURATION
# ============================================================================

@pytest.fixture(scope='session')
def anyio_backend():
    """Configure anyio to use asyncio backend only."""
    return 'asyncio'


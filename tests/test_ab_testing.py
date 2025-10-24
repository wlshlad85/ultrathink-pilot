"""
Comprehensive tests for A/B Testing framework.
Tests traffic splitting, shadow mode, metrics collection, and integration.
"""
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services', 'inference_service'))

from ab_testing_manager import (
    ABTestingManager, ABTestConfig, ABTestMode, ABTestResult
)


class MockModelCache:
    """Mock model cache for testing."""

    def __init__(self):
        self.models = {
            'control_model': {'version': 'v1'},
            'treatment_model': {'version': 'v2'},
            'bull_specialist': {'version': 'v1'},
            'bull_specialist_v2': {'version': 'v2'}
        }

    def is_loaded(self, model_name: str) -> bool:
        """Check if model is loaded."""
        return model_name in self.models

    def predict(self, model_name: str, features: np.ndarray):
        """Mock prediction."""
        # Simulate different models making different predictions
        if 'v2' in model_name or 'treatment' in model_name:
            # Treatment model: slightly different behavior
            action = "BUY"
            confidence = 0.85
        else:
            # Control model
            action = "HOLD"
            confidence = 0.75

        action_probs = np.array([0.2, 0.3, 0.5])
        return action, confidence, action_probs


class TestABTestConfig:
    """Test A/B test configuration."""

    def test_valid_config(self):
        """Test creating valid config."""
        config = ABTestConfig(
            test_id="test1",
            mode=ABTestMode.TRAFFIC_SPLIT,
            control_model="control",
            treatment_model="treatment",
            traffic_split=0.1
        )

        assert config.test_id == "test1"
        assert config.traffic_split == 0.1
        assert config.enabled is True

    def test_invalid_traffic_split(self):
        """Test that invalid traffic splits are rejected."""
        with pytest.raises(ValueError):
            ABTestConfig(
                test_id="test1",
                mode=ABTestMode.TRAFFIC_SPLIT,
                control_model="control",
                treatment_model="treatment",
                traffic_split=1.5  # Invalid: > 1.0
            )

    def test_shadow_mode_auto_enable(self):
        """Test that shadow mode automatically sets shadow_mode flag."""
        config = ABTestConfig(
            test_id="test1",
            mode=ABTestMode.SHADOW,
            control_model="control",
            treatment_model="treatment",
            traffic_split=0.5
        )

        assert config.shadow_mode is True


class TestABTestingManager:
    """Test A/B testing manager."""

    def test_initialization(self):
        """Test manager initialization."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        assert manager.model_cache == cache
        assert len(manager.active_tests) == 0
        assert len(manager.results_buffer) == 0

    def test_create_test(self):
        """Test creating an A/B test."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        config = manager.create_test(
            test_id="test1",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=0.1
        )

        assert config.test_id == "test1"
        assert config.control_model == "control_model"
        assert config.treatment_model == "treatment_model"
        assert config.traffic_split == 0.1

        # Verify it's in active tests
        assert "test1" in manager.active_tests

    def test_create_test_invalid_model(self):
        """Test that creating test with invalid model fails."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        with pytest.raises(ValueError, match="not loaded"):
            manager.create_test(
                test_id="test1",
                control_model="nonexistent_model",
                treatment_model="treatment_model",
                traffic_split=0.1
            )

    def test_assign_group_consistency(self):
        """Test that group assignment is consistent for same request_id."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        manager.create_test(
            test_id="test1",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=0.5
        )

        # Same request_id should always get same group
        request_id = "req_123"
        group1 = manager.assign_group(request_id, "test1")
        group2 = manager.assign_group(request_id, "test1")
        group3 = manager.assign_group(request_id, "test1")

        assert group1 == group2 == group3

    def test_traffic_split_distribution(self):
        """Test that traffic split is approximately correct."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        # Create test with 20% traffic to treatment
        manager.create_test(
            test_id="test1",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=0.2
        )

        # Generate 1000 assignments
        n_samples = 1000
        treatment_count = 0

        for i in range(n_samples):
            group = manager.assign_group(f"req_{i}", "test1")
            if group == "treatment":
                treatment_count += 1

        # Should be approximately 20% (within ±5%)
        actual_split = treatment_count / n_samples
        assert 0.15 <= actual_split <= 0.25, f"Split was {actual_split:.2%}, expected ~20%"

    @pytest.mark.anyio
    async def test_predict_single_control(self):
        """Test single prediction with control model."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        manager.create_test(
            test_id="test1",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=0.0  # All traffic to control
        )

        features = np.random.randn(43)
        action, confidence, probs, result = await manager.predict_with_ab_test(
            test_id="test1",
            request_id="req_1",
            features=features,
            symbol="BTC-USD"
        )

        assert action == "HOLD"  # Control model returns HOLD
        assert result.assigned_group == "control"
        assert result.control_model == "control_model"
        assert result.treatment_model is None  # Not in shadow mode

    @pytest.mark.anyio
    async def test_predict_single_treatment(self):
        """Test single prediction with treatment model."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        manager.create_test(
            test_id="test1",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=1.0  # All traffic to treatment
        )

        features = np.random.randn(43)
        action, confidence, probs, result = await manager.predict_with_ab_test(
            test_id="test1",
            request_id="req_1",
            features=features,
            symbol="BTC-USD"
        )

        assert action == "BUY"  # Treatment model returns BUY
        assert result.assigned_group == "treatment"
        assert result.treatment_model == "treatment_model"

    @pytest.mark.anyio
    async def test_predict_shadow_mode(self):
        """Test shadow mode prediction (runs both models)."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        manager.create_test(
            test_id="test1",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=0.5,
            mode=ABTestMode.SHADOW
        )

        features = np.random.randn(43)
        action, confidence, probs, result = await manager.predict_with_ab_test(
            test_id="test1",
            request_id="req_1",
            features=features,
            symbol="BTC-USD"
        )

        # Should ALWAYS return control prediction in shadow mode
        assert action == "HOLD"
        assert result.assigned_group == "shadow"

        # But should have run treatment model too
        assert result.treatment_model == "treatment_model"
        assert result.treatment_action == "BUY"
        assert result.treatment_confidence is not None

        # Should have comparison metrics
        assert result.actions_match is not None
        assert result.confidence_delta is not None
        assert result.latency_delta_ms is not None

    def test_disable_enable_test(self):
        """Test disabling and enabling tests."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        manager.create_test(
            test_id="test1",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=0.5
        )

        # Disable test
        manager.disable_test("test1")
        assert manager.active_tests["test1"].enabled is False

        # Enable test
        manager.enable_test("test1")
        assert manager.active_tests["test1"].enabled is True

    def test_update_traffic_split(self):
        """Test updating traffic split."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        manager.create_test(
            test_id="test1",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=0.05
        )

        # Update to 10%
        manager.update_traffic_split("test1", 0.10)
        assert manager.active_tests["test1"].traffic_split == 0.10

        # Update to 50%
        manager.update_traffic_split("test1", 0.50)
        assert manager.active_tests["test1"].traffic_split == 0.50

    def test_update_traffic_split_invalid(self):
        """Test that invalid traffic split updates are rejected."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        manager.create_test(
            test_id="test1",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=0.05
        )

        with pytest.raises(ValueError):
            manager.update_traffic_split("test1", 1.5)

    @pytest.mark.anyio
    async def test_results_buffering(self):
        """Test that results are buffered."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)
        manager.buffer_max_size = 10  # Small buffer for testing

        manager.create_test(
            test_id="test1",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=0.5
        )

        # Make predictions to fill buffer
        features = np.random.randn(43)
        for i in range(5):
            await manager.predict_with_ab_test(
                test_id="test1",
                request_id=f"req_{i}",
                features=features
            )

        # Buffer should have 5 results
        assert len(manager.results_buffer) == 5

    def test_get_test_stats(self):
        """Test getting test statistics from buffer."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        manager.create_test(
            test_id="test1",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=0.2
        )

        # Add some mock results to buffer
        for i in range(10):
            group = "treatment" if i < 2 else "control"
            result = ABTestResult(
                test_id="test1",
                request_id=f"req_{i}",
                timestamp=datetime.utcnow(),
                assigned_group=group,
                control_model="control_model",
                control_action="HOLD",
                control_confidence=0.8,
                control_latency_ms=10.0
            )
            manager.results_buffer.append(result)

        # Get stats
        stats = manager.get_test_stats("test1")

        assert stats['sample_size'] == 10
        assert stats['control_count'] == 8
        assert stats['treatment_count'] == 2
        assert stats['control_pct'] == 0.8
        assert stats['treatment_pct'] == 0.2

    def test_list_tests(self):
        """Test listing active tests."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        # Create multiple tests
        manager.create_test(
            test_id="test1",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=0.1
        )

        manager.create_test(
            test_id="test2",
            control_model="bull_specialist",
            treatment_model="bull_specialist_v2",
            traffic_split=0.05,
            mode=ABTestMode.SHADOW
        )

        # List tests
        tests = manager.list_tests()

        assert len(tests) == 2
        assert any(t['test_id'] == 'test1' for t in tests)
        assert any(t['test_id'] == 'test2' for t in tests)


class TestABTestResult:
    """Test A/B test result data structure."""

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ABTestResult(
            test_id="test1",
            request_id="req_1",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            assigned_group="control",
            control_model="model_v1",
            control_action="BUY",
            control_confidence=0.85,
            control_latency_ms=12.5,
            symbol="BTC-USD"
        )

        result_dict = result.to_dict()

        assert result_dict['test_id'] == "test1"
        assert result_dict['request_id'] == "req_1"
        assert result_dict['assigned_group'] == "control"
        assert result_dict['symbol'] == "BTC-USD"
        # Timestamp should be ISO format string
        assert isinstance(result_dict['timestamp'], str)


class TestIntegration:
    """Integration tests for A/B testing."""

    @pytest.mark.anyio
    async def test_end_to_end_traffic_split(self):
        """Test complete flow with traffic splitting."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        # Create test
        manager.create_test(
            test_id="canary_test",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=0.1,
            description="10% canary deployment"
        )

        # Run 100 predictions
        features = np.random.randn(43)
        results = []

        for i in range(100):
            action, conf, probs, result = await manager.predict_with_ab_test(
                test_id="canary_test",
                request_id=f"req_{i}",
                features=features,
                symbol="BTC-USD"
            )
            results.append(result)

        # Verify traffic split
        treatment_count = sum(1 for r in results if r.assigned_group == "treatment")
        control_count = sum(1 for r in results if r.assigned_group == "control")

        assert control_count + treatment_count == 100
        # Should be approximately 10% treatment (within reasonable bounds)
        assert 5 <= treatment_count <= 15

    @pytest.mark.anyio
    async def test_end_to_end_shadow_mode(self):
        """Test complete flow with shadow mode."""
        cache = MockModelCache()
        manager = ABTestingManager(model_cache=cache)

        # Create shadow mode test
        manager.create_test(
            test_id="shadow_test",
            control_model="control_model",
            treatment_model="treatment_model",
            traffic_split=0.5,
            mode=ABTestMode.SHADOW,
            description="Shadow mode comparison"
        )

        # Run 50 predictions
        features = np.random.randn(43)
        results = []

        for i in range(50):
            action, conf, probs, result = await manager.predict_with_ab_test(
                test_id="shadow_test",
                request_id=f"req_{i}",
                features=features,
                symbol="BTC-USD"
            )
            results.append(result)

            # All predictions should use control
            assert action == "HOLD"  # Control model action

        # All results should be shadow mode
        assert all(r.assigned_group == "shadow" for r in results)

        # All should have both control and treatment predictions
        assert all(r.control_action is not None for r in results)
        assert all(r.treatment_action is not None for r in results)

        # All should have comparison metrics
        assert all(r.actions_match is not None for r in results)
        assert all(r.confidence_delta is not None for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

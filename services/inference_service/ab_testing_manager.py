"""
A/B Testing Manager for safe model rollouts.

Supports:
- Traffic splitting with configurable percentages (e.g., 5% canary, 95% control)
- Shadow mode: run both models in parallel, compare without affecting trades
- Request ID tracking for consistent A/B group assignment
- Metrics collection and comparison
- MLflow integration for multi-version model loading
"""
import logging
import hashlib
import time
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any
from enum import Enum
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


class ABTestMode(str, Enum):
    """A/B test execution modes."""
    TRAFFIC_SPLIT = "traffic_split"  # Route requests to different models
    SHADOW = "shadow"  # Run both models, only use control for actual decision
    DISABLED = "disabled"  # No A/B testing


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    test_id: str
    mode: ABTestMode
    control_model: str  # Name of control model
    treatment_model: str  # Name of treatment model
    traffic_split: float  # Percentage of traffic to treatment (0.0-1.0)
    enabled: bool = True

    # Shadow mode settings
    shadow_mode: bool = False

    # Metadata
    created_at: datetime = None
    description: str = ""

    def __post_init__(self):
        """Validate configuration."""
        if self.traffic_split < 0.0 or self.traffic_split > 1.0:
            raise ValueError(f"traffic_split must be between 0.0 and 1.0, got {self.traffic_split}")

        if self.mode == ABTestMode.SHADOW and not self.shadow_mode:
            self.shadow_mode = True

        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ABTestResult:
    """Result from an A/B test prediction."""
    test_id: str
    request_id: str
    timestamp: datetime

    # Group assignment
    assigned_group: str  # "control" or "treatment"

    # Control model results
    control_model: str
    control_action: str
    control_confidence: float
    control_latency_ms: float

    # Treatment model results (None if not in shadow mode or treatment group)
    treatment_model: Optional[str] = None
    treatment_action: Optional[str] = None
    treatment_confidence: Optional[float] = None
    treatment_latency_ms: Optional[float] = None

    # Comparison metrics (for shadow mode)
    actions_match: Optional[bool] = None
    confidence_delta: Optional[float] = None
    latency_delta_ms: Optional[float] = None

    # Context
    symbol: str = ""
    features: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        result = asdict(self)
        # Convert datetime to ISO format
        result['timestamp'] = self.timestamp.isoformat()
        if 'features' in result and result['features'] is not None:
            result['features'] = json.dumps(result['features'])
        return result


class ABTestingManager:
    """
    Manages A/B testing for model deployments.

    Features:
    - Traffic splitting: Route X% to treatment, (100-X)% to control
    - Shadow mode: Run both models, compare predictions, use control for actual decision
    - Consistent hashing: Same request ID always goes to same group
    - Metrics tracking: Store results for analysis
    """

    def __init__(self, model_cache=None, storage_backend=None):
        """
        Initialize A/B testing manager.

        Args:
            model_cache: ModelCache instance for loading models
            storage_backend: Optional backend for storing test results (e.g., TimescaleDB)
        """
        self.model_cache = model_cache
        self.storage_backend = storage_backend

        # Active tests (test_id -> ABTestConfig)
        self.active_tests: Dict[str, ABTestConfig] = {}

        # Results buffer for batch writes
        self.results_buffer: List[ABTestResult] = []
        self.buffer_max_size = 100

        logger.info("ABTestingManager initialized")

    def create_test(
        self,
        test_id: str,
        control_model: str,
        treatment_model: str,
        traffic_split: float = 0.05,
        mode: ABTestMode = ABTestMode.TRAFFIC_SPLIT,
        description: str = ""
    ) -> ABTestConfig:
        """
        Create a new A/B test.

        Args:
            test_id: Unique identifier for the test
            control_model: Name of control model (baseline)
            treatment_model: Name of treatment model (new version)
            traffic_split: Fraction of traffic to route to treatment (0.0-1.0)
            mode: Test mode (traffic_split or shadow)
            description: Human-readable description

        Returns:
            ABTestConfig instance
        """
        # Verify models are loaded
        if self.model_cache:
            if not self.model_cache.is_loaded(control_model):
                raise ValueError(f"Control model '{control_model}' not loaded")
            if not self.model_cache.is_loaded(treatment_model):
                raise ValueError(f"Treatment model '{treatment_model}' not loaded")

        config = ABTestConfig(
            test_id=test_id,
            mode=mode,
            control_model=control_model,
            treatment_model=treatment_model,
            traffic_split=traffic_split,
            shadow_mode=(mode == ABTestMode.SHADOW),
            description=description
        )

        self.active_tests[test_id] = config
        logger.info(f"Created A/B test '{test_id}': {control_model} vs {treatment_model} "
                   f"(split={traffic_split:.1%}, mode={mode})")

        return config

    def assign_group(self, request_id: str, test_id: str) -> str:
        """
        Assign request to control or treatment group using consistent hashing.

        Args:
            request_id: Unique request identifier
            test_id: A/B test identifier

        Returns:
            "control" or "treatment"
        """
        if test_id not in self.active_tests:
            return "control"

        config = self.active_tests[test_id]

        # Hash request_id to get consistent assignment
        hash_input = f"{request_id}:{test_id}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)

        # Normalize to 0-1 range
        normalized = (hash_value % 10000) / 10000.0

        # Compare to traffic split
        if normalized < config.traffic_split:
            return "treatment"
        else:
            return "control"

    async def predict_with_ab_test(
        self,
        test_id: str,
        request_id: str,
        features: np.ndarray,
        symbol: str = ""
    ) -> Tuple[str, float, np.ndarray, ABTestResult]:
        """
        Make prediction with A/B testing.

        In traffic_split mode:
        - Routes request to either control or treatment model
        - Returns prediction from assigned model

        In shadow mode:
        - Runs BOTH models
        - Returns prediction from control model (safe)
        - Compares both predictions for analysis

        Args:
            test_id: A/B test identifier
            request_id: Unique request ID
            features: Feature vector for prediction
            symbol: Trading symbol (for logging)

        Returns:
            Tuple of (action, confidence, action_probs, ab_result)
        """
        if test_id not in self.active_tests:
            raise ValueError(f"A/B test '{test_id}' not found")

        config = self.active_tests[test_id]

        if not config.enabled:
            # Test is disabled, use control model
            return await self._predict_single(
                config.control_model,
                features,
                test_id,
                request_id,
                "control",
                symbol
            )

        # Assign to group
        assigned_group = self.assign_group(request_id, test_id)

        if config.mode == ABTestMode.SHADOW:
            # Shadow mode: run both models, use control for actual decision
            return await self._predict_shadow(
                config,
                request_id,
                features,
                symbol
            )

        else:  # TRAFFIC_SPLIT mode
            # Route to assigned model
            if assigned_group == "treatment":
                return await self._predict_single(
                    config.treatment_model,
                    features,
                    test_id,
                    request_id,
                    "treatment",
                    symbol
                )
            else:
                return await self._predict_single(
                    config.control_model,
                    features,
                    test_id,
                    request_id,
                    "control",
                    symbol
                )

    async def _predict_single(
        self,
        model_name: str,
        features: np.ndarray,
        test_id: str,
        request_id: str,
        group: str,
        symbol: str
    ) -> Tuple[str, float, np.ndarray, ABTestResult]:
        """Make prediction with a single model."""
        start_time = time.time()

        # Make prediction
        action, confidence, action_probs = self.model_cache.predict(model_name, features)

        latency_ms = (time.time() - start_time) * 1000

        # Create result record
        result = ABTestResult(
            test_id=test_id,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            assigned_group=group,
            control_model=model_name if group == "control" else self.active_tests[test_id].control_model,
            control_action=action if group == "control" else None,
            control_confidence=confidence if group == "control" else None,
            control_latency_ms=latency_ms if group == "control" else None,
            treatment_model=model_name if group == "treatment" else None,
            treatment_action=action if group == "treatment" else None,
            treatment_confidence=confidence if group == "treatment" else None,
            treatment_latency_ms=latency_ms if group == "treatment" else None,
            symbol=symbol
        )

        # Store result
        await self._store_result(result)

        return action, confidence, action_probs, result

    async def _predict_shadow(
        self,
        config: ABTestConfig,
        request_id: str,
        features: np.ndarray,
        symbol: str
    ) -> Tuple[str, float, np.ndarray, ABTestResult]:
        """
        Shadow mode: run both models, compare predictions.
        Always returns control model prediction for actual use.
        """
        # Run control model
        control_start = time.time()
        control_action, control_confidence, control_probs = self.model_cache.predict(
            config.control_model, features
        )
        control_latency_ms = (time.time() - control_start) * 1000

        # Run treatment model
        treatment_start = time.time()
        treatment_action, treatment_confidence, treatment_probs = self.model_cache.predict(
            config.treatment_model, features
        )
        treatment_latency_ms = (time.time() - treatment_start) * 1000

        # Compare predictions
        actions_match = (control_action == treatment_action)
        confidence_delta = treatment_confidence - control_confidence
        latency_delta_ms = treatment_latency_ms - control_latency_ms

        # Create result record
        result = ABTestResult(
            test_id=config.test_id,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            assigned_group="shadow",
            control_model=config.control_model,
            control_action=control_action,
            control_confidence=control_confidence,
            control_latency_ms=control_latency_ms,
            treatment_model=config.treatment_model,
            treatment_action=treatment_action,
            treatment_confidence=treatment_confidence,
            treatment_latency_ms=treatment_latency_ms,
            actions_match=actions_match,
            confidence_delta=confidence_delta,
            latency_delta_ms=latency_delta_ms,
            symbol=symbol
        )

        # Log if predictions differ
        if not actions_match:
            logger.warning(
                f"Shadow mode disagreement: {config.control_model} predicted {control_action}, "
                f"{config.treatment_model} predicted {treatment_action} "
                f"(confidence delta: {confidence_delta:+.3f})"
            )

        # Store result
        await self._store_result(result)

        # ALWAYS return control prediction in shadow mode
        return control_action, control_confidence, control_probs, result

    async def _store_result(self, result: ABTestResult):
        """
        Store A/B test result.
        Uses buffering for efficiency.
        """
        self.results_buffer.append(result)

        # Flush buffer if full
        if len(self.results_buffer) >= self.buffer_max_size:
            await self.flush_results()

    async def flush_results(self):
        """Flush results buffer to storage backend."""
        if not self.results_buffer:
            return

        if self.storage_backend is None:
            logger.debug(f"No storage backend, discarding {len(self.results_buffer)} results")
            self.results_buffer.clear()
            return

        try:
            # Convert to dicts for storage
            records = [r.to_dict() for r in self.results_buffer]

            # Store (implementation depends on backend)
            await self.storage_backend.store_ab_results(records)

            logger.info(f"Flushed {len(records)} A/B test results to storage")
            self.results_buffer.clear()

        except Exception as e:
            logger.error(f"Failed to flush A/B test results: {e}", exc_info=True)
            # Keep buffer for retry

    def get_test_stats(self, test_id: str) -> Dict[str, Any]:
        """
        Get statistics for an A/B test from buffered results.
        For full statistics, query the storage backend directly.

        Args:
            test_id: Test identifier

        Returns:
            Dictionary with test statistics
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test '{test_id}' not found")

        # Filter results for this test from buffer
        test_results = [r for r in self.results_buffer if r.test_id == test_id]

        if not test_results:
            return {
                "test_id": test_id,
                "sample_size": 0,
                "message": "No results in buffer. Query storage backend for full statistics."
            }

        # Calculate stats
        total = len(test_results)
        control_count = sum(1 for r in test_results if r.assigned_group == "control")
        treatment_count = sum(1 for r in test_results if r.assigned_group == "treatment")
        shadow_count = sum(1 for r in test_results if r.assigned_group == "shadow")

        stats = {
            "test_id": test_id,
            "sample_size": total,
            "control_count": control_count,
            "treatment_count": treatment_count,
            "shadow_count": shadow_count,
            "control_pct": control_count / total if total > 0 else 0,
            "treatment_pct": treatment_count / total if total > 0 else 0,
        }

        # Shadow mode specific stats
        shadow_results = [r for r in test_results if r.actions_match is not None]
        if shadow_results:
            stats["shadow_mode"] = {
                "agreement_rate": sum(1 for r in shadow_results if r.actions_match) / len(shadow_results),
                "avg_confidence_delta": np.mean([r.confidence_delta for r in shadow_results]),
                "avg_latency_delta_ms": np.mean([r.latency_delta_ms for r in shadow_results]),
            }

        return stats

    def disable_test(self, test_id: str):
        """Disable an A/B test (will route all traffic to control)."""
        if test_id not in self.active_tests:
            raise ValueError(f"Test '{test_id}' not found")

        self.active_tests[test_id].enabled = False
        logger.info(f"Disabled A/B test '{test_id}'")

    def enable_test(self, test_id: str):
        """Re-enable an A/B test."""
        if test_id not in self.active_tests:
            raise ValueError(f"Test '{test_id}' not found")

        self.active_tests[test_id].enabled = True
        logger.info(f"Enabled A/B test '{test_id}'")

    def update_traffic_split(self, test_id: str, new_split: float):
        """
        Update traffic split for a test (e.g., ramp from 5% to 10% to 50%).

        Args:
            test_id: Test identifier
            new_split: New traffic split (0.0-1.0)
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test '{test_id}' not found")

        if new_split < 0.0 or new_split > 1.0:
            raise ValueError(f"traffic_split must be between 0.0 and 1.0, got {new_split}")

        old_split = self.active_tests[test_id].traffic_split
        self.active_tests[test_id].traffic_split = new_split

        logger.info(f"Updated traffic split for '{test_id}': {old_split:.1%} -> {new_split:.1%}")

    def list_tests(self) -> List[Dict]:
        """List all active A/B tests."""
        return [
            {
                "test_id": config.test_id,
                "control_model": config.control_model,
                "treatment_model": config.treatment_model,
                "traffic_split": config.traffic_split,
                "mode": config.mode,
                "enabled": config.enabled,
                "created_at": config.created_at.isoformat(),
                "description": config.description
            }
            for config in self.active_tests.values()
        ]

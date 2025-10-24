"""
Circuit Breaker Implementation for UltraThink Services

Provides failure protection for external service calls with:
- Automatic circuit breaking on repeated failures
- Exponential backoff retry logic
- Fallback mechanisms
- Metrics collection

Usage:
    from common_utils.circuit_breaker import circuit_breaker, retry_with_backoff

    @circuit_breaker(failure_threshold=5, timeout=60)
    @retry_with_backoff(max_retries=3)
    def call_external_service():
        # Your service call here
        pass
"""

import functools
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failures exceeded threshold, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation with state tracking."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        """Initialize circuit breaker.

        Args:
            name: Unique identifier for this circuit
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting half-open
            expected_exception: Exception type to track as failure
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now()

        self._lock = Lock()

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"threshold={failure_threshold}, timeout={timeout}s"
        )

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception if circuit is closed/half-open
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.last_state_change = datetime.now()
                    logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN "
                        f"(failures: {self.failure_count}/{self.failure_threshold})"
                    )

        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout)

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self.success_count += 1
            self.failure_count = 0

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.last_state_change = datetime.now()
                logger.info(
                    f"Circuit '{self.name}' recovered, transitioning to CLOSED "
                    f"(successes: {self.success_count})"
                )

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                if self.state != CircuitState.OPEN:
                    self.state = CircuitState.OPEN
                    self.last_state_change = datetime.now()
                    logger.warning(
                        f"Circuit '{self.name}' OPENED "
                        f"(failures: {self.failure_count}/{self.failure_threshold})"
                    )

    def reset(self):
        """Manually reset circuit breaker."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.last_state_change = datetime.now()
            logger.info(f"Circuit '{self.name}' manually reset")

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state.

        Returns:
            State information dictionary
        """
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'failure_threshold': self.failure_threshold,
                'timeout': self.timeout,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'last_state_change': self.last_state_change.isoformat()
            }


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: int = 60,
    expected_exception: type = Exception
) -> CircuitBreaker:
    """Get or create a circuit breaker.

    Args:
        name: Unique identifier for this circuit
        failure_threshold: Number of failures before opening circuit
        timeout: Seconds to wait before attempting half-open
        expected_exception: Exception type to track as failure

    Returns:
        CircuitBreaker instance
    """
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                timeout=timeout,
                expected_exception=expected_exception
            )
        return _circuit_breakers[name]


def get_all_circuit_states() -> Dict[str, Dict[str, Any]]:
    """Get states of all registered circuit breakers.

    Returns:
        Dictionary mapping circuit names to their states
    """
    with _registry_lock:
        return {name: cb.get_state() for name, cb in _circuit_breakers.items()}


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    timeout: int = 60,
    expected_exception: type = Exception,
    fallback: Optional[Callable] = None
):
    """Decorator to add circuit breaker protection to a function.

    Args:
        name: Circuit breaker name (defaults to function name)
        failure_threshold: Number of failures before opening circuit
        timeout: Seconds to wait before attempting half-open
        expected_exception: Exception type to track as failure
        fallback: Optional fallback function to call when circuit is open

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        circuit_name = name or f"{func.__module__}.{func.__name__}"
        cb = get_circuit_breaker(
            circuit_name,
            failure_threshold=failure_threshold,
            timeout=timeout,
            expected_exception=expected_exception
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return cb.call(func, *args, **kwargs)
            except CircuitBreakerError as e:
                if fallback:
                    logger.warning(f"Circuit breaker open, using fallback: {e}")
                    return fallback(*args, **kwargs)
                raise

        return wrapper
    return decorator


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True,
    exceptions: tuple = (Exception,)
):
    """Decorator to add retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential: Use exponential backoff if True, else constant
        exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise

                    # Calculate delay
                    if exponential:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)

            raise last_exception  # Should never reach here

        return wrapper
    return decorator


class HealthCheckFailure(Exception):
    """Raised when health check fails."""
    pass


def health_check_endpoint(
    check_functions: Dict[str, Callable[[], bool]],
    include_circuit_breakers: bool = True
) -> Dict[str, Any]:
    """Generate health check response.

    Args:
        check_functions: Dict mapping component names to health check functions
        include_circuit_breakers: Include circuit breaker states in response

    Returns:
        Health check response dictionary
    """
    checks = {}
    overall_status = "healthy"

    # Run health checks
    for component, check_func in check_functions.items():
        try:
            result = check_func()
            checks[component] = {
                'status': 'healthy' if result else 'unhealthy',
                'healthy': result
            }
            if not result:
                overall_status = "degraded"
        except Exception as e:
            checks[component] = {
                'status': 'unhealthy',
                'healthy': False,
                'error': str(e)
            }
            overall_status = "unhealthy"

    response = {
        'status': overall_status,
        'timestamp': datetime.now().isoformat(),
        'checks': checks
    }

    # Include circuit breaker states
    if include_circuit_breakers:
        circuit_states = get_all_circuit_states()
        response['circuit_breakers'] = circuit_states

        # Mark as degraded if any circuit is open
        if any(cb['state'] == 'open' for cb in circuit_states.values()):
            overall_status = "degraded"
            response['status'] = overall_status

    return response


# Example usage functions
def example_database_health_check() -> bool:
    """Example database health check."""
    # Implement actual DB check
    return True


def example_cache_health_check() -> bool:
    """Example cache health check."""
    # Implement actual cache check
    return True


def example_kafka_health_check() -> bool:
    """Example Kafka health check."""
    # Implement actual Kafka check
    return True


# Health check configuration for services
SERVICE_HEALTH_CHECKS = {
    'database': example_database_health_check,
    'cache': example_cache_health_check,
    'kafka': example_kafka_health_check
}


if __name__ == '__main__':
    # Example usage
    @circuit_breaker(name="test_service", failure_threshold=3, timeout=30)
    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def test_function():
        """Example protected function."""
        # Simulate external call
        import random
        if random.random() < 0.5:
            raise ConnectionError("Simulated failure")
        return "Success"

    # Test the function
    for i in range(10):
        try:
            result = test_function()
            print(f"Attempt {i+1}: {result}")
        except (CircuitBreakerError, ConnectionError) as e:
            print(f"Attempt {i+1}: Failed - {e}")
        time.sleep(1)

    # Print circuit state
    print("\nCircuit Breaker States:")
    import json
    print(json.dumps(get_all_circuit_states(), indent=2))

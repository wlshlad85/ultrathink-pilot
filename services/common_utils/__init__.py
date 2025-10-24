"""
Common utilities for UltraThink services.

This package provides shared utilities for all microservices including:
- Circuit breakers for fault tolerance
- Retry logic with exponential backoff
- Health check utilities
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    circuit_breaker,
    get_all_circuit_states,
    get_circuit_breaker,
    health_check_endpoint,
    retry_with_backoff,
)

__all__ = [
    'CircuitBreaker',
    'CircuitBreakerError',
    'CircuitState',
    'circuit_breaker',
    'get_all_circuit_states',
    'get_circuit_breaker',
    'health_check_endpoint',
    'retry_with_backoff',
]

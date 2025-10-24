#!/usr/bin/env python3
"""
Circuit Breaker Validation Test

Tests the circuit breaker implementation to ensure:
- Circuit opens after threshold failures
- Circuit transitions to half-open after timeout
- Circuit closes after successful recovery
- Retry logic works correctly
"""

import sys
import time
from pathlib import Path

# Add services directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'services'))

from common_utils.circuit_breaker import (
    circuit_breaker,
    retry_with_backoff,
    CircuitBreakerError,
    get_all_circuit_states,
    get_circuit_breaker
)


# Test counter for simulating failures
call_count = 0
fail_until = 5


@circuit_breaker(name="test_service", failure_threshold=3, timeout=5)
def flaky_service():
    """Simulates a flaky service that fails the first N calls."""
    global call_count
    call_count += 1

    if call_count <= fail_until:
        print(f"  Call {call_count}: FAILING")
        raise ConnectionError(f"Simulated failure {call_count}")

    print(f"  Call {call_count}: SUCCESS")
    return "Success"


@circuit_breaker(name="retry_test", failure_threshold=5, timeout=10)
@retry_with_backoff(max_retries=3, base_delay=0.5, exponential=True)
def service_with_retry():
    """Service with retry logic."""
    global call_count
    call_count += 1

    if call_count <= 2:
        print(f"  Retry attempt {call_count}: FAILING")
        raise ConnectionError(f"Retry failure {call_count}")

    print(f"  Retry attempt {call_count}: SUCCESS")
    return "Retry succeeded"


def test_circuit_breaker():
    """Test circuit breaker functionality."""
    global call_count, fail_until

    print("=" * 60)
    print("Circuit Breaker Validation Tests")
    print("=" * 60)

    # Test 1: Circuit opens after failures
    print("\nTest 1: Circuit opens after threshold failures")
    print("-" * 60)
    call_count = 0
    fail_until = 10  # Keep failing

    for i in range(6):
        try:
            result = flaky_service()
            print(f"  Attempt {i+1}: {result}")
        except (CircuitBreakerError, ConnectionError) as e:
            print(f"  Attempt {i+1}: {type(e).__name__}: {e}")
        time.sleep(0.5)

    # Check circuit state
    states = get_all_circuit_states()
    test_service_state = states.get('test_service', {})
    print(f"\nCircuit state after failures: {test_service_state.get('state')}")
    print(f"Failure count: {test_service_state.get('failure_count')}")

    assert test_service_state.get('state') == 'open', "Circuit should be OPEN after threshold failures"
    print("✓ Test 1 PASSED: Circuit opened correctly")

    # Test 2: Circuit transitions to half-open after timeout
    print("\n\nTest 2: Circuit transitions to half-open after timeout")
    print("-" * 60)
    print(f"Waiting {5} seconds for timeout...")
    time.sleep(6)

    # Service now succeeds
    call_count = 0
    fail_until = 0

    try:
        result = flaky_service()
        print(f"  After timeout: {result}")
    except CircuitBreakerError as e:
        print(f"  After timeout: CircuitBreakerError: {e}")

    states = get_all_circuit_states()
    test_service_state = states.get('test_service', {})
    print(f"\nCircuit state after timeout: {test_service_state.get('state')}")

    # Should be closed now after successful call
    assert test_service_state.get('state') == 'closed', "Circuit should be CLOSED after successful recovery"
    print("✓ Test 2 PASSED: Circuit recovered correctly")

    # Test 3: Retry logic works
    print("\n\nTest 3: Retry logic with exponential backoff")
    print("-" * 60)
    call_count = 0

    start_time = time.time()
    try:
        result = service_with_retry()
        elapsed = time.time() - start_time
        print(f"  Final result: {result}")
        print(f"  Total time: {elapsed:.2f}s (should include backoff delays)")
    except Exception as e:
        print(f"  Failed: {e}")

    # Should have taken ~1.5s (0.5s + 1.0s delays before success on 3rd attempt)
    assert call_count == 3, f"Should have made 3 attempts, made {call_count}"
    print("✓ Test 3 PASSED: Retry logic worked correctly")

    # Test 4: Manual circuit reset
    print("\n\nTest 4: Manual circuit reset")
    print("-" * 60)

    cb = get_circuit_breaker("test_service")
    cb.reset()

    states = get_all_circuit_states()
    test_service_state = states.get('test_service', {})
    print(f"Circuit state after manual reset: {test_service_state.get('state')}")
    print(f"Failure count: {test_service_state.get('failure_count')}")

    assert test_service_state.get('state') == 'closed', "Circuit should be CLOSED after reset"
    assert test_service_state.get('failure_count') == 0, "Failure count should be 0"
    print("✓ Test 4 PASSED: Manual reset worked correctly")

    # Summary
    print("\n" + "=" * 60)
    print("All Circuit Breaker Tests PASSED ✓")
    print("=" * 60)

    print("\nFinal Circuit Breaker States:")
    for name, state in get_all_circuit_states().items():
        print(f"\n{name}:")
        print(f"  State: {state['state']}")
        print(f"  Failures: {state['failure_count']}")
        print(f"  Successes: {state['success_count']}")


if __name__ == '__main__':
    try:
        test_circuit_breaker()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

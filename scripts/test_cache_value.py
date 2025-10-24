#!/usr/bin/env python3
"""
Test Cache Value - Demonstrates when cache provides speedup
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.trading_env_v3 import TradingEnvV3

def test_cache_value():
    """
    Demonstrate cache value by creating multiple environments.

    First environment: Cache MISS (loads data)
    Subsequent environments: Cache HIT (reuses cached data)
    """
    print("=" * 80)
    print("CACHE VALUE DEMONSTRATION")
    print("=" * 80)
    print()
    print("Creating 5 environments with caching enabled...")
    print("Expected: First is slow (cache miss), rest are fast (cache hits)")
    print()

    times = []

    for i in range(1, 6):
        start = time.time()

        env = TradingEnvV3(
            symbol="BTC-USD",
            start_date="2023-01-01",
            end_date="2023-12-31",
            enable_cache=True
        )

        elapsed = time.time() - start
        times.append(elapsed)

        # Get cache stats
        cache_stats = env.get_cache_stats()

        print(f"Environment {i}:")
        print(f"  Creation time: {elapsed:.3f}s")
        if cache_stats:
            print(f"  Cache hit rate: {cache_stats['hit_rate_pct']:.1f}%")
            print(f"  Cache requests: {cache_stats['total_requests']}")
        print()

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"First env (cache miss):  {times[0]:.3f}s")
    print(f"Avg rest (cache hits):   {sum(times[1:])/len(times[1:]):.3f}s")
    print(f"Speedup from cache:      {times[0] / (sum(times[1:])/len(times[1:])):.2f}x")
    print()

    # Final cache stats from last env
    final_env = TradingEnvV3(
        symbol="BTC-USD",
        start_date="2023-01-01",
        end_date="2023-12-31",
        enable_cache=True
    )
    final_stats = final_env.get_cache_stats()

    if final_stats:
        print("Final Cache Statistics:")
        print(f"  Total requests: {final_stats['total_requests']}")
        print(f"  Cache hits: {final_stats['hits']}")
        print(f"  Cache misses: {final_stats['misses']}")
        print(f"  Hit rate: {final_stats['hit_rate_pct']:.1f}%")
        print()

    # Expectation: ~80-90% hit rate with 6 environments
    expected_hit_rate = 83.3  # 5 hits out of 6 requests
    if final_stats and final_stats['hit_rate_pct'] >= 80:
        print("✅ SUCCESS: Cache providing 80%+ hit rate!")
        print(f"   Achieved: {final_stats['hit_rate_pct']:.1f}% (Expected: ~{expected_hit_rate:.1f}%)")
    else:
        print("⚠️  Cache hit rate lower than expected")

    print("=" * 80)

if __name__ == "__main__":
    test_cache_value()

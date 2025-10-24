#!/usr/bin/env python3
"""
Cache Layer for Feature Pipeline
Provides in-memory caching with LRU eviction and optional Redis backend (Phase 2)
"""

import time
import logging
import hashlib
from typing import Any, Optional, Dict, Callable
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    def touch(self):
        """Update access timestamp and count"""
        self.last_accessed = time.time()
        self.access_count += 1


class InMemoryCache:
    """
    Thread-safe in-memory LRU cache with TTL support

    Features:
    - LRU eviction policy
    - TTL (time-to-live) support
    - Size-based eviction
    - Thread-safe operations
    - Cache statistics tracking
    """

    def __init__(
        self,
        max_size_mb: int = 1024,
        default_ttl_seconds: int = 300,
        max_entries: int = 10000
    ):
        """
        Initialize cache

        Args:
            max_size_mb: Maximum cache size in MB
            default_ttl_seconds: Default TTL for entries (300s = 5 minutes)
            max_entries: Maximum number of entries
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        self.max_entries = max_entries

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._current_size_bytes = 0

        logger.info(f"Initialized InMemoryCache: max_size={max_size_mb}MB, ttl={default_ttl_seconds}s")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self._misses += 1
                return None

            # Update access metadata
            entry.touch()

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            self._hits += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ):
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional TTL override
        """
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)

            # Check if value is too large
            if size_bytes > self.max_size_bytes:
                logger.warning(f"Value too large for cache: {size_bytes} bytes > {self.max_size_bytes} bytes")
                return

            # Remove old entry if exists
            if key in self._cache:
                self._remove_entry(key)

            # Evict until we have space
            while (
                self._current_size_bytes + size_bytes > self.max_size_bytes
                or len(self._cache) >= self.max_entries
            ):
                if not self._evict_lru():
                    break  # Can't evict anymore

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds or self.default_ttl_seconds
            )

            # Add to cache
            self._cache[key] = entry
            self._current_size_bytes += size_bytes

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            logger.info("Cache cleared")

    def _remove_entry(self, key: str):
        """Remove entry and update size"""
        entry = self._cache.pop(key, None)
        if entry:
            self._current_size_bytes -= entry.size_bytes

    def _evict_lru(self) -> bool:
        """
        Evict least recently used entry

        Returns:
            True if evicted, False if cache is empty
        """
        if not self._cache:
            return False

        # Get first item (least recently used)
        key = next(iter(self._cache))
        self._remove_entry(key)
        self._evictions += 1
        return True

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        if isinstance(value, pd.DataFrame):
            return value.memory_usage(deep=True).sum()
        elif isinstance(value, pd.Series):
            return value.memory_usage(deep=True)
        elif isinstance(value, (str, bytes)):
            return len(value)
        elif isinstance(value, (int, float)):
            return 8
        elif isinstance(value, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size(item) for item in value)
        else:
            # Fallback: rough estimate
            try:
                import sys
                return sys.getsizeof(value)
            except:
                return 1024  # Default 1KB

    def cleanup_expired(self):
        """Remove all expired entries"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                "entries": len(self._cache),
                "size_mb": self._current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization_pct": (self._current_size_bytes / self.max_size_bytes) * 100,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": hit_rate * 100,
                "evictions": self._evictions,
                "total_requests": total_requests
            }

    def get_entry_info(self, key: str) -> Optional[Dict]:
        """Get metadata about specific cache entry"""
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None

            return {
                "key": entry.key,
                "size_bytes": entry.size_bytes,
                "created_at": datetime.fromtimestamp(entry.created_at).isoformat(),
                "last_accessed": datetime.fromtimestamp(entry.last_accessed).isoformat(),
                "access_count": entry.access_count,
                "ttl_seconds": entry.ttl_seconds,
                "age_seconds": time.time() - entry.created_at,
                "is_expired": entry.is_expired()
            }


class CachedFeaturePipeline:
    """
    Wrapper around FeaturePipeline with caching

    Caches:
    - Feature DataFrames by (symbol, start_date, end_date) key
    - Individual feature vectors by timestamp
    """

    def __init__(
        self,
        feature_pipeline: Any,
        cache: Optional[InMemoryCache] = None,
        enable_cache: bool = True
    ):
        """
        Initialize cached feature pipeline

        Args:
            feature_pipeline: FeaturePipeline instance
            cache: Optional cache instance (creates default if None)
            enable_cache: Enable/disable caching
        """
        self.pipeline = feature_pipeline
        self.cache = cache or InMemoryCache()
        self.enable_cache = enable_cache

    def get_features(
        self,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get features with caching

        Args:
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            DataFrame with features
        """
        # Generate cache key
        cache_key = self._make_cache_key(
            self.pipeline.symbol,
            start_date,
            end_date,
            interval
        )

        # Try cache first
        if self.enable_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached

        # Cache miss - compute features
        logger.debug(f"Cache miss: {cache_key}")

        # Fetch and compute
        self.pipeline.fetch_data(start_date, end_date, interval, use_cache=True)
        features = self.pipeline.compute_features(validate=True)

        # Cache result
        if self.enable_cache:
            self.cache.set(cache_key, features, ttl_seconds=300)

        return features

    def get_features_at_time(self, timestamp: str) -> Optional[pd.Series]:
        """
        Get feature vector at specific timestamp with caching

        Args:
            timestamp: Target timestamp

        Returns:
            Feature vector or None
        """
        cache_key = f"features_{self.pipeline.symbol}_{timestamp}"

        # Try cache
        if self.enable_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Get from pipeline
        features = self.pipeline.get_features_at_time(timestamp)

        # Cache result
        if features is not None and self.enable_cache:
            self.cache.set(cache_key, features, ttl_seconds=300)

        return features

    @staticmethod
    def _make_cache_key(*args) -> str:
        """Generate cache key from arguments"""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache.get_stats()

    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()


if __name__ == "__main__":
    # Test cache
    cache = InMemoryCache(max_size_mb=100, default_ttl_seconds=60)

    # Add some test data
    cache.set("test1", "value1")
    cache.set("test2", {"data": [1, 2, 3]})

    # Create test DataFrame
    import pandas as pd
    df = pd.DataFrame({
        'a': range(1000),
        'b': range(1000, 2000)
    })
    cache.set("test_df", df)

    # Test retrieval
    print(f"test1: {cache.get('test1')}")
    print(f"test2: {cache.get('test2')}")
    print(f"test_df shape: {cache.get('test_df').shape}")

    # Print stats
    stats = cache.get_stats()
    print("\nCache Statistics:")
    for key, value in stats.items():
        if 'pct' in key:
            print(f"  {key}: {value:.2f}%")
        elif 'mb' in key:
            print(f"  {key}: {value:.2f} MB")
        else:
            print(f"  {key}: {value}")

    # Test expiration
    import time
    cache.set("short_lived", "data", ttl_seconds=1)
    print(f"\nshort_lived before expiration: {cache.get('short_lived')}")
    time.sleep(2)
    print(f"short_lived after expiration: {cache.get('short_lived')}")

    # Final stats
    print("\nFinal Statistics:")
    stats = cache.get_stats()
    print(f"Hit rate: {stats['hit_rate_pct']:.2f}%")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Cache utilization: {stats['utilization_pct']:.2f}%")

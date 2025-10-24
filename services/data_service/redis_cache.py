#!/usr/bin/env python3
"""
Redis Cache Layer for Data Service
Provides distributed caching with Redis backend and fallback to in-memory cache
"""

import json
import logging
import pickle
import time
from typing import Any, Optional, Dict
from datetime import datetime
import pandas as pd

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisError = Exception
    RedisConnectionError = Exception

from .cache_layer import InMemoryCache

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-backed cache with automatic fallback to in-memory cache

    Features:
    - Distributed caching across multiple service instances
    - Automatic serialization/deserialization of DataFrames
    - Connection pooling and automatic reconnection
    - Fallback to in-memory cache on Redis failure
    - TTL support with automatic expiration
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl_seconds: int = 300,
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        retry_on_timeout: bool = True,
        enable_fallback: bool = True,
        fallback_cache_size_mb: int = 512
    ):
        """
        Initialize Redis cache

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            default_ttl_seconds: Default TTL for cache entries
            max_connections: Maximum connection pool size
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            retry_on_timeout: Retry on timeout errors
            enable_fallback: Enable in-memory fallback on Redis failure
            fallback_cache_size_mb: Size of fallback cache in MB
        """
        self.default_ttl_seconds = default_ttl_seconds
        self.enable_fallback = enable_fallback
        self._connected = False
        self._redis_client: Optional['redis.Redis'] = None

        # Statistics
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._fallback_hits = 0

        # Initialize fallback cache
        if enable_fallback:
            self.fallback_cache = InMemoryCache(
                max_size_mb=fallback_cache_size_mb,
                default_ttl_seconds=default_ttl_seconds
            )
            logger.info(f"Initialized fallback cache ({fallback_cache_size_mb}MB)")
        else:
            self.fallback_cache = None

        # Initialize Redis connection
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available - using fallback cache only")
            self._connected = False
            return

        try:
            # Create connection pool
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                retry_on_timeout=retry_on_timeout,
                decode_responses=False  # We handle serialization manually
            )

            # Create Redis client
            self._redis_client = redis.Redis(connection_pool=pool)

            # Test connection
            self._redis_client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {host}:{port} (db={db})")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            if not enable_fallback:
                raise

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        # Try Redis first
        if self._connected and self._redis_client:
            try:
                data = self._redis_client.get(key)
                if data is not None:
                    self._hits += 1
                    return self._deserialize(data)
                else:
                    self._misses += 1
            except (RedisError, RedisConnectionError) as e:
                logger.warning(f"Redis get error: {e}")
                self._errors += 1
                self._connected = False  # Mark as disconnected
                # Fall through to fallback cache

        # Try fallback cache
        if self.enable_fallback and self.fallback_cache:
            value = self.fallback_cache.get(key)
            if value is not None:
                self._fallback_hits += 1
                return value

        self._misses += 1
        return None

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
        ttl = ttl_seconds or self.default_ttl_seconds

        # Try Redis first
        if self._connected and self._redis_client:
            try:
                serialized = self._serialize(value)
                self._redis_client.setex(key, ttl, serialized)
            except (RedisError, RedisConnectionError) as e:
                logger.warning(f"Redis set error: {e}")
                self._errors += 1
                self._connected = False
                # Fall through to fallback cache

        # Also set in fallback cache (write-through)
        if self.enable_fallback and self.fallback_cache:
            self.fallback_cache.set(key, value, ttl_seconds=ttl)

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        deleted = False

        # Delete from Redis
        if self._connected and self._redis_client:
            try:
                result = self._redis_client.delete(key)
                deleted = result > 0
            except (RedisError, RedisConnectionError) as e:
                logger.warning(f"Redis delete error: {e}")
                self._errors += 1

        # Delete from fallback cache
        if self.enable_fallback and self.fallback_cache:
            fallback_deleted = self.fallback_cache.delete(key)
            deleted = deleted or fallback_deleted

        return deleted

    def clear(self):
        """Clear all cache entries"""
        # Clear Redis
        if self._connected and self._redis_client:
            try:
                self._redis_client.flushdb()
                logger.info("Redis cache cleared")
            except (RedisError, RedisConnectionError) as e:
                logger.warning(f"Redis clear error: {e}")
                self._errors += 1

        # Clear fallback cache
        if self.enable_fallback and self.fallback_cache:
            self.fallback_cache.clear()

    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value for Redis storage

        Uses pickle for DataFrames and JSON for other types
        """
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            # Use pickle for pandas objects (most efficient)
            return pickle.dumps(value)
        else:
            # Use JSON for simple types
            try:
                return json.dumps(value).encode('utf-8')
            except (TypeError, ValueError):
                # Fallback to pickle for complex objects
                return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize value from Redis storage
        """
        try:
            # Try pickle first
            return pickle.loads(data)
        except (pickle.UnpicklingError, EOFError):
            # Try JSON
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.error("Failed to deserialize cached data")
                return None

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        stats = {
            "backend": "redis" if self._connected else "fallback",
            "connected": self._connected,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": hit_rate * 100,
            "errors": self._errors,
            "fallback_hits": self._fallback_hits,
            "total_requests": total_requests
        }

        # Add Redis-specific stats
        if self._connected and self._redis_client:
            try:
                info = self._redis_client.info('stats')
                stats.update({
                    "redis_total_connections": info.get('total_connections_received', 0),
                    "redis_total_commands": info.get('total_commands_processed', 0),
                    "redis_keyspace_hits": info.get('keyspace_hits', 0),
                    "redis_keyspace_misses": info.get('keyspace_misses', 0)
                })

                # Get memory info
                mem_info = self._redis_client.info('memory')
                stats.update({
                    "redis_used_memory_mb": mem_info.get('used_memory', 0) / (1024 * 1024),
                    "redis_used_memory_peak_mb": mem_info.get('used_memory_peak', 0) / (1024 * 1024)
                })

            except (RedisError, RedisConnectionError):
                pass

        # Add fallback cache stats
        if self.enable_fallback and self.fallback_cache:
            fallback_stats = self.fallback_cache.get_stats()
            stats["fallback_cache"] = fallback_stats

        return stats

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Redis connection

        Returns:
            Health status dict
        """
        if not self._redis_client:
            return {
                "healthy": False,
                "message": "Redis not initialized",
                "fallback_active": self.enable_fallback
            }

        try:
            start_time = time.time()
            self._redis_client.ping()
            latency_ms = (time.time() - start_time) * 1000

            self._connected = True

            return {
                "healthy": True,
                "latency_ms": latency_ms,
                "connected": True,
                "fallback_active": False
            }

        except (RedisError, RedisConnectionError) as e:
            self._connected = False
            return {
                "healthy": False,
                "message": str(e),
                "connected": False,
                "fallback_active": self.enable_fallback
            }

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to Redis

        Returns:
            True if reconnection successful
        """
        if not self._redis_client:
            return False

        try:
            self._redis_client.ping()
            self._connected = True
            logger.info("Reconnected to Redis")
            return True
        except (RedisError, RedisConnectionError) as e:
            logger.warning(f"Reconnection failed: {e}")
            return False

    def warm_cache(
        self,
        pipeline: Any,
        symbols: list,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Warm cache by pre-computing features for symbols

        Args:
            pipeline: FeaturePipeline instance
            symbols: List of symbols to warm
            start_date: Start date
            end_date: End date

        Returns:
            Warming statistics
        """
        logger.info(f"Warming cache for {len(symbols)} symbols...")

        warmed = 0
        errors = []
        start_time = time.time()

        for symbol in symbols:
            try:
                # Create temporary pipeline
                temp_pipeline = pipeline.__class__(
                    symbol=symbol,
                    validate_lookahead=False,  # Skip validation for speed
                    cache_dir=pipeline.cache_dir
                )

                # Fetch and compute features
                temp_pipeline.fetch_data(start_date, end_date, use_cache=True)
                features = temp_pipeline.compute_features(validate=False)

                # Cache the features
                cache_key = f"features_{symbol}_{start_date}_{end_date}"
                self.set(cache_key, features, ttl_seconds=self.default_ttl_seconds)

                warmed += 1
                logger.debug(f"Warmed cache for {symbol}")

            except Exception as e:
                error_msg = f"Failed to warm {symbol}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)

        duration = time.time() - start_time

        stats = {
            "symbols_requested": len(symbols),
            "symbols_warmed": warmed,
            "errors": len(errors),
            "error_messages": errors,
            "duration_seconds": duration
        }

        logger.info(f"Cache warming complete: {warmed}/{len(symbols)} symbols in {duration:.2f}s")
        return stats

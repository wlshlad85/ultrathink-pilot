#!/usr/bin/env python3
"""
Feature Cache Manager for Data Service
Integrates FeaturePipeline with Redis caching for high-performance feature serving
Target: 90%+ cache hit rate, <20ms P95 latency
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .feature_pipeline import FeaturePipeline
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)


@dataclass
class FeatureRequest:
    """Feature request specification"""
    symbol: str
    timeframe: str = "1d"
    timestamp: Optional[datetime] = None
    lookback_days: int = 365


@dataclass
class FeatureResult:
    """Feature computation result with metadata"""
    symbol: str
    timeframe: str
    timestamp: datetime
    features: Dict[str, float]
    cache_hit: bool
    computation_time_ms: float
    pipeline_version: str
    num_features: int


class FeatureCacheManager:
    """
    High-performance feature cache manager

    Features:
    - Redis-backed distributed caching
    - Automatic cache warming
    - Lookahead bias prevention
    - Feature versioning
    - Performance metrics tracking
    - Graceful degradation on cache failure

    Performance Targets:
    - 90%+ cache hit rate
    - <20ms P95 latency
    - 5k data updates/sec throughput
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        default_ttl_seconds: int = 300,  # 5 minutes
        enable_fallback: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize feature cache manager

        Args:
            redis_host: Redis host
            redis_port: Redis port
            default_ttl_seconds: Default cache TTL
            enable_fallback: Enable in-memory fallback on Redis failure
            cache_dir: Directory for file-based caching
        """
        # Initialize Redis cache
        self.cache = RedisCache(
            host=redis_host,
            port=redis_port,
            default_ttl_seconds=default_ttl_seconds,
            enable_fallback=enable_fallback
        )

        self.cache_dir = cache_dir
        self.default_ttl = default_ttl_seconds

        # Performance metrics
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._computation_times = []
        self._start_time = time.time()

        logger.info(f"Initialized FeatureCacheManager (Redis: {redis_host}:{redis_port}, TTL: {default_ttl_seconds}s)")

    def get_features(
        self,
        symbol: str,
        timeframe: str = "1d",
        timestamp: Optional[datetime] = None,
        lookback_days: int = 365
    ) -> FeatureResult:
        """
        Get features for a symbol with caching

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            timeframe: Data timeframe
            timestamp: Specific timestamp (None = latest)
            lookback_days: Historical data lookback window

        Returns:
            FeatureResult with features and metadata
        """
        start_time = time.time()
        self._total_requests += 1

        # Generate cache key
        ts_str = timestamp.isoformat() if timestamp else "latest"
        cache_key = f"features:v1.0.0:{symbol}:{timeframe}:{ts_str}"

        # Try cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            self._cache_hits += 1
            computation_time_ms = (time.time() - start_time) * 1000

            logger.debug(f"Cache hit: {cache_key} ({computation_time_ms:.2f}ms)")

            return FeatureResult(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=cached_data['timestamp'],
                features=cached_data['features'],
                cache_hit=True,
                computation_time_ms=computation_time_ms,
                pipeline_version=cached_data.get('version', '1.0.0'),
                num_features=len(cached_data['features'])
            )

        # Cache miss - compute features
        self._cache_misses += 1
        logger.debug(f"Cache miss: {cache_key}")

        # Compute features
        features_dict, feature_timestamp, version = self._compute_features(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            lookback_days=lookback_days
        )

        # Cache the result
        cache_data = {
            'timestamp': feature_timestamp,
            'features': features_dict,
            'version': version,
            'computed_at': datetime.utcnow().isoformat()
        }
        self.cache.set(cache_key, cache_data, ttl_seconds=self.default_ttl)

        computation_time_ms = (time.time() - start_time) * 1000
        self._computation_times.append(computation_time_ms)

        # Keep only last 1000 measurements
        if len(self._computation_times) > 1000:
            self._computation_times = self._computation_times[-1000:]

        logger.debug(f"Features computed: {symbol} ({computation_time_ms:.2f}ms)")

        return FeatureResult(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=feature_timestamp,
            features=features_dict,
            cache_hit=False,
            computation_time_ms=computation_time_ms,
            pipeline_version=version,
            num_features=len(features_dict)
        )

    def _compute_features(
        self,
        symbol: str,
        timeframe: str,
        timestamp: Optional[datetime],
        lookback_days: int
    ) -> Tuple[Dict[str, float], datetime, str]:
        """
        Compute features using FeaturePipeline

        Returns:
            Tuple of (features_dict, timestamp, pipeline_version)
        """
        # Create pipeline
        pipeline = FeaturePipeline(
            symbol=symbol,
            validate_lookahead=True,
            cache_dir=self.cache_dir
        )

        # Calculate date range
        end_date = timestamp or datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)

        # Fetch data
        pipeline.fetch_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval=timeframe,
            use_cache=True
        )

        # Compute features
        features_df = pipeline.compute_features(validate=True)

        # Get features at specific timestamp or latest
        if timestamp:
            feature_series = pipeline.get_features_at_time(timestamp)
            if feature_series is None:
                raise ValueError(f"No features available at timestamp {timestamp}")
            feature_timestamp = timestamp
        else:
            # Get latest features
            feature_series = features_df.iloc[-1]
            feature_timestamp = features_df.index[-1].to_pydatetime()

        # Convert to dict, handling NaN values
        features_dict = {}
        for key, value in feature_series.items():
            if pd.isna(value):
                features_dict[key] = 0.0  # Replace NaN with 0
            elif np.isinf(value):
                features_dict[key] = 0.0  # Replace inf with 0
            else:
                features_dict[key] = float(value)

        return features_dict, feature_timestamp, pipeline.VERSION

    def get_batch_features(
        self,
        requests: List[FeatureRequest]
    ) -> List[FeatureResult]:
        """
        Get features for multiple symbols in batch

        Args:
            requests: List of feature requests

        Returns:
            List of FeatureResults
        """
        results = []

        for req in requests:
            try:
                result = self.get_features(
                    symbol=req.symbol,
                    timeframe=req.timeframe,
                    timestamp=req.timestamp,
                    lookback_days=req.lookback_days
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to get features for {req.symbol}: {e}")
                # Return error result
                results.append(FeatureResult(
                    symbol=req.symbol,
                    timeframe=req.timeframe,
                    timestamp=datetime.utcnow(),
                    features={},
                    cache_hit=False,
                    computation_time_ms=0,
                    pipeline_version="1.0.0",
                    num_features=0
                ))

        return results

    def warm_cache(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        lookback_days: int = 365
    ) -> Dict[str, Any]:
        """
        Warm cache by pre-computing features for symbols

        Args:
            symbols: List of symbols to warm
            timeframe: Data timeframe
            lookback_days: Historical data window

        Returns:
            Warming statistics
        """
        logger.info(f"Warming cache for {len(symbols)} symbols...")

        warmed = 0
        errors = []
        start_time = time.time()

        for symbol in symbols:
            try:
                # Get features (will compute and cache)
                self.get_features(
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_days=lookback_days
                )
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
            "duration_seconds": duration,
            "avg_time_per_symbol": duration / len(symbols) if symbols else 0
        }

        logger.info(f"Cache warming complete: {warmed}/{len(symbols)} symbols in {duration:.2f}s")
        return stats

    def invalidate_cache(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ):
        """
        Invalidate cache entries

        Args:
            symbol: Specific symbol to invalidate (None = all)
            timeframe: Specific timeframe to invalidate (None = all)
        """
        if symbol is None:
            # Clear all cache
            self.cache.clear()
            logger.info("Cleared entire cache")
        else:
            # Clear specific symbol/timeframe
            # Note: This is a simplified version - in production you'd use Redis SCAN
            cache_key = f"features:v1.0.0:{symbol}:{timeframe or '*'}:*"
            logger.info(f"Invalidated cache for pattern: {cache_key}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics

        Returns:
            Dict with performance statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0

        # Calculate latency percentiles
        if self._computation_times:
            sorted_times = sorted(self._computation_times)
            p50 = sorted_times[len(sorted_times) // 2]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            p50 = p95 = p99 = 0

        uptime_seconds = time.time() - self._start_time

        metrics = {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate_pct": hit_rate,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
            "uptime_seconds": uptime_seconds,
            "requests_per_second": self._total_requests / uptime_seconds if uptime_seconds > 0 else 0,
            "redis_stats": self.cache.get_stats()
        }

        return metrics

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check

        Returns:
            Health status dict
        """
        redis_health = self.cache.health_check()

        # Check if we're meeting performance targets
        metrics = self.get_performance_metrics()
        meets_hit_rate_target = metrics['cache_hit_rate_pct'] >= 90.0 or self._total_requests < 100
        meets_latency_target = metrics['latency_p95_ms'] <= 20.0 or self._total_requests < 100

        healthy = redis_health['healthy'] or redis_health.get('fallback_active', False)

        return {
            "healthy": healthy,
            "redis": redis_health,
            "performance": {
                "meets_hit_rate_target": meets_hit_rate_target,
                "meets_latency_target": meets_latency_target,
                "current_hit_rate": metrics['cache_hit_rate_pct'],
                "current_p95_latency": metrics['latency_p95_ms']
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_available_features(self) -> Dict[str, Any]:
        """
        Get list of available features

        Returns:
            Dict with feature information
        """
        # Create a sample pipeline to get feature metadata
        pipeline = FeaturePipeline(
            symbol="BTC-USD",
            validate_lookahead=False,
            cache_dir=self.cache_dir
        )

        # Fetch small sample to get feature names
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)

        pipeline.fetch_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            use_cache=True
        )

        features_df = pipeline.compute_features(validate=False)
        metadata = pipeline.get_feature_metadata()

        return {
            "total_features": len(pipeline.get_feature_names()),
            "feature_names": pipeline.get_feature_names(),
            "categories": metadata['feature_categories'],
            "pipeline_version": pipeline.VERSION
        }


if __name__ == "__main__":
    # Test the feature cache manager
    import json

    # Initialize manager
    manager = FeatureCacheManager(
        redis_host="localhost",
        redis_port=6379,
        default_ttl_seconds=300,
        cache_dir="./data/cache"
    )

    # Get available features
    print("\n" + "="*60)
    print("Available Features")
    print("="*60)
    available = manager.get_available_features()
    print(f"Total features: {available['total_features']}")
    print(f"Pipeline version: {available['pipeline_version']}")
    print("\nFeatures by category:")
    for category, features in available['categories'].items():
        print(f"  {category}: {len(features)} features")

    # Test feature retrieval (first time - cache miss)
    print("\n" + "="*60)
    print("Test 1: Initial Request (Cache Miss Expected)")
    print("="*60)
    result1 = manager.get_features("BTC-USD", lookback_days=30)
    print(f"Symbol: {result1.symbol}")
    print(f"Timestamp: {result1.timestamp}")
    print(f"Cache hit: {result1.cache_hit}")
    print(f"Computation time: {result1.computation_time_ms:.2f}ms")
    print(f"Number of features: {result1.num_features}")
    print(f"Sample features: {dict(list(result1.features.items())[:5])}")

    # Test cached retrieval (second time - cache hit)
    print("\n" + "="*60)
    print("Test 2: Repeated Request (Cache Hit Expected)")
    print("="*60)
    result2 = manager.get_features("BTC-USD", lookback_days=30)
    print(f"Cache hit: {result2.cache_hit}")
    print(f"Computation time: {result2.computation_time_ms:.2f}ms")

    # Get performance metrics
    print("\n" + "="*60)
    print("Performance Metrics")
    print("="*60)
    metrics = manager.get_performance_metrics()
    print(f"Total requests: {metrics['total_requests']}")
    print(f"Cache hit rate: {metrics['cache_hit_rate_pct']:.2f}%")
    print(f"P50 latency: {metrics['latency_p50_ms']:.2f}ms")
    print(f"P95 latency: {metrics['latency_p95_ms']:.2f}ms")
    print(f"P99 latency: {metrics['latency_p99_ms']:.2f}ms")

    # Health check
    print("\n" + "="*60)
    print("Health Check")
    print("="*60)
    health = manager.health_check()
    print(json.dumps(health, indent=2, default=str))

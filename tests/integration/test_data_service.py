#!/usr/bin/env python3
"""
Integration Tests for Data Service (Feature Pipeline + Cache Layer)

Tests:
1. Feature pipeline initialization and data fetching
2. Feature computation and lookahead validation
3. Cache layer functionality (LRU, TTL)
4. CachedFeaturePipeline integration
5. Feature consistency (same features every time)
6. Performance benchmarks
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.data_service import FeaturePipeline, InMemoryCache, CachedFeaturePipeline


class TestFeaturePipeline:
    """Test FeaturePipeline functionality"""

    def test_initialization(self):
        """Test pipeline initialization"""
        pipeline = FeaturePipeline(
            symbol="BTC-USD",
            validate_lookahead=True
        )

        assert pipeline.symbol == "BTC-USD"
        assert pipeline.validate_lookahead is True
        assert pipeline.VERSION is not None

    def test_data_fetching(self):
        """Test data fetching from yfinance"""
        pipeline = FeaturePipeline(symbol="BTC-USD")

        df = pipeline.fetch_data(
            start_date="2023-01-01",
            end_date="2023-12-31",
            use_cache=False
        )

        assert df is not None
        assert len(df) > 0
        assert 'close' in df.columns
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'volume' in df.columns

    def test_feature_computation(self):
        """Test feature computation"""
        pipeline = FeaturePipeline(symbol="BTC-USD")

        pipeline.fetch_data("2023-01-01", "2023-12-31", use_cache=False)
        features = pipeline.compute_features(validate=True)

        assert features is not None
        assert len(features) > 0

        # Check for expected feature categories
        feature_names = features.columns.tolist()

        # Price features
        assert 'returns_1d' in feature_names
        assert 'returns_5d' in feature_names
        assert 'log_returns_1d' in feature_names

        # Volume features
        assert 'volume_sma_10' in feature_names or 'volume_ratio_10' in feature_names

        # Momentum indicators
        assert 'rsi_14' in feature_names
        assert 'macd' in feature_names

        # Trend indicators
        assert 'sma_20' in feature_names or 'ema_12' in feature_names

        # Volatility indicators
        assert 'atr_14' in feature_names or 'bb_20_width' in feature_names

    def test_lookahead_validation(self):
        """Test lookahead bias validation"""
        pipeline = FeaturePipeline(
            symbol="BTC-USD",
            validate_lookahead=True
        )

        pipeline.fetch_data("2023-01-01", "2023-12-31", use_cache=False)

        # Should not raise exception if no lookahead
        features = pipeline.compute_features(validate=True)
        assert features is not None

    def test_feature_consistency(self):
        """Test that features are consistent across multiple computations"""
        pipeline = FeaturePipeline(symbol="BTC-USD")

        # Compute features twice
        pipeline.fetch_data("2023-06-01", "2023-06-30", use_cache=False)
        features1 = pipeline.compute_features(validate=False)

        pipeline.fetch_data("2023-06-01", "2023-06-30", use_cache=False)
        features2 = pipeline.compute_features(validate=False)

        # Features should be identical
        pd.testing.assert_frame_equal(features1, features2)

    def test_get_features_at_time(self):
        """Test getting features at specific timestamp"""
        pipeline = FeaturePipeline(symbol="BTC-USD")

        pipeline.fetch_data("2023-06-01", "2023-06-30", use_cache=False)
        features = pipeline.compute_features(validate=False)

        # Get features at a specific time
        timestamp = features.index[10]
        features_at_time = pipeline.get_features_at_time(timestamp)

        assert features_at_time is not None
        assert len(features_at_time) == len(features.columns)

    def test_feature_metadata(self):
        """Test feature metadata generation"""
        pipeline = FeaturePipeline(symbol="BTC-USD")

        pipeline.fetch_data("2023-06-01", "2023-06-30", use_cache=False)
        pipeline.compute_features(validate=False)

        metadata = pipeline.get_feature_metadata()

        assert metadata is not None
        assert 'version' in metadata
        assert 'symbol' in metadata
        assert 'num_features' in metadata
        assert 'data_hash' in metadata
        assert 'date_range' in metadata

    def test_caching_with_disk(self):
        """Test disk caching functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FeaturePipeline(
                symbol="BTC-USD",
                cache_dir=tmpdir
            )

            # First fetch (cache miss)
            start_time = time.time()
            pipeline.fetch_data("2023-06-01", "2023-06-30", use_cache=True)
            first_fetch_time = time.time() - start_time

            # Second fetch (cache hit)
            start_time = time.time()
            pipeline.fetch_data("2023-06-01", "2023-06-30", use_cache=True)
            second_fetch_time = time.time() - start_time

            # Second fetch should be faster (cache hit)
            assert second_fetch_time < first_fetch_time


class TestInMemoryCache:
    """Test InMemoryCache functionality"""

    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = InMemoryCache(max_size_mb=100, default_ttl_seconds=60)

        assert cache.max_size_bytes == 100 * 1024 * 1024
        assert cache.default_ttl_seconds == 60

        stats = cache.get_stats()
        assert stats['entries'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0

    def test_cache_set_get(self):
        """Test basic cache set and get"""
        cache = InMemoryCache()

        cache.set("key1", "value1")
        value = cache.get("key1")

        assert value == "value1"

        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 0

    def test_cache_miss(self):
        """Test cache miss"""
        cache = InMemoryCache()

        value = cache.get("nonexistent")

        assert value is None

        stats = cache.get_stats()
        assert stats['misses'] == 1

    def test_cache_expiration(self):
        """Test TTL expiration"""
        cache = InMemoryCache(default_ttl_seconds=1)

        cache.set("key1", "value1", ttl_seconds=1)

        # Should exist immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(2)

        # Should be expired
        assert cache.get("key1") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction"""
        cache = InMemoryCache(max_size_mb=1, max_entries=3)

        # Add 3 entries (max)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"

        # Add 4th entry (should evict least recently used)
        cache.set("key4", "value4")

        # key2 should be evicted (least recently used)
        # key1 was accessed, so still in cache
        assert cache.get("key1") == "value1"
        assert cache.get("key4") == "value4"

        stats = cache.get_stats()
        assert stats['evictions'] > 0

    def test_cache_delete(self):
        """Test cache deletion"""
        cache = InMemoryCache()

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        deleted = cache.delete("key1")
        assert deleted is True

        assert cache.get("key1") is None

    def test_cache_clear(self):
        """Test cache clear"""
        cache = InMemoryCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

        stats = cache.get_stats()
        assert stats['entries'] == 0

    def test_cache_stats(self):
        """Test cache statistics"""
        cache = InMemoryCache()

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()

        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate_pct'] == 50.0
        assert stats['total_requests'] == 2


class TestCachedFeaturePipeline:
    """Test CachedFeaturePipeline integration"""

    def test_cached_pipeline_initialization(self):
        """Test cached pipeline initialization"""
        pipeline = FeaturePipeline(symbol="BTC-USD")
        cache = InMemoryCache()
        cached_pipeline = CachedFeaturePipeline(pipeline, cache, enable_cache=True)

        assert cached_pipeline.pipeline is not None
        assert cached_pipeline.cache is not None
        assert cached_pipeline.enable_cache is True

    def test_cached_pipeline_caching(self):
        """Test feature caching in cached pipeline"""
        pipeline = FeaturePipeline(symbol="BTC-USD")
        cache = InMemoryCache()
        cached_pipeline = CachedFeaturePipeline(pipeline, cache, enable_cache=True)

        # First call (cache miss)
        start_time = time.time()
        features1 = cached_pipeline.get_features("2023-06-01", "2023-06-30")
        first_time = time.time() - start_time

        # Second call (cache hit)
        start_time = time.time()
        features2 = cached_pipeline.get_features("2023-06-01", "2023-06-30")
        second_time = time.time() - start_time

        # Features should be identical
        pd.testing.assert_frame_equal(features1, features2)

        # Second call should be much faster (cache hit)
        assert second_time < first_time / 10  # At least 10x faster

        # Check cache stats
        stats = cached_pipeline.get_cache_stats()
        assert stats['hits'] >= 1
        assert stats['hit_rate_pct'] > 0

    def test_cached_pipeline_disabled(self):
        """Test cached pipeline with caching disabled"""
        pipeline = FeaturePipeline(symbol="BTC-USD")
        cache = InMemoryCache()
        cached_pipeline = CachedFeaturePipeline(pipeline, cache, enable_cache=False)

        features1 = cached_pipeline.get_features("2023-06-01", "2023-06-30")
        features2 = cached_pipeline.get_features("2023-06-01", "2023-06-30")

        # Cache should not be used
        stats = cached_pipeline.get_cache_stats()
        assert stats['hits'] == 0


class TestPerformance:
    """Performance benchmarks"""

    def test_feature_computation_speed(self):
        """Benchmark feature computation speed"""
        pipeline = FeaturePipeline(symbol="BTC-USD")

        # Fetch 1 year of data
        pipeline.fetch_data("2023-01-01", "2023-12-31", use_cache=False)

        # Time feature computation
        start_time = time.time()
        features = pipeline.compute_features(validate=True)
        compute_time = time.time() - start_time

        print(f"\nFeature computation time (1 year): {compute_time:.2f}s")
        print(f"Features computed: {len(features.columns)}")
        print(f"Data points: {len(features)}")

        # Should complete in reasonable time (<5 seconds for 1 year)
        assert compute_time < 5.0

    def test_cache_performance(self):
        """Benchmark cache performance"""
        cache = InMemoryCache(max_size_mb=100)

        # Create test data
        test_df = pd.DataFrame({
            'a': np.random.randn(1000),
            'b': np.random.randn(1000)
        })

        # Benchmark set
        start_time = time.time()
        for i in range(100):
            cache.set(f"key_{i}", test_df)
        set_time = time.time() - start_time

        # Benchmark get
        start_time = time.time()
        for i in range(100):
            cache.get(f"key_{i}")
        get_time = time.time() - start_time

        print(f"\nCache set time (100 DataFrames): {set_time:.4f}s")
        print(f"Cache get time (100 DataFrames): {get_time:.4f}s")
        print(f"Average get time: {(get_time/100)*1000:.2f}ms")

        # Cache hits should be very fast (<1ms average)
        assert (get_time / 100) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

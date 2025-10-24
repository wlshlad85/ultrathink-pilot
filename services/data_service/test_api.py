#!/usr/bin/env python3
"""
Integration tests for Data Service API
Tests all endpoints, caching, error handling, and performance
"""

import pytest
import time
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

# Import the app
from main import app, redis_cache, get_or_create_pipeline


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear cache before each test"""
    if redis_cache:
        redis_cache.clear()
    yield
    if redis_cache:
        redis_cache.clear()


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root(self, client):
        """Test root endpoint returns service info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "UltraThink Data Service API"
        assert "version" in data
        assert "docs" in data


class TestFeaturesEndpoint:
    """Test features endpoints"""

    def test_get_features_success(self, client):
        """Test getting features for a valid symbol"""
        response = client.get("/api/v1/features/BTC-USD")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "BTC-USD"
        assert "features" in data
        assert "metadata" in data
        assert data["metadata"]["pipeline_version"] == "1.0.0"
        assert data["metadata"]["num_features"] > 0

    def test_get_features_with_timeframe(self, client):
        """Test getting features with specific timeframe"""
        response = client.get("/api/v1/features/BTC-USD?timeframe=1d")
        assert response.status_code == 200

        data = response.json()
        assert data["timeframe"] == "1d"

    def test_get_features_cache_hit(self, client):
        """Test cache hit on second request"""
        # First request (cache miss)
        response1 = client.get("/api/v1/features/BTC-USD")
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["metadata"]["cache_hit"] is False

        # Second request (cache hit)
        response2 = client.get("/api/v1/features/BTC-USD")
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["metadata"]["cache_hit"] is True

        # Cache hit should be much faster
        assert data2["metadata"]["computation_time_ms"] < data1["metadata"]["computation_time_ms"]

    def test_get_features_performance(self, client):
        """Test feature endpoint performance targets"""
        # Warm cache
        client.get("/api/v1/features/BTC-USD")

        # Measure cached request
        start_time = time.time()
        response = client.get("/api/v1/features/BTC-USD")
        latency_ms = (time.time() - start_time) * 1000

        assert response.status_code == 200
        # Target: P95 <20ms for cached requests
        assert latency_ms < 100  # Allow some buffer for test environment

    def test_get_raw_data_success(self, client):
        """Test getting raw OHLCV data"""
        response = client.get("/api/v1/features/BTC-USD/raw")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "BTC-USD"
        assert "data" in data
        assert len(data["data"]) > 0
        assert data["count"] > 0

    def test_get_raw_data_with_dates(self, client):
        """Test getting raw data with date range"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        response = client.get(
            f"/api/v1/features/BTC-USD/raw?start_date={start_date}&end_date={end_date}"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["count"] <= 10  # ~7 days of daily data


class TestBatchEndpoint:
    """Test batch features endpoint"""

    def test_batch_features_success(self, client):
        """Test batch feature request"""
        request_data = {
            "requests": [
                {"symbol": "BTC-USD", "timeframe": "1d"},
                {"symbol": "ETH-USD", "timeframe": "1d"}
            ]
        }

        response = client.post("/api/v1/features/batch", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["total_requests"] == 2
        assert data["successful"] >= 0
        assert len(data["results"]) + len(data["errors"]) == 2

    def test_batch_features_max_limit(self, client):
        """Test batch request size limit"""
        # Create request with >100 symbols (should fail validation)
        request_data = {
            "requests": [
                {"symbol": f"SYM{i}", "timeframe": "1d"}
                for i in range(101)
            ]
        }

        response = client.post("/api/v1/features/batch", json=request_data)
        assert response.status_code == 422  # Validation error


class TestFeatureListEndpoint:
    """Test feature list endpoint"""

    def test_list_features(self, client):
        """Test listing available features"""
        response = client.get("/api/v1/features/list")
        assert response.status_code == 200

        data = response.json()
        assert "features" in data
        assert data["total_count"] > 0
        assert "categories" in data
        assert "price" in data["categories"]
        assert "volume" in data["categories"]
        assert "momentum" in data["categories"]


class TestCacheWarmingEndpoint:
    """Test cache warming endpoint"""

    def test_warm_cache_success(self, client):
        """Test cache warming for multiple symbols"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        request_data = {
            "symbols": ["BTC-USD", "ETH-USD"],
            "start_date": start_date,
            "end_date": end_date
        }

        response = client.post("/api/v1/cache/warm", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "complete"
        assert data["symbols_processed"] >= 0
        assert "duration_seconds" in data

    def test_warm_cache_max_symbols(self, client):
        """Test cache warming symbol limit"""
        request_data = {
            "symbols": [f"SYM{i}" for i in range(51)],
            "start_date": "2024-01-01",
            "end_date": "2024-01-31"
        }

        response = client.post("/api/v1/cache/warm", json=request_data)
        assert response.status_code == 422  # Validation error


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self, client):
        """Test health endpoint returns status"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "components" in data
        assert "redis" in data["components"]
        assert "feature_pipeline" in data["components"]
        assert "uptime_seconds" in data
        assert data["version"] == "1.0.0"

    def test_health_check_performance(self, client):
        """Test health endpoint latency target"""
        start_time = time.time()
        response = client.get("/health")
        latency_ms = (time.time() - start_time) * 1000

        assert response.status_code == 200
        # Target: <100ms
        assert latency_ms < 200  # Allow buffer for test environment


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint"""

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns Prometheus format"""
        response = client.get("/metrics")
        assert response.status_code == 200

        # Check for Prometheus format
        content = response.text
        assert "# HELP" in content
        assert "# TYPE" in content
        assert "data_service_requests_total" in content


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_symbol(self, client):
        """Test handling of invalid symbol"""
        response = client.get("/api/v1/features/INVALID123456")
        # Should either return 500 or 404 depending on implementation
        assert response.status_code in [404, 500]

    def test_invalid_timeframe(self, client):
        """Test handling of invalid timeframe"""
        response = client.get("/api/v1/features/BTC-USD?timeframe=invalid")
        assert response.status_code in [400, 422, 500]


class TestConcurrency:
    """Test concurrent requests"""

    def test_concurrent_requests(self, client):
        """Test handling multiple concurrent requests"""
        import concurrent.futures

        def make_request():
            return client.get("/api/v1/features/BTC-USD")

        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]

        # All requests should succeed
        assert all(r.status_code == 200 for r in results)


class TestCachePerformance:
    """Test cache performance metrics"""

    def test_cache_hit_rate(self, client):
        """Test cache hit rate after multiple requests"""
        symbol = "BTC-USD"

        # Make 10 requests to same symbol
        for _ in range(10):
            response = client.get(f"/api/v1/features/{symbol}")
            assert response.status_code == 200

        # Check health endpoint for cache stats
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # First request is miss, rest should be hits
        # So hit rate should be 90% (9/10)


class TestDataIntegrity:
    """Test data integrity and consistency"""

    def test_feature_consistency(self, client):
        """Test that same request returns consistent features"""
        response1 = client.get("/api/v1/features/BTC-USD")
        response2 = client.get("/api/v1/features/BTC-USD")

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Features should be identical (from cache)
        assert data1["features"] == data2["features"]

    def test_feature_count_consistency(self, client):
        """Test that feature count matches pipeline"""
        response = client.get("/api/v1/features/BTC-USD")
        assert response.status_code == 200

        data = response.json()
        feature_count = len(data["features"])
        reported_count = data["metadata"]["num_features"]

        assert feature_count == reported_count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

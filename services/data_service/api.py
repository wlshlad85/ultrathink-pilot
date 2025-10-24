#!/usr/bin/env python3
"""
Data Service Production API
Unified feature engineering with Redis caching
Target: <20ms P95 latency, 90%+ cache hit rate
"""

import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .feature_cache_manager import FeatureCacheManager, FeatureRequest as FCMFeatureRequest
from .models import (
    FeatureResponse,
    BatchFeatureRequest,
    BatchFeatureResponse,
    HealthResponse,
    HealthStatus,
    ComponentHealth,
    MetricsResponse,
    CacheStats,
    FeatureListResponse,
    WarmCacheRequest,
    WarmCacheResponse,
    ErrorResponse,
    FeatureMetadata,
    TimeFrame
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global feature cache manager instance
feature_manager: Optional[FeatureCacheManager] = None
app_start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global feature_manager, app_start_time

    # Startup
    logger.info("Starting Data Service API v1.0.0")
    app_start_time = time.time()

    # Initialize feature cache manager
    feature_manager = FeatureCacheManager(
        redis_host="redis",  # Docker service name
        redis_port=6379,
        default_ttl_seconds=300,  # 5 minutes
        enable_fallback=True,
        cache_dir="/app/data/cache"
    )

    logger.info("Feature cache manager initialized")

    # Warm cache for common symbols (async background task)
    logger.info("Warming cache for common symbols...")
    try:
        common_symbols = ["BTC-USD", "ETH-USD"]
        stats = feature_manager.warm_cache(common_symbols, lookback_days=30)
        logger.info(f"Cache warming complete: {stats}")
    except Exception as e:
        logger.warning(f"Cache warming failed (non-critical): {e}")

    yield

    # Shutdown
    logger.info("Shutting down Data Service API")


# Create FastAPI app
app = FastAPI(
    title="UltraThink Data Service",
    description="Unified feature engineering with Redis caching",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns overall service health and component status
    """
    if feature_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    health = feature_manager.health_check()

    # Determine overall status
    if health['healthy']:
        status = HealthStatus.HEALTHY
    elif health['redis'].get('fallback_active'):
        status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.UNHEALTHY

    # Build component health
    components = {
        "redis": ComponentHealth(
            status=HealthStatus.HEALTHY if health['redis']['healthy'] else HealthStatus.DEGRADED,
            message=health['redis'].get('message'),
            latency_ms=health['redis'].get('latency_ms')
        ),
        "feature_pipeline": ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="Operational"
        ),
        "cache_performance": ComponentHealth(
            status=HealthStatus.HEALTHY if health['performance']['meets_hit_rate_target'] else HealthStatus.DEGRADED,
            message=f"Hit rate: {health['performance']['current_hit_rate']:.1f}%, P95: {health['performance']['current_p95_latency']:.1f}ms"
        )
    }

    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow(),
        uptime_seconds=time.time() - app_start_time,
        components=components,
        version="1.0.0"
    )


@app.get(
    "/api/v1/features/{symbol}",
    response_model=FeatureResponse,
    tags=["Features"],
    summary="Get features for a symbol"
)
async def get_features(
    symbol: str = Path(..., description="Trading symbol (e.g., BTC-USD)"),
    timeframe: TimeFrame = Query(TimeFrame.ONE_DAY, description="Data timeframe"),
    timestamp: Optional[datetime] = Query(None, description="Specific timestamp (defaults to latest)"),
    lookback_days: int = Query(365, ge=30, le=730, description="Historical data window (30-730 days)")
):
    """
    Get preprocessed features for a trading symbol

    Features are cached for 5 minutes by default.
    Returns 60+ technical indicators across multiple categories.

    Performance: <20ms P95 latency (target)
    """
    if feature_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        start_time = time.time()

        # Get features
        result = feature_manager.get_features(
            symbol=symbol.upper(),
            timeframe=timeframe.value,
            timestamp=timestamp,
            lookback_days=lookback_days
        )

        # Build response
        return FeatureResponse(
            symbol=result.symbol,
            timeframe=result.timeframe,
            timestamp=result.timestamp,
            features=result.features,
            metadata=FeatureMetadata(
                cache_hit=result.cache_hit,
                pipeline_version=result.pipeline_version,
                generated_at=datetime.utcnow(),
                computation_time_ms=result.computation_time_ms,
                num_features=result.num_features
            )
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting features for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post(
    "/api/v1/features/batch",
    response_model=BatchFeatureResponse,
    tags=["Features"],
    summary="Get features for multiple symbols"
)
async def get_batch_features(request: BatchFeatureRequest):
    """
    Get features for multiple symbols in a single request

    Maximum 100 requests per batch.
    Each request is processed independently with individual caching.
    """
    if feature_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert requests
        fcm_requests = [
            FCMFeatureRequest(
                symbol=req.symbol,
                timeframe=req.timeframe.value,
                timestamp=req.timestamp
            )
            for req in request.requests
        ]

        # Get batch features
        results = feature_manager.get_batch_features(fcm_requests)

        # Build responses
        responses = []
        errors = []
        successful = 0

        for i, result in enumerate(results):
            if result.num_features > 0:
                responses.append(FeatureResponse(
                    symbol=result.symbol,
                    timeframe=result.timeframe,
                    timestamp=result.timestamp,
                    features=result.features,
                    metadata=FeatureMetadata(
                        cache_hit=result.cache_hit,
                        pipeline_version=result.pipeline_version,
                        generated_at=datetime.utcnow(),
                        computation_time_ms=result.computation_time_ms,
                        num_features=result.num_features
                    )
                ))
                successful += 1
            else:
                errors.append({
                    "index": i,
                    "symbol": result.symbol,
                    "error": "Failed to compute features"
                })

        return BatchFeatureResponse(
            results=responses,
            total_requests=len(request.requests),
            successful=successful,
            failed=len(errors),
            errors=errors
        )

    except Exception as e:
        logger.error(f"Error processing batch request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/api/v1/features",
    response_model=FeatureListResponse,
    tags=["Features"],
    summary="List available features"
)
async def list_features():
    """
    Get list of all available features

    Returns feature names grouped by category (price, volume, momentum, etc.)
    """
    if feature_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        available = feature_manager.get_available_features()

        return FeatureListResponse(
            features=available['feature_names'],
            total_count=available['total_features'],
            pipeline_version=available['pipeline_version'],
            categories=available['categories']
        )

    except Exception as e:
        logger.error(f"Error listing features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Get service performance metrics

    Returns:
    - Request counts and rates
    - Latency percentiles (P50, P95, P99)
    - Cache statistics (hit rate, size, etc.)
    - Error rates
    """
    if feature_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        perf_metrics = feature_manager.get_performance_metrics()
        redis_stats = perf_metrics['redis_stats']

        # Build cache stats
        cache_stats = CacheStats(
            entries=redis_stats.get('fallback_cache', {}).get('entries', 0),
            size_mb=redis_stats.get('fallback_cache', {}).get('size_mb', 0),
            max_size_mb=redis_stats.get('fallback_cache', {}).get('max_size_mb', 512),
            utilization_pct=redis_stats.get('fallback_cache', {}).get('utilization_pct', 0),
            hits=perf_metrics['cache_hits'],
            misses=perf_metrics['cache_misses'],
            hit_rate_pct=perf_metrics['cache_hit_rate_pct'],
            evictions=redis_stats.get('fallback_cache', {}).get('evictions', 0),
            total_requests=perf_metrics['total_requests']
        )

        return MetricsResponse(
            total_requests=perf_metrics['total_requests'],
            requests_per_second=perf_metrics['requests_per_second'],
            latency_p50=perf_metrics['latency_p50_ms'],
            latency_p95=perf_metrics['latency_p95_ms'],
            latency_p99=perf_metrics['latency_p99_ms'],
            cache=cache_stats,
            features_computed=perf_metrics['cache_misses'],
            features_served_from_cache=perf_metrics['cache_hits'],
            total_errors=0,  # TODO: Implement error tracking
            error_rate_pct=0.0
        )

    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post(
    "/api/v1/cache/warm",
    response_model=WarmCacheResponse,
    tags=["Cache Management"]
)
async def warm_cache(request: WarmCacheRequest, background_tasks: BackgroundTasks):
    """
    Warm cache by pre-computing features for symbols

    This is useful for:
    - Reducing cold start latency
    - Ensuring cache availability for critical symbols
    - Pre-loading features before market hours

    Processing happens asynchronously in the background.
    """
    if feature_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        start_time = time.time()

        # Warm cache for each timeframe
        total_warmed = 0
        total_errors = []

        for timeframe in request.timeframes:
            stats = feature_manager.warm_cache(
                symbols=request.symbols,
                timeframe=timeframe.value,
                lookback_days=30  # Use shorter window for warming
            )
            total_warmed += stats['symbols_warmed']
            total_errors.extend(stats['error_messages'])

        duration = time.time() - start_time

        return WarmCacheResponse(
            status="completed",
            symbols_processed=len(request.symbols) * len(request.timeframes),
            features_cached=total_warmed,
            duration_seconds=duration,
            errors=total_errors
        )

    except Exception as e:
        logger.error(f"Error warming cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete(
    "/api/v1/cache/{symbol}",
    tags=["Cache Management"],
    summary="Invalidate cache for symbol"
)
async def invalidate_cache(
    symbol: str = Path(..., description="Symbol to invalidate (or 'all' for complete cache clear)"),
    timeframe: Optional[TimeFrame] = Query(None, description="Specific timeframe to invalidate")
):
    """
    Invalidate cached features for a symbol

    Use this when:
    - Data has been corrected/updated
    - Feature pipeline version changed
    - Manual cache refresh needed
    """
    if feature_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        if symbol.upper() == "ALL":
            feature_manager.invalidate_cache()
            return {"status": "success", "message": "All cache cleared"}
        else:
            tf_value = timeframe.value if timeframe else None
            feature_manager.invalidate_cache(symbol=symbol.upper(), timeframe=tf_value)
            return {"status": "success", "message": f"Cache cleared for {symbol}"}

    except Exception as e:
        logger.error(f"Error invalidating cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="NotFound",
            message="Endpoint not found",
            timestamp=datetime.utcnow()
        ).dict()
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            timestamp=datetime.utcnow()
        ).dict()
    )


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

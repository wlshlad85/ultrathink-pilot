#!/usr/bin/env python3
"""
Pydantic data models for Data Service API
Provides request/response schemas for all endpoints
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class TimeFrame(str, Enum):
    """Supported timeframes for market data"""
    ONE_MIN = "1min"
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"


class FeatureRequest(BaseModel):
    """Request model for feature endpoint"""
    symbol: str = Field(..., description="Trading symbol (e.g., BTC-USD)")
    timeframe: TimeFrame = Field(default=TimeFrame.ONE_DAY, description="Time resolution")
    timestamp: Optional[datetime] = Field(None, description="Specific timestamp (defaults to latest)")
    version: Optional[str] = Field(None, description="Feature pipeline version")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Validate symbol format"""
        if not v or len(v) < 2:
            raise ValueError("Symbol must be at least 2 characters")
        return v.upper()


class BatchFeatureRequest(BaseModel):
    """Request model for batch feature requests"""
    requests: List[FeatureRequest] = Field(..., description="List of feature requests")

    @field_validator('requests')
    @classmethod
    def validate_requests(cls, v):
        """Validate batch size"""
        if len(v) > 100:
            raise ValueError("Maximum 100 requests per batch")
        if len(v) == 0:
            raise ValueError("At least 1 request required")
        return v


class FeatureMetadata(BaseModel):
    """Metadata about feature computation"""
    cache_hit: bool = Field(..., description="Whether features were served from cache")
    pipeline_version: str = Field(..., description="Feature pipeline version")
    generated_at: datetime = Field(..., description="When features were generated")
    computation_time_ms: float = Field(..., description="Time to compute features (ms)")
    num_features: int = Field(..., description="Number of features returned")


class FeatureResponse(BaseModel):
    """Response model for feature endpoint"""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Time resolution")
    timestamp: datetime = Field(..., description="Timestamp of features")
    features: Dict[str, float] = Field(..., description="Feature values")
    metadata: FeatureMetadata = Field(..., description="Computation metadata")


class BatchFeatureResponse(BaseModel):
    """Response model for batch feature requests"""
    results: List[FeatureResponse] = Field(..., description="Feature responses")
    total_requests: int = Field(..., description="Total number of requests")
    successful: int = Field(..., description="Number of successful requests")
    failed: int = Field(..., description="Number of failed requests")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error details")


class RawDataResponse(BaseModel):
    """Response model for raw market data"""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Time resolution")
    data: List[Dict[str, Any]] = Field(..., description="OHLCV data points")
    count: int = Field(..., description="Number of data points")
    start_date: datetime = Field(..., description="First timestamp")
    end_date: datetime = Field(..., description="Last timestamp")


class CacheStats(BaseModel):
    """Cache statistics"""
    entries: int = Field(..., description="Number of cached entries")
    size_mb: float = Field(..., description="Cache size in MB")
    max_size_mb: float = Field(..., description="Maximum cache size in MB")
    utilization_pct: float = Field(..., description="Cache utilization percentage")
    hits: int = Field(..., description="Cache hit count")
    misses: int = Field(..., description="Cache miss count")
    hit_rate_pct: float = Field(..., description="Cache hit rate percentage")
    evictions: int = Field(..., description="Number of evictions")
    total_requests: int = Field(..., description="Total cache requests")


class HealthStatus(str, Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a service component"""
    status: HealthStatus = Field(..., description="Component health status")
    message: Optional[str] = Field(None, description="Status message")
    latency_ms: Optional[float] = Field(None, description="Component latency")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: HealthStatus = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    components: Dict[str, ComponentHealth] = Field(..., description="Component health details")
    version: str = Field(..., description="Service version")


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint"""
    # Request metrics
    total_requests: int = Field(..., description="Total API requests")
    requests_per_second: float = Field(..., description="Current requests per second")

    # Latency metrics (in milliseconds)
    latency_p50: float = Field(..., description="50th percentile latency")
    latency_p95: float = Field(..., description="95th percentile latency")
    latency_p99: float = Field(..., description="99th percentile latency")

    # Cache metrics
    cache: CacheStats = Field(..., description="Cache statistics")

    # Feature computation metrics
    features_computed: int = Field(..., description="Total features computed")
    features_served_from_cache: int = Field(..., description="Features served from cache")

    # Error metrics
    total_errors: int = Field(..., description="Total error count")
    error_rate_pct: float = Field(..., description="Error rate percentage")


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class FeatureListResponse(BaseModel):
    """Response model for feature list endpoint"""
    features: List[str] = Field(..., description="List of available features")
    total_count: int = Field(..., description="Total number of features")
    pipeline_version: str = Field(..., description="Feature pipeline version")
    categories: Dict[str, List[str]] = Field(..., description="Features grouped by category")


class WarmCacheRequest(BaseModel):
    """Request model for cache warming"""
    symbols: List[str] = Field(..., description="Symbols to warm cache for")
    timeframes: List[TimeFrame] = Field(default=[TimeFrame.ONE_DAY], description="Timeframes to warm")
    start_date: str = Field(..., description="Start date for data (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for data (YYYY-MM-DD)")

    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        """Validate symbol list"""
        if len(v) > 50:
            raise ValueError("Maximum 50 symbols for cache warming")
        if len(v) == 0:
            raise ValueError("At least 1 symbol required")
        return [s.upper() for s in v]


class WarmCacheResponse(BaseModel):
    """Response model for cache warming"""
    status: str = Field(..., description="Warming status")
    symbols_processed: int = Field(..., description="Number of symbols processed")
    features_cached: int = Field(..., description="Number of feature sets cached")
    duration_seconds: float = Field(..., description="Total warming duration")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")

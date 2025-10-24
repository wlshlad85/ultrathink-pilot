# Data Service - Unified Feature Pipeline

The Data Service provides a unified, consistent feature engineering pipeline for all training and inference operations in UltraThink Pilot.

## Features

- **Unified Feature Engineering**: Single source of truth for all technical indicators
- **Lookahead Prevention**: Built-in validation to prevent data leakage
- **In-Memory Caching**: LRU cache with TTL for fast repeated access
- **Feature Versioning**: Track feature pipeline versions for reproducibility
- **Repository Pattern**: Abstract data access from business logic

## Quick Start

### Basic Usage

```python
from services.data_service import FeaturePipeline, CachedFeaturePipeline, InMemoryCache

# Initialize pipeline
pipeline = FeaturePipeline(
    symbol="BTC-USD",
    validate_lookahead=True,
    cache_dir="./data/cache"
)

# Fetch and compute features
df_raw = pipeline.fetch_data(
    start_date="2023-01-01",
    end_date="2024-01-01"
)

df_features = pipeline.compute_features(validate=True)

# Access feature names
feature_names = pipeline.get_feature_names()
print(f"Total features: {len(feature_names)}")
```

### With Caching

```python
# Initialize cache
cache = InMemoryCache(max_size_mb=1024, default_ttl_seconds=300)

# Wrap pipeline with cache
cached_pipeline = CachedFeaturePipeline(
    feature_pipeline=pipeline,
    cache=cache,
    enable_cache=True
)

# Get features (will use cache on subsequent calls)
features = cached_pipeline.get_features(
    start_date="2023-01-01",
    end_date="2024-01-01"
)

# Check cache performance
stats = cached_pipeline.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate_pct']:.2f}%")
```

## Feature Categories

### Price Features (12 features)
- Returns at multiple horizons (1d, 2d, 5d, 10d, 20d)
- Log returns
- Price range position
- Candle patterns (body, wicks)

### Volume Features (7 features)
- Volume moving averages (10d, 20d)
- Volume ratios
- Volume momentum
- Price-volume correlation

### Momentum Indicators (11 features)
- RSI (14, 28 period)
- MACD (line, signal, histogram)
- Stochastic Oscillator (K, D)
- Rate of Change

### Trend Indicators (20 features)
- Simple Moving Averages (10, 20, 50, 100, 200)
- Exponential Moving Averages (8, 12, 26)
- MA distance metrics
- MA crossover signals
- Trend strength

### Volatility Indicators (12 features)
- Average True Range (14, 28 period)
- Bollinger Bands (width, position)
- Historical volatility (10d, 20d, 30d)

### Statistical Features (6 features)
- Z-scores (20, 50 period)
- Returns skewness and kurtosis
- Autocorrelation

**Total: ~70+ features** (exact count depends on configuration)

## Lookahead Prevention

The pipeline includes built-in validation to prevent lookahead bias:

1. **Shift-based calculations**: All features use `.shift()`, `.rolling()`, or `.ewm()` with proper parameters
2. **NaN validation**: Checks for unexpected NaN patterns indicating future data usage
3. **Spot-check validation**: Verifies feature values at time T don't change when future data is added

## Caching

### In-Memory Cache

- **LRU Eviction**: Least recently used entries evicted when full
- **TTL Support**: Entries expire after configurable time (default 5 minutes)
- **Size-based Limits**: Maximum cache size in MB (default 1024 MB)
- **Thread-safe**: Safe for concurrent access
- **Statistics Tracking**: Hit rate, evictions, utilization metrics

### Cache Statistics

```python
stats = cache.get_stats()
# {
#     "entries": 42,
#     "size_mb": 245.67,
#     "max_size_mb": 1024.0,
#     "utilization_pct": 24.0,
#     "hits": 1523,
#     "misses": 87,
#     "hit_rate_pct": 94.6,
#     "evictions": 12,
#     "total_requests": 1610
# }
```

## Integration with Training

### Training Script Example

```python
from services.data_service import FeaturePipeline

# Initialize pipeline
pipeline = FeaturePipeline(
    symbol="BTC-USD",
    validate_lookahead=True,
    cache_dir="./data/cache"
)

# Fetch training data
pipeline.fetch_data("2020-01-01", "2024-01-01")
features = pipeline.compute_features(validate=True)

# Get metadata for experiment tracking
metadata = pipeline.get_feature_metadata()
# Log to MLflow/TimescaleDB
experiment_tracker.log_metadata(metadata)

# Use features in training environment
# See rl/trading_env.py for integration example
```

## Migration from Legacy

### Before (Legacy)
```python
# Multiple data fetchers, inconsistent features
from backtesting.data_fetcher import DataFetcher

fetcher = DataFetcher("BTC-USD")
df = fetcher.fetch("2023-01-01", "2024-01-01")
df = fetcher.add_technical_indicators()  # Limited set of indicators
```

### After (Unified Pipeline)
```python
# Single feature pipeline, comprehensive features
from services.data_service import FeaturePipeline

pipeline = FeaturePipeline("BTC-USD")
pipeline.fetch_data("2023-01-01", "2024-01-01")
df = pipeline.compute_features(validate=True)  # 70+ features, validated
```

## Performance

Based on 1 year of BTC-USD daily data:

- **Fetch time**: ~2-3 seconds (from yfinance)
- **Compute time**: ~0.5-1 second (all features)
- **Cache hit**: <1ms (in-memory)
- **Cache miss + compute**: ~1-2 seconds

## Validation

### Lookahead Validation

The pipeline automatically validates features on `compute_features(validate=True)`:

```
INFO:root:Validating lookahead prevention...
INFO:root:✓ Lookahead validation passed (spot check)
INFO:root:✓ Lookahead prevention validation complete
```

### Manual Validation

```python
# Get feature at specific time
features_t = pipeline.get_features_at_time("2023-06-15")

# Add more data
pipeline.fetch_data("2023-01-01", "2024-06-01")
pipeline.compute_features()

# Get same feature again - should not change!
features_t_after = pipeline.get_features_at_time("2023-06-15")

assert features_t.equals(features_t_after), "Lookahead detected!"
```

## Future Enhancements (Phase 2)

- **Redis Backend**: Distributed caching for multi-process training
- **Feature Store**: Pre-computed feature database
- **Real-time Features**: WebSocket integration for live data
- **Feature Selection**: Automatic feature importance ranking
- **Custom Indicators**: Plugin system for user-defined features

## Troubleshooting

### High Cache Miss Rate

```python
# Check cache stats
stats = cache.get_stats()
if stats['hit_rate_pct'] < 50:
    # Increase cache size
    cache = InMemoryCache(max_size_mb=2048)
    # Or increase TTL
    cache = InMemoryCache(default_ttl_seconds=600)
```

### Memory Issues

```python
# Reduce cache size
cache = InMemoryCache(max_size_mb=512)

# Or disable caching for large datasets
pipeline = FeaturePipeline(..., cache_dir=None)
cached_pipeline = CachedFeaturePipeline(pipeline, enable_cache=False)
```

### Feature Computation Slow

```python
# Use disk cache for repeated access
pipeline = FeaturePipeline(cache_dir="./data/cache")
pipeline.fetch_data(..., use_cache=True)  # Uses disk cache

# Or save/load pre-computed features
pipeline.save_features("features.csv")
pipeline.load_features("features.csv")  # Fast loading
```

## Testing

Run tests:
```bash
pytest tests/integration/test_data_service.py -v
```

## Version History

- **1.0.0** (2025-10-21): Initial release
  - Unified feature pipeline
  - In-memory caching with LRU
  - Lookahead validation
  - 70+ technical indicators

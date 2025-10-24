# A/B Testing Framework for Safe Model Rollouts

## Overview

The A/B testing framework enables safe, data-driven model deployments in the UltraThink trading system. It supports two primary modes:

1. **Traffic Splitting** - Route X% of traffic to a new model (canary deployments)
2. **Shadow Mode** - Run both models in parallel, compare predictions without affecting production

## Architecture

### Components

```
┌─────────────────────────────────────────────────────┐
│            Inference API (FastAPI)                  │
├─────────────────────────────────────────────────────┤
│  /api/v1/predict  (with A/B test routing)          │
│  /api/v1/ab-test/* (A/B test management endpoints) │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│         ABTestingManager                            │
├─────────────────────────────────────────────────────┤
│  - Traffic splitting (consistent hashing)           │
│  - Shadow mode execution                            │
│  - Metrics collection                               │
│  - Result buffering                                 │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│        ABTestStorageBackend                         │
├─────────────────────────────────────────────────────┤
│  - Batch inserts to TimescaleDB                     │
│  - Query methods for analysis                       │
│  - Aggregated metrics                               │
└─────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│           TimescaleDB                               │
├─────────────────────────────────────────────────────┤
│  ab_test_configs      - Test configurations         │
│  ab_test_results      - Prediction comparisons      │
│  ab_test_metrics_*    - Aggregated metrics          │
└─────────────────────────────────────────────────────┘
```

### Files

- **`ab_testing_manager.py`** - Core A/B testing logic
- **`ab_storage.py`** - TimescaleDB storage backend
- **`ab_api_integration.py`** - API endpoints for A/B test management
- **`infrastructure/ab_testing_schema.sql`** - Database schema
- **`tests/test_ab_testing.py`** - Comprehensive test suite

## Quick Start

### 1. Set up the database schema

```bash
# Connect to TimescaleDB
psql -h localhost -U ultrathink -d ultrathink_experiments

# Run schema migration
\i infrastructure/ab_testing_schema.sql
```

### 2. Initialize A/B testing in your inference service

```python
from ab_testing_manager import ABTestingManager
from ab_api_integration import initialize_ab_manager, initialize_ab_storage, ab_router

# In your FastAPI app lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing setup ...

    # Initialize A/B testing
    ab_manager = initialize_ab_manager(model_cache)
    await initialize_ab_storage()

    yield

    # Cleanup
    await ab_manager.flush_results()

# Add A/B testing routes
app.include_router(ab_router)
```

### 3. Create an A/B test

**Option A: Using the API**

```bash
curl -X POST http://localhost:8080/api/v1/ab-test/create \
  -H "Content-Type: application/json" \
  -d '{
    "test_id": "bull_v2_canary",
    "control_model": "bull_specialist",
    "treatment_model": "bull_specialist_v2",
    "traffic_split": 0.05,
    "mode": "traffic_split",
    "description": "5% canary deployment of bull specialist v2"
  }'
```

**Option B: Programmatically**

```python
from ab_testing_manager import ABTestMode

# Create a canary test (5% to new model)
config = ab_manager.create_test(
    test_id="bull_v2_canary",
    control_model="bull_specialist",
    treatment_model="bull_specialist_v2",
    traffic_split=0.05,
    mode=ABTestMode.TRAFFIC_SPLIT,
    description="5% canary deployment"
)

# Create a shadow mode test (runs both, uses control)
shadow_config = ab_manager.create_test(
    test_id="bear_v2_shadow",
    control_model="bear_specialist",
    treatment_model="bear_specialist_v2",
    traffic_split=0.5,  # Doesn't matter in shadow mode
    mode=ABTestMode.SHADOW,
    description="Shadow comparison before rollout"
)
```

## Usage Patterns

### Traffic Splitting (Canary Deployment)

Use this to gradually roll out a new model:

```python
# Start with 5% traffic
ab_manager.create_test(
    test_id="new_model_rollout",
    control_model="current_model",
    treatment_model="new_model_v2",
    traffic_split=0.05
)

# Monitor for 24 hours, then ramp to 10%
ab_manager.update_traffic_split("new_model_rollout", 0.10)

# Continue ramping: 25%, 50%, 100%
ab_manager.update_traffic_split("new_model_rollout", 0.25)
ab_manager.update_traffic_split("new_model_rollout", 0.50)
ab_manager.update_traffic_split("new_model_rollout", 1.00)
```

**Key features:**
- Consistent hashing ensures same request_id always goes to same group
- Traffic split accuracy: ±2% (verified automatically)
- Instant rollback: `ab_manager.disable_test("test_id")` routes all traffic to control

### Shadow Mode (Risk-Free Comparison)

Use this to compare models without affecting production:

```python
# Run both models, always use control for actual decisions
ab_manager.create_test(
    test_id="compare_models",
    control_model="current_model",
    treatment_model="experimental_model",
    mode=ABTestMode.SHADOW
)

# Collect data for 7 days, analyze:
stats = await storage_backend.get_test_stats("compare_models", hours_back=168)

print(f"Agreement rate: {stats['metrics']['comparison']['agreement_rate']:.2%}")
print(f"Confidence delta: {stats['metrics']['comparison']['avg_confidence_delta']:+.3f}")
print(f"Latency delta: {stats['metrics']['comparison']['avg_latency_delta_ms']:+.1f}ms")
```

**Key features:**
- Zero production risk (always uses control model)
- Captures full comparison metrics
- Identifies edge cases where models disagree
- Measures latency impact of new model

## API Reference

### Endpoints

#### `POST /api/v1/ab-test/create`
Create a new A/B test.

**Request:**
```json
{
  "test_id": "string",
  "control_model": "string",
  "treatment_model": "string",
  "traffic_split": 0.05,
  "mode": "traffic_split",
  "description": "string"
}
```

**Response:** `201 Created`
```json
{
  "test_id": "string",
  "control_model": "string",
  "treatment_model": "string",
  "traffic_split": 0.05,
  "mode": "traffic_split",
  "enabled": true,
  "created_at": "2024-10-25T12:00:00Z",
  "description": "string"
}
```

#### `GET /api/v1/ab-test/list`
List all active A/B tests.

**Response:** `200 OK`
```json
[
  {
    "test_id": "bull_v2_canary",
    "control_model": "bull_specialist",
    "treatment_model": "bull_specialist_v2",
    "traffic_split": 0.05,
    "mode": "traffic_split",
    "enabled": true,
    "created_at": "2024-10-25T12:00:00Z",
    "description": "5% canary deployment"
  }
]
```

#### `GET /api/v1/ab-test/{test_id}/stats?hours_back=24`
Get statistics for an A/B test.

**Response:** `200 OK`
```json
{
  "test_id": "bull_v2_canary",
  "time_window_hours": 24,
  "total_samples": 10000,
  "control_count": 9500,
  "treatment_count": 500,
  "shadow_count": 0,
  "metrics": {
    "control": {
      "avg_confidence": 0.78,
      "avg_latency_ms": 12.5,
      "stddev_latency_ms": 3.2
    },
    "treatment": {
      "avg_confidence": 0.82,
      "avg_latency_ms": 15.1,
      "stddev_latency_ms": 4.1
    },
    "comparison": {
      "agreement_rate": 0.95,
      "avg_confidence_delta": 0.04,
      "avg_latency_delta_ms": 2.6
    }
  }
}
```

#### `POST /api/v1/ab-test/{test_id}/update-split`
Update traffic split for a test.

**Request:**
```json
{
  "traffic_split": 0.10
}
```

**Response:** `200 OK`
```json
{
  "test_id": "bull_v2_canary",
  "traffic_split": 0.10,
  "status": "updated"
}
```

#### `POST /api/v1/ab-test/{test_id}/disable`
Disable a test (routes all traffic to control).

**Response:** `200 OK`
```json
{
  "test_id": "bull_v2_canary",
  "status": "disabled"
}
```

#### `POST /api/v1/ab-test/{test_id}/enable`
Re-enable a test.

**Response:** `200 OK`
```json
{
  "test_id": "bull_v2_canary",
  "status": "enabled"
}
```

## Database Schema

### `ab_test_configs` Table
Stores A/B test configurations.

| Column | Type | Description |
|--------|------|-------------|
| test_id | VARCHAR(100) | Primary key |
| mode | VARCHAR(50) | traffic_split, shadow, disabled |
| control_model | VARCHAR(255) | Control model name |
| treatment_model | VARCHAR(255) | Treatment model name |
| traffic_split | DOUBLE PRECISION | Percentage to treatment (0.0-1.0) |
| enabled | BOOLEAN | Whether test is active |
| created_at | TIMESTAMPTZ | Creation timestamp |
| description | TEXT | Human-readable description |

### `ab_test_results` Table (Hypertable)
Stores prediction comparisons (time-series data).

| Column | Type | Description |
|--------|------|-------------|
| time | TIMESTAMPTZ | Prediction timestamp |
| test_id | VARCHAR(100) | Test identifier |
| request_id | VARCHAR(100) | Request identifier |
| assigned_group | VARCHAR(20) | control, treatment, shadow |
| symbol | VARCHAR(20) | Trading symbol |
| control_action | VARCHAR(10) | Control model prediction |
| control_confidence | DOUBLE PRECISION | Control confidence score |
| control_latency_ms | DOUBLE PRECISION | Control latency |
| treatment_action | VARCHAR(10) | Treatment model prediction (nullable) |
| treatment_confidence | DOUBLE PRECISION | Treatment confidence (nullable) |
| treatment_latency_ms | DOUBLE PRECISION | Treatment latency (nullable) |
| actions_match | BOOLEAN | Whether predictions agree (shadow mode) |
| confidence_delta | DOUBLE PRECISION | Confidence difference |
| latency_delta_ms | DOUBLE PRECISION | Latency difference |

### Continuous Aggregates

#### `ab_test_metrics_hourly`
Hourly aggregated metrics for monitoring.

#### `ab_test_metrics_daily`
Daily aggregated metrics for trend analysis.

### Helper Functions

```sql
-- Get test summary
SELECT * FROM get_ab_test_summary('test_id');

-- Verify traffic split accuracy
SELECT * FROM verify_traffic_split('test_id', hours_back => 1);
```

## Monitoring and Alerting

### Key Metrics

1. **Traffic Split Accuracy**
   ```sql
   SELECT * FROM verify_traffic_split('test_id', 1);
   ```
   Should be within ±2% of target.

2. **Agreement Rate** (Shadow Mode)
   ```sql
   SELECT agreement_rate
   FROM ab_test_metrics_hourly
   WHERE test_id = 'test_id'
   ORDER BY bucket DESC LIMIT 1;
   ```
   Low agreement (<90%) indicates models behave differently.

3. **Latency Impact**
   ```sql
   SELECT avg_latency_delta_ms
   FROM ab_test_metrics_hourly
   WHERE test_id = 'test_id'
   ORDER BY bucket DESC LIMIT 1;
   ```
   Monitor for latency regressions.

### Recommended Alerts

- **Traffic split deviation >2%**: Indicates hash function issue
- **Agreement rate <85%** (shadow mode): Models significantly disagree
- **P95 latency delta >10ms**: New model is slower
- **Confidence delta >0.1**: Significant prediction differences

## Best Practices

### 1. Start with Shadow Mode
Before any traffic splitting, run in shadow mode for at least 7 days:
```python
ab_manager.create_test(
    test_id="new_model_validation",
    control_model="current",
    treatment_model="new",
    mode=ABTestMode.SHADOW
)
```

### 2. Gradual Rollout
Use conservative traffic ramp:
- Day 1-2: 5%
- Day 3-4: 10%
- Day 5-7: 25%
- Week 2: 50%
- Week 3: 100%

### 3. Monitor Continuously
Set up Grafana dashboards tracking:
- Sample sizes per group
- Agreement rates
- Latency percentiles
- Confidence distributions

### 4. Quick Rollback Plan
Always have instant rollback ready:
```python
# Emergency rollback
ab_manager.disable_test("test_id")  # All traffic -> control
```

### 5. Document Findings
After each test, document:
- Agreement rate
- Performance differences
- Edge cases discovered
- Decision (rollout, rollback, iterate)

## Performance Considerations

### Latency Impact

- **Traffic Split Mode**: <1ms overhead (single model prediction)
- **Shadow Mode**: +10-20ms (runs both models sequentially)
- **Buffering**: <5ms for result storage (async)

### Optimization Tips

1. **Use Shadow Mode selectively**: Only for critical rollouts
2. **Adjust buffer size**: Default is 100 results, tune based on traffic
3. **Batch inserts**: Results are batched to minimize DB load
4. **Retention policies**: Data automatically deleted after 90 days

## Troubleshooting

### Issue: Traffic split not matching target

**Symptoms:** Actual split is 7% but expecting 5%

**Solution:**
```sql
SELECT * FROM verify_traffic_split('test_id', hours_back => 24);
```
Check `delta` column. If >0.02, increase sample size or check hash function.

### Issue: Shadow mode predictions always differ

**Symptoms:** Agreement rate <50%

**Possible causes:**
1. Models trained on different data
2. Different preprocessing
3. Different random seeds (if using sampling)

**Debug:**
```python
# Examine specific disagreements
results = await storage_backend.get_action_distribution('test_id')
print(results)
```

### Issue: High latency in shadow mode

**Symptoms:** P95 latency >50ms

**Solution:**
- Models run sequentially; consider GPU batching if available
- Check if treatment model is larger/slower
- Monitor `latency_delta_ms` to isolate bottleneck

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest tests/test_ab_testing.py -v

# Specific test classes
pytest tests/test_ab_testing.py::TestABTestingManager -v

# Integration tests only
pytest tests/test_ab_testing.py::TestIntegration -v
```

**Test Coverage:** 20 tests covering:
- Configuration validation
- Traffic splitting accuracy
- Shadow mode execution
- Results buffering
- End-to-end workflows

## Migration Guide

### From Manual Model Swapping

**Before:**
```python
# Risky: instant 100% rollout
model_cache.load_model('production', 'new_model_v2.pth')
```

**After:**
```python
# Safe: gradual rollout with monitoring
ab_manager.create_test(
    test_id="safe_rollout",
    control_model="current_production",
    treatment_model="new_model_v2",
    traffic_split=0.05  # Start small
)
```

### Integrating Existing Models

1. Ensure models are loaded in ModelCache
2. Create A/B test configuration
3. Monitor for 24+ hours
4. Gradually increase traffic split
5. Promote to production when confident

## FAQ

**Q: Can I run multiple A/B tests simultaneously?**
A: Yes, but route each test to different request paths to avoid interference.

**Q: How long should I run a shadow mode test?**
A: Minimum 7 days to capture weekly patterns. 14-30 days for high-stakes rollouts.

**Q: What happens if treatment model crashes?**
A: Requests are automatically routed to control. Set up health checks and alerts.

**Q: Can I A/B test different strategies (bull/bear/sideways)?**
A: Yes, set up separate tests for each regime type with appropriate baseline models.

**Q: How do I promote treatment to production?**
A: Either (1) update traffic_split to 1.0, or (2) swap model names in production config.

## References

- **Technical Spec**: `/home/rich/ultrathink-pilot/technical-spec.md`
- **Database Schema**: `/home/rich/ultrathink-pilot/infrastructure/ab_testing_schema.sql`
- **Test Suite**: `/home/rich/ultrathink-pilot/tests/test_ab_testing.py`
- **Model Loader**: `/home/rich/ultrathink-pilot/services/inference_service/model_loader.py`

## Support

For issues or questions:
1. Check logs in `/app/logs/inference_api.log`
2. Query TimescaleDB for A/B test results
3. Review test failures for debugging
4. Consult system architecture documentation

---

**Version:** 1.0.0
**Last Updated:** 2024-10-25
**Maintainer:** ML Training Specialist (Agent 9)

# Inference Service

Low-latency prediction API for algorithmic trading with GPU acceleration and regime-aware strategy blending.

## Overview

The Inference Service provides real-time trading decisions (<50ms P95 latency) using ensemble reinforcement learning models with market regime detection. It implements the `/api/v1/predict` endpoint specified in the trading system architecture.

## Features

- **Fast Inference:** GPU-accelerated predictions with PyTorch
- **Warm Model Cache:** Models loaded at startup, no cold starts
- **Async Architecture:** Non-blocking I/O for service integration
- **Regime-Aware:** Integrates with regime detection for adaptive strategy selection
- **Risk Integration:** Optional risk validation before returning predictions
- **Observability:** Prometheus metrics for latency and throughput monitoring
- **Docker Ready:** Containerized with GPU support

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional, falls back to CPU)
- Docker with NVIDIA runtime (for containerized deployment)

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MODEL_DIR=/path/to/models
export MOCK_DATA_SERVICE=true
export MOCK_REGIME_DETECTION=true
export MOCK_META_CONTROLLER=true
export MOCK_RISK_MANAGER=true

# Run the service
python inference_api.py

# Or with uvicorn directly
uvicorn inference_api:app --host 0.0.0.0 --port 8080 --reload
```

### Docker Deployment

```bash
# From infrastructure directory
cd ../../infrastructure

# Build and start
docker-compose up -d inference-service

# Check logs
docker-compose logs -f inference-service

# Check health
curl http://localhost:8080/health
```

## API Endpoints

### POST /api/v1/predict

Make a trading prediction.

**Request:**
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-10-24T00:00:00Z",
  "strategy_override": null,
  "risk_check": true,
  "explain": false
}
```

**Response:**
```json
{
  "decision_id": "uuid",
  "symbol": "AAPL",
  "action": "BUY",
  "confidence": 0.85,
  "recommended_quantity": 100,
  "regime_probabilities": {
    "bull": 0.6,
    "bear": 0.2,
    "sideways": 0.2,
    "entropy": 0.82
  },
  "strategy_weights": {
    "bull_specialist": 0.6,
    "bear_specialist": 0.1,
    "sideways_specialist": 0.3
  },
  "risk_validation": {
    "approved": true,
    "warnings": [],
    "checks": {
      "position_limit": "pass",
      "concentration": "pass",
      "daily_loss_limit": "pass"
    }
  },
  "metadata": {
    "model_version": "bull_specialist",
    "latency_ms": 45,
    "timestamp": "2025-10-24T00:00:00.123Z"
  }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "timestamp": "2025-10-24T00:00:00Z"
}
```

### GET /api/v1/models

List loaded models.

**Response:**
```json
{
  "models": {
    "bull_specialist": {
      "name": "bull_specialist",
      "version": "best_model",
      "loaded_at": "2025-10-24T00:00:00Z",
      "device": "cuda:0",
      "parameters": 2456789
    }
  },
  "total_models": 3
}
```

### GET /metrics

Prometheus metrics endpoint.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `/app/models` | Directory containing model checkpoints |
| `MOCK_DATA_SERVICE` | `true` | Use mock data service if true |
| `MOCK_REGIME_DETECTION` | `true` | Use mock regime detection if true |
| `MOCK_META_CONTROLLER` | `true` | Use mock meta-controller if true |
| `MOCK_RISK_MANAGER` | `true` | Use mock risk manager if true |
| `DATA_SERVICE_URL` | `http://data-service:8000` | Data service endpoint |
| `REGIME_DETECTION_URL` | `http://regime-detection:8001` | Regime detection endpoint |
| `META_CONTROLLER_URL` | `http://meta-controller:8002` | Meta-controller endpoint |
| `RISK_MANAGER_URL` | `http://risk-manager:8003` | Risk manager endpoint |

### Model Directory Structure

Models should be organized as follows:

```
models/
├── bull_specialist/
│   └── best_model.pth
├── bear_specialist/
│   └── best_model.pth
├── sideways_specialist/
│   └── best_model.pth
└── best_model.pth  # Universal fallback
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest test_inference_api.py -v

# Run specific test class
pytest test_inference_api.py::TestModelLoader -v

# Run with coverage
pytest test_inference_api.py --cov=. --cov-report=html
```

### Load Testing

```bash
# Basic load test (1000 requests)
python load_test.py

# Custom configuration
python load_test.py \
  --url http://localhost:8080 \
  --requests 10000 \
  --concurrent 20 \
  --output results.json
```

**Load Test Output:**
- Mean, median, and percentile latencies (P50, P75, P90, P95, P99)
- Success rate
- Action distribution
- JSON results file for analysis

## Performance

### Latency Targets

- **P50:** <25ms
- **P95:** <50ms (strict target)
- **P99:** <100ms

### Latency Budget

| Component | Target | Notes |
|-----------|--------|-------|
| Feature fetch | <20ms | Async, parallel with other calls |
| Regime detection | <10ms | Async, parallel |
| Meta-controller | <10ms | Async, parallel |
| Model inference | <15ms | GPU-accelerated |
| Risk validation | <10ms | Async, optional |
| Serialization | <5ms | Pydantic |

### Optimization Techniques

1. **Warm Model Cache:** Models loaded at startup, kept in GPU memory
2. **Async I/O:** Non-blocking service calls with `aiohttp`
3. **Parallel Fetching:** Features, regime, and strategy weights fetched concurrently
4. **GPU Acceleration:** CUDA-enabled PyTorch inference
5. **Request Validation:** Fast Pydantic validation
6. **Connection Pooling:** Reused HTTP connections

## Architecture

### Components

1. **inference_api.py:** FastAPI application with endpoints
2. **model_loader.py:** PyTorch model cache with GPU support
3. **service_clients.py:** Async HTTP clients for microservices
4. **models.py:** Pydantic request/response schemas

### Data Flow

```
Client Request
    ↓
FastAPI (validation)
    ↓
Service Clients (async parallel)
    ├── Data Service → Features
    ├── Regime Detection → Probabilities
    └── Meta-Controller → Weights
    ↓
Model Selection (based on weights)
    ↓
GPU Inference (PyTorch)
    ↓
Quantity Calculation
    ↓
Risk Manager (optional validation)
    ↓
Response (with metadata)
```

### Model Selection

The service selects which specialist model to use based on strategy weights:

1. Get strategy weights from Meta-Controller
2. Select specialist with highest weight
3. Fallback to universal model if specialist unavailable
4. Or use `strategy_override` if provided in request

## Monitoring

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `predictions_total{status}` | Counter | Total predictions by status |
| `prediction_latency_seconds` | Histogram | Prediction latency distribution |
| `active_requests` | Gauge | Current active requests |
| `model_load_time_seconds` | Gauge | Startup model loading time |

### Grafana Dashboard

Create a dashboard with:
- Prediction throughput (requests/sec)
- P50/P95/P99 latency over time
- Success vs. error rate
- Action distribution (BUY/SELL/HOLD)
- GPU utilization

## Troubleshooting

### Models Not Loading

**Problem:** Service starts but no models loaded

**Solution:**
1. Check model directory mount: `docker-compose exec inference-service ls /app/models`
2. Verify model files exist: `ls -la /path/to/rl/models/`
3. Check logs: `docker-compose logs inference-service | grep -i model`

### High Latency

**Problem:** P95 latency exceeds 50ms

**Solutions:**
1. Check GPU usage: `nvidia-smi`
2. Profile with load test: `python load_test.py --requests 100`
3. Check service timeouts in `service_clients.py`
4. Verify async operations are not blocking

### Out of Memory

**Problem:** GPU OOM errors

**Solutions:**
1. Reduce concurrent requests
2. Check model size: `GET /api/v1/models`
3. Monitor GPU memory: `nvidia-smi`
4. Consider CPU fallback for some models

## Development

### Adding New Models

1. Train model using PPO agent
2. Save checkpoint to `rl/models/{specialist_name}/best_model.pth`
3. Restart inference service to load new model
4. Verify: `curl http://localhost:8080/api/v1/models`

### Extending Service Clients

To add a new service integration:

1. Create client class in `service_clients.py`
2. Add async methods with timeout
3. Implement mock fallback
4. Add environment variables for URL and mock flag
5. Update `ServiceClients` aggregator

### Adding Metrics

```python
from prometheus_client import Counter, Histogram

MY_METRIC = Counter('my_metric_total', 'Description')

# In endpoint
MY_METRIC.inc()
```

## Future Enhancements

### Wave 2 (Performance)
- [ ] Kafka producer for forensics events
- [ ] MLflow model registry integration
- [ ] Request batching
- [ ] Model warm-up at startup

### Wave 3 (Production)
- [ ] A/B testing framework
- [ ] Shadow mode
- [ ] Rate limiting
- [ ] SHAP explanations
- [ ] Authentication (JWT/API key)

## References

- Technical Spec: `/trading-system-architectural-enhancement-docs/technical-spec.md`
- Deployment Plan: `/ultrathink-pilot/deployment-plan.md`
- Validation Report: `/ultrathink-pilot/INFERENCE_API_VALIDATION.md`

## Support

For issues or questions:
1. Check logs: `docker-compose logs inference-service`
2. Run health check: `curl http://localhost:8080/health`
3. Review validation report: `INFERENCE_API_VALIDATION.md`
4. Contact: inference-api-engineer (Wave 1 Agent 3)

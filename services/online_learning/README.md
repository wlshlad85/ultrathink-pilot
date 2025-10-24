# Online Learning Service

Incremental model updates with Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.

## Features

- **EWC Algorithm**: Prevents catastrophic forgetting during incremental learning
- **Conservative Updates**: Learning rate 1e-5, EWC lambda 1000
- **Stability Monitoring**: Automatic rollback if Sharpe ratio degrades >30%
- **Sliding Window**: 30-90 days of recent market data
- **Performance Target**: <5% degradation over 30 days

## Architecture

```
┌─────────────────────────────────────────────┐
│         Online Learning Service             │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┐  ┌──────────────────┐     │
│  │ EWC Trainer │  │ Stability Checker│     │
│  └─────────────┘  └──────────────────┘     │
│         │                  │                │
│         └──────────┬───────┘                │
│                    │                        │
│         ┌──────────▼──────────┐             │
│         │   Data Manager      │             │
│         │ (Sliding Window)    │             │
│         └─────────────────────┘             │
│                                             │
└─────────────────────────────────────────────┘
```

## API Endpoints

### POST /api/v1/models/online-update

Trigger incremental model update.

**Request:**
```json
{
  "window_days": 60,
  "learning_rate": 1e-5,
  "ewc_lambda": 1000,
  "skip_stability_check": false
}
```

**Response:**
```json
{
  "success": true,
  "update_count": 5,
  "metrics": {
    "total_loss": 0.234,
    "task_loss": 0.189,
    "ewc_loss": 0.045
  },
  "stability_status": "stable",
  "degradation_percent": 2.3,
  "checkpoint_path": "/path/to/checkpoint.pth",
  "timestamp": "2025-10-25T12:00:00"
}
```

### GET /api/v1/models/stability

Get current stability status.

**Response:**
```json
{
  "status": "stable",
  "last_check": "2025-10-25T12:00:00",
  "degradation_percent": 2.3,
  "performance_trend": {
    "avg_sharpe": 1.45,
    "sharpe_trend": 0.02,
    "avg_return": 0.015
  },
  "alerts_count": 0
}
```

### GET /api/v1/models/performance

Get latest performance metrics.

**Response:**
```json
{
  "sharpe_ratio": 1.45,
  "total_return": 15.3,
  "volatility": 0.18,
  "max_drawdown": -5.2,
  "win_rate": 58.5,
  "timestamp": "2025-10-25T12:00:00"
}
```

### POST /api/v1/models/rollback

Rollback to previous checkpoint.

**Parameters:**
- `checkpoint_index`: Index of checkpoint (0=most recent)

**Response:**
```json
{
  "success": true,
  "checkpoint": "/path/to/checkpoint.pth",
  "update_count": 4,
  "timestamp": "2025-10-25T12:00:00"
}
```

### GET /api/v1/health

Health check endpoint.

## Usage

### Docker Deployment

```bash
# Build image
docker build -t online-learning-service .

# Run container
docker run -p 8005:8005 \
  -v /path/to/data:/app/data \
  -v /path/to/models:/app/models \
  online-learning-service
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn api:app --host 0.0.0.0 --port 8005 --reload

# Test endpoints
curl http://localhost:8005/api/v1/health
```

### Trigger Update

```bash
curl -X POST http://localhost:8005/api/v1/models/online-update \
  -H "Content-Type: application/json" \
  -d '{
    "window_days": 60,
    "learning_rate": 1e-5,
    "ewc_lambda": 1000
  }'
```

## Configuration

Default configuration in `ewc_trainer.py`:

```python
@dataclass
class EWCConfig:
    learning_rate: float = 1e-5
    ewc_lambda: float = 1000.0
    window_size_days: int = 60
    update_frequency: str = "daily"
    batch_size: int = 64
    epochs_per_update: int = 10
```

## Stability Monitoring

Automatic rollback triggers:
- Sharpe ratio degradation >30%
- Win rate degradation >40%
- Volatility increase >50%

## Performance Guarantees

- **Target**: <5% degradation over 30 days
- **Acceptable**: <10% degradation
- **Rollback**: >30% degradation

## Testing

```bash
# Run tests
pytest tests/test_online_learning.py -v

# Test EWC trainer
python ewc_trainer.py

# Test stability checker
python stability_checker.py

# Test data manager
python data_manager.py
```

## Monitoring

Metrics exported:
- `online_learning_update_count`
- `online_learning_degradation_percent`
- `online_learning_sharpe_ratio`
- `online_learning_update_duration_seconds`

## Alerts

Alerts are logged to:
- `/app/models/online_learning/stability_alerts.jsonl`
- Integration with Prometheus/Grafana (future)
- Slack notifications (future)

## References

- Kirkpatrick et al. (2017): "Overcoming catastrophic forgetting in neural networks"
- EWC Paper: https://arxiv.org/abs/1612.00796

## License

Internal use only - UltraThink Pilot Project

# Online Learning Service - Deployment Guide

## Quick Start

### 1. Build and Start Service

```bash
# Build the service
docker-compose build online-learning

# Start with dependencies
docker-compose --profile infrastructure up -d

# Or start just online learning with data service
docker-compose up -d online-learning
```

### 2. Verify Service is Running

```bash
# Check health
curl http://localhost:8005/api/v1/health

# Expected response:
{
  "status": "healthy",
  "service": "online_learning",
  "trainer_initialized": true,
  "update_in_progress": false,
  "last_update": null,
  "uptime_seconds": 123.45
}
```

### 3. Trigger First Update

```bash
curl -X POST http://localhost:8005/api/v1/models/online-update \
  -H "Content-Type: application/json" \
  -d '{
    "window_days": 60,
    "learning_rate": 1e-5,
    "ewc_lambda": 1000
  }'
```

---

## Environment Variables

Configure in `docker-compose.yml` or `.env` file:

```bash
# Learning parameters
LEARNING_RATE=1e-5          # Very conservative (recommended)
EWC_LAMBDA=1000            # Strong regularization (recommended)
WINDOW_DAYS=60             # Sliding window size (30-90)
UPDATE_FREQUENCY=daily     # Update frequency

# Paths
CHECKPOINT_DIR=/app/models/online_learning
DATA_DIR=/app/data
```

---

## Integration with Training Orchestrator

### API Endpoint

```python
import requests

# Trigger daily update
response = requests.post(
    'http://localhost:8005/api/v1/models/online-update',
    json={
        'window_days': 60,
        'learning_rate': 1e-5,
        'ewc_lambda': 1000
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Update successful: {result['stability_status']}")
    print(f"Degradation: {result['degradation_percent']:.2f}%")
elif response.status_code == 422:
    # Rollback occurred
    error = response.json()
    print(f"Rollback triggered: {error['detail']['message']}")
```

### Scheduled Updates

Add to Training Orchestrator's daily schedule:

```python
# In training_orchestrator.py

async def daily_online_learning_update():
    """Trigger daily EWC update."""
    try:
        response = await http_client.post(
            'http://online-learning:8005/api/v1/models/online-update',
            json={'window_days': 60}
        )

        if response.status == 200:
            result = await response.json()
            logger.info(f"Online learning update: {result['stability_status']}")

        elif response.status == 422:
            # Rollback occurred
            logger.error("Online learning rollback triggered")

    except Exception as e:
        logger.error(f"Online learning update failed: {e}")
```

---

## Monitoring

### Prometheus Metrics

Export metrics for monitoring:

```python
# Add to api.py

from prometheus_client import Counter, Gauge, Histogram

# Metrics
update_count = Counter('online_learning_updates_total', 'Total updates')
degradation = Gauge('online_learning_degradation_percent', 'Performance degradation')
sharpe_ratio = Gauge('online_learning_sharpe_ratio', 'Current Sharpe ratio')
update_duration = Histogram('online_learning_update_duration_seconds', 'Update duration')
```

### Grafana Dashboard

Query examples:

```promql
# Degradation over time
online_learning_degradation_percent

# Update rate
rate(online_learning_updates_total[1h])

# Current Sharpe ratio
online_learning_sharpe_ratio

# Update duration (P95)
histogram_quantile(0.95, online_learning_update_duration_seconds)
```

---

## Troubleshooting

### Service Won't Start

**Check logs:**
```bash
docker-compose logs online-learning
```

**Common issues:**
- Missing dependencies: Rebuild image
- Port conflict: Change port in docker-compose.yml
- Data directory missing: Create `/home/rich/ultrathink-pilot/data`

### Update Fails

**Check stability status:**
```bash
curl http://localhost:8005/api/v1/models/stability
```

**If degradation too high:**
- Reduce learning rate: `1e-5 → 5e-6`
- Increase EWC lambda: `1000 → 2000`
- Increase window size: `60 → 90 days`

### Rollback Triggered

**View alerts:**
```bash
cat /home/rich/ultrathink-pilot/rl/models/online_learning/stability_alerts.jsonl
```

**Manual rollback:**
```bash
curl -X POST http://localhost:8005/api/v1/models/rollback?checkpoint_index=0
```

---

## Production Checklist

Before production deployment:

- [ ] Set conservative learning rate (1e-5)
- [ ] Configure strong EWC lambda (1000+)
- [ ] Enable automatic rollback
- [ ] Set up Prometheus metrics
- [ ] Configure Grafana dashboards
- [ ] Test rollback mechanism
- [ ] Configure daily update schedule
- [ ] Set up alert notifications
- [ ] Document emergency procedures
- [ ] Test with shadow mode (1 week)

---

## Performance Tuning

### Faster Updates

If update latency is critical:
- Reduce epochs per update: `10 → 5`
- Reduce Fisher sample size: `1000 → 500`
- Use smaller batch size: `64 → 32`

**Trade-off:** Less stable, may need stronger regularization

### More Stability

If experiencing frequent rollbacks:
- Increase EWC lambda: `1000 → 2000`
- Reduce learning rate: `1e-5 → 5e-6`
- Increase window size: `60 → 90 days`
- Reduce update frequency: `daily → weekly`

**Trade-off:** Slower adaptation to market changes

---

## Maintenance

### Daily Tasks
- Check stability status
- Review degradation metrics
- Monitor update success rate

### Weekly Tasks
- Review checkpoint disk usage
- Analyze performance trends
- Update configuration if needed

### Monthly Tasks
- Audit Fisher Information Matrix
- Review and tune hyperparameters
- Archive old checkpoints
- Update documentation

---

## Emergency Procedures

### Service Unresponsive

1. Check service logs: `docker-compose logs online-learning`
2. Restart service: `docker-compose restart online-learning`
3. If still failing, rollback: Deploy previous version

### Critical Performance Degradation

1. Immediate: Stop automatic updates
   ```bash
   # Disable online learning in orchestrator
   ```

2. Manual rollback:
   ```bash
   curl -X POST http://localhost:8005/api/v1/models/rollback?checkpoint_index=0
   ```

3. Investigate root cause
4. Adjust hyperparameters
5. Test in shadow mode before re-enabling

### Data Corruption

1. Stop service
2. Restore from backup
3. Clear cache: `manager.clear_cache()`
4. Restart with fresh data

---

## Support

For issues or questions:
- Check validation report: `ONLINE_LEARNING_VALIDATION.md`
- Review API docs: `README.md`
- Check deployment plan: `/home/rich/ultrathink-pilot/deployment-plan.md`

---

**Last Updated:** 2025-10-25
**Version:** 1.0.0
**Agent:** online-learning-engineer

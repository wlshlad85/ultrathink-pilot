# UltraThink Pilot - Monitoring & Observability Runbook

**Version:** 1.0
**Last Updated:** 2025-10-25
**Maintained by:** Infrastructure Team

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Dashboard Guide](#dashboard-guide)
4. [Alert Response Procedures](#alert-response-procedures)
5. [Common Troubleshooting Scenarios](#common-troubleshooting-scenarios)
6. [On-Call Rotation Guidelines](#on-call-rotation-guidelines)
7. [Appendix](#appendix)

---

## Overview

### Monitoring Stack

The UltraThink Pilot monitoring infrastructure consists of:

- **Prometheus** (Port 9090): Metrics collection and time-series database
- **AlertManager** (Port 9093): Alert routing and notification management
- **Grafana** (Port 3000): Visualization and dashboards
- **TimescaleDB** (Port 5432): Long-term storage for training metrics

### Key Metrics Categories

1. **Training Performance**: Episode returns, Sharpe ratios, win rates
2. **System Performance**: CPU/GPU utilization, memory, cache hit rates
3. **Trading Operations**: Action distribution, portfolio value, P&L
4. **Service Health**: Uptime, latency, throughput

---

## Architecture

### Data Flow

```
┌─────────────────┐
│   Services      │
│ (Prometheus     │
│  Exporters)     │
└────────┬────────┘
         │ metrics (pull)
         ▼
┌─────────────────┐      ┌──────────────┐
│   Prometheus    │─────▶│ AlertManager │
│  (Port 9090)    │      │ (Port 9093)  │
└────────┬────────┘      └──────┬───────┘
         │                      │
         │ query                │ notifications
         ▼                      ▼
┌─────────────────┐      ┌──────────────┐
│    Grafana      │      │    Slack     │
│  (Port 3000)    │      │   Channels   │
└─────────────────┘      └──────────────┘
```

### Slack Alert Channels

- **#ultrathink-alerts-critical**: Critical alerts requiring immediate action
- **#ultrathink-alerts**: Warning-level alerts
- **#ultrathink-ops**: Infrastructure and operational alerts

---

## Dashboard Guide

### Access URLs

- **Grafana UI**: http://localhost:3000 (admin/admin)
- **Prometheus UI**: http://localhost:9090
- **AlertManager UI**: http://localhost:9093

### Dashboard 1: Training Metrics

**Path**: Grafana → UltraThink → Training Metrics
**URL**: http://localhost:3000/d/ultrathink-training-metrics
**Refresh**: 30s auto-refresh

#### Panels Overview

1. **Episode Returns (Time Series)**
   - Shows episode return percentage over time
   - **What to look for**: Positive trending line indicates improving performance
   - **Red flag**: Consistently negative returns or sudden drops

2. **Rolling Sharpe Ratio (10/50/100 Episode Windows)**
   - Risk-adjusted returns across three window sizes
   - **What to look for**: Values > 1.0 indicate good risk-adjusted performance
   - **Red flag**: Sharpe ratio dropping below 0.5 for extended periods

3. **Win Rate % (Gauge + Time Series)**
   - Percentage of profitable episodes
   - **Target**: > 50% for profitable strategy
   - **Red flag**: Win rate < 40% sustained over 1 hour

4. **Episode Length Distribution**
   - Histogram showing distribution of episode durations
   - **What to look for**: Normal distribution around expected length
   - **Red flag**: Bimodal distribution or very short episodes (< 50 steps)

5. **Cumulative Rewards Over Time**
   - Total accumulated rewards since tracking began
   - **What to look for**: Positive slope indicating profitability
   - **Red flag**: Flattening or negative slope

#### Usage Tips

- Use the **Experiment ID** dropdown to filter by specific training runs
- Compare multiple experiments by selecting different time ranges
- Export data via panel menu → Inspect → Data → Download CSV

---

### Dashboard 2: System Performance

**Path**: Grafana → UltraThink → System Performance
**URL**: http://localhost:3000/d/ultrathink-system-performance
**Refresh**: 30s auto-refresh

#### Panels Overview

1. **CPU/GPU Utilization**
   - Real-time resource usage across services
   - **Target**: 60-80% sustained utilization (good efficiency)
   - **Red flag**: Sustained > 90% (bottleneck) or < 20% (underutilization)

2. **Memory Consumption (with Leak Detection)**
   - RSS memory usage + leak detection via rate of change
   - **Red flag**: Continuous upward trend (memory leak)
   - **Action**: Restart affected service if leak detected

3. **Cache Hit Rate**
   - Redis cache effectiveness
   - **Target**: > 90% hit rate
   - **Warning**: < 80% hit rate for 30+ minutes
   - **Impact**: Lower cache hits → higher database load & latency

4. **Training Throughput**
   - Episodes completed per hour
   - **What to look for**: Stable throughput matching capacity
   - **Red flag**: Sudden drop in throughput

5. **API Latency (P50/P95/P99)**
   - Request latency percentiles
   - **Target**: P95 < 100ms, P99 < 200ms
   - **Critical**: P95 > 200ms sustained for 5+ minutes

6. **Service Health Status**
   - Binary UP/DOWN indicators for all services
   - **Red flag**: Any service showing DOWN

#### Usage Tips

- Use the **Service** dropdown to filter specific services
- Set up alerts for latency spikes via Grafana alerting
- Check container logs if service shows DOWN: `docker logs <service-name>`

---

### Dashboard 3: Trading Decisions

**Path**: Grafana → UltraThink → Trading Decisions
**URL**: http://localhost:3000/d/ultrathink-trading-decisions
**Refresh**: 30s auto-refresh

#### Panels Overview

1. **Action Distribution (Pie Chart)**
   - Breakdown of BUY/HOLD/SELL decisions
   - **What to look for**: Balanced distribution (not all HOLD)
   - **Red flag**: > 95% single action type (stuck strategy)

2. **Portfolio Value Over Time**
   - Current portfolio value trajectory
   - **What to look for**: Upward trending with controlled volatility
   - **Red flag**: Sudden large drops or erratic swings

3. **P&L Per Trade (Histogram)**
   - Distribution of profit/loss across trades
   - **What to look for**: Right-skewed distribution (more profits)
   - **Red flag**: Left-skewed (more losses) or fat left tail (large losses)

4. **Trade Frequency (Gauge)**
   - Trades per minute rate
   - **Expected**: 1-20 trades/min depending on strategy
   - **Red flag**: Sudden spike (possible runaway trading) or zero (stuck)

5. **Risk Limit Violations**
   - Count of risk check violations in last hour
   - **Target**: 0 violations
   - **Critical**: Any violation should trigger investigation

6. **Market Regime Probabilities**
   - Stacked area chart of bull/bear/sideways regime probabilities
   - **What to look for**: Smooth transitions between regimes
   - **Use case**: Correlate performance with regime shifts

#### Usage Tips

- Filter by **Experiment ID** to analyze specific trading sessions
- Cross-reference P&L patterns with regime probabilities
- Export trade history for detailed post-mortem analysis

---

## Alert Response Procedures

### Critical Alerts (Immediate Action Required)

#### 1. TradingLatencyHigh

**Severity**: CRITICAL
**Trigger**: P95 API latency > 200ms for 5+ minutes
**Impact**: Degraded trade execution, potential slippage

**Response Steps**:
1. Check system dashboard for bottlenecks (CPU/memory/disk)
2. Verify no network issues: `docker exec ultrathink-prometheus ping inference-service`
3. Check service logs: `docker logs ultrathink-inference-service --tail 100`
4. If GPU bottleneck: Check GPU utilization, consider horizontal scaling
5. If database bottleneck: Check TimescaleDB query performance
6. **Escalation**: If unresolved in 15 minutes, page senior engineer

**Mitigation**:
- Restart affected service if memory leak detected
- Scale horizontally if sustained high load
- Enable caching for frequently accessed data

---

#### 2. RiskLimitViolationNotBlocked

**Severity**: CRITICAL
**Trigger**: Risk check bypass detected
**Impact**: Uncontrolled position sizes, potential financial loss

**Response Steps**:
1. **IMMEDIATELY**: Stop trading via emergency circuit breaker
   ```bash
   docker exec ultrathink-risk-manager curl -X POST http://localhost:8001/circuit-breaker/enable
   ```
2. Review recent trades in forensics dashboard
3. Check risk-manager logs for bypass reason:
   ```bash
   docker logs ultrathink-risk-manager --since 10m | grep -i "bypass"
   ```
4. Verify risk configuration: `/home/rich/ultrathink-pilot/services/risk_manager/config.yml`
5. **DO NOT** resume trading until root cause identified

**Escalation**: Immediately notify team lead and compliance

---

#### 3. ModelServingDown

**Severity**: CRITICAL
**Trigger**: Inference service unreachable for 2+ minutes
**Impact**: Cannot make trading decisions

**Response Steps**:
1. Check service status: `docker ps | grep inference`
2. Attempt restart: `docker restart ultrathink-inference-service`
3. Check logs for crash reason: `docker logs ultrathink-inference-service --tail 200`
4. Verify model files exist: `ls -lh /home/rich/ultrathink-pilot/rl/models/`
5. Check GPU availability: `nvidia-smi`
6. If OOM error: Reduce batch size in service config

**Fallback**:
- Enable "safe mode" trading with conservative rules
- Revert to previous known-good model version via MLflow

---

#### 4. DataPipelineFailure

**Severity**: CRITICAL
**Trigger**: Data service down for 5+ minutes
**Impact**: No feature engineering, stale features

**Response Steps**:
1. Restart service: `docker restart ultrathink-data-service`
2. Check Redis connectivity: `docker exec ultrathink-redis redis-cli ping`
3. Verify data sources are accessible (API keys, network)
4. Check for Kafka lag: `docker exec ultrathink-kafka-1 kafka-consumer-groups --bootstrap-server localhost:9092 --describe --all-groups`
5. Review service logs for errors

**Temporary Mitigation**:
- Use cached features if available (check TTL)
- Reduce feature update frequency to preserve resources

---

#### 5. TimescaleDBConnectionLost

**Severity**: CRITICAL
**Trigger**: TimescaleDB unreachable for 2+ minutes
**Impact**: Cannot log metrics, query historical data

**Response Steps**:
1. Check database status: `docker ps | grep timescaledb`
2. Attempt restart: `docker restart ultrathink-timescaledb`
3. Check disk space: `df -h` (database may be full)
4. Review logs: `docker logs ultrathink-timescaledb --tail 100`
5. Verify connection string in `.env` file
6. Test connection: `docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c "SELECT 1;"`

**Data Loss Prevention**:
- Services should buffer metrics in memory during outage
- Verify backup integrity: `/home/rich/ultrathink-pilot/infrastructure/backups/`

---

### Warning Alerts (Monitor & Plan)

#### 1. ModelRetrainingFailed

**Severity**: WARNING
**Action**: Investigate within 1 hour

**Steps**:
1. Check MLflow UI: http://localhost:5000 → Failed Runs
2. Review training logs: `docker logs ultrathink-training-orchestrator --tail 500`
3. Common causes:
   - Insufficient data (check data volume in TimescaleDB)
   - Hyperparameter instability (gradient explosion)
   - OOM during training (reduce batch size)
4. Manually trigger retry if transient error

---

#### 2. ForensicsBacklogHigh

**Severity**: WARNING
**Action**: Investigate within 2 hours

**Steps**:
1. Check Kafka consumer lag:
   ```bash
   docker exec ultrathink-kafka-1 kafka-consumer-groups \
     --bootstrap-server localhost:9092 \
     --describe --group forensics-consumer
   ```
2. Scale forensics consumer if sustained high lag
3. Check for slow queries in TimescaleDB
4. Consider increasing retention policies if backlog grows

---

#### 3. CacheHitRateLow

**Severity**: WARNING
**Action**: Investigate within 30 minutes

**Steps**:
1. Check Redis memory usage: `docker exec ultrathink-redis redis-cli info memory`
2. Review eviction policy (currently: allkeys-lru)
3. Identify hot keys: `docker exec ultrathink-redis redis-cli --hotkeys`
4. Consider increasing Redis memory limit in docker-compose.yml

---

#### 4. DiskUsageHigh

**Severity**: WARNING
**Action**: Plan cleanup within 4 hours

**Steps**:
1. Identify large files: `du -sh /var/lib/docker/volumes/* | sort -h`
2. Clean old Docker artifacts:
   ```bash
   docker system prune -a --volumes --filter "until=72h"
   ```
3. Archive old experiment data from TimescaleDB
4. Compress/delete old MLflow artifacts

---

#### 5. OnlineLearningDegradation

**Severity**: WARNING
**Action**: Investigate within 1 hour

**Steps**:
1. Check for regime shift in Regime Probabilities dashboard
2. Compare current performance to historical baseline
3. Review recent trades for pattern changes
4. Consider triggering model retraining
5. Enable defensive risk limits if degradation continues

---

## Common Troubleshooting Scenarios

### Scenario 1: Dashboard Not Loading

**Symptoms**: Grafana shows "No data" or "Error reading response"

**Resolution**:
1. Check Grafana service: `docker ps | grep grafana`
2. Verify datasource connection: Grafana → Configuration → Data Sources
3. Test Prometheus query: http://localhost:9090/graph
4. Restart Grafana: `docker restart ultrathink-grafana`
5. Check datasource credentials in provisioning files

---

### Scenario 2: Alerts Not Firing

**Symptoms**: Expected alert doesn't trigger despite threshold breach

**Resolution**:
1. Verify alert rule exists: http://localhost:9090/alerts
2. Check alert state (pending/firing/inactive)
3. Verify AlertManager is receiving alerts: http://localhost:9093/#/alerts
4. Check Slack webhook configuration in `.env`:
   ```bash
   echo $SLACK_WEBHOOK_URL
   ```
5. Test alert manually via Prometheus UI
6. Review AlertManager logs: `docker logs ultrathink-alertmanager`

---

### Scenario 3: Metrics Not Appearing

**Symptoms**: Newly added metric doesn't show in Prometheus

**Resolution**:
1. Verify service is exposing `/metrics` endpoint:
   ```bash
   curl http://localhost:8080/metrics
   ```
2. Check Prometheus scrape targets: http://localhost:9090/targets
3. Verify scrape config in `prometheus.yml`
4. Reload Prometheus config:
   ```bash
   curl -X POST http://localhost:9090/-/reload
   ```
5. Check for metric naming conflicts (duplicates)

---

### Scenario 4: High Memory Usage in Services

**Symptoms**: Service consuming excessive memory

**Resolution**:
1. Identify memory hog: `docker stats`
2. Check for memory leaks via Memory Consumption panel
3. Review recent code changes for unbounded data structures
4. Enable memory profiling:
   ```python
   from memory_profiler import profile
   @profile
   def your_function():
       ...
   ```
5. Restart service as temporary fix
6. Implement fix and redeploy

---

### Scenario 5: Prometheus Storage Full

**Symptoms**: Prometheus fails with "out of disk space" error

**Resolution**:
1. Check storage usage: `docker exec ultrathink-prometheus df -h /prometheus`
2. Reduce retention time (currently 30d) in docker-compose.yml:
   ```yaml
   --storage.tsdb.retention.time=15d
   ```
3. Delete old data: `docker exec ultrathink-prometheus promtool tsdb delete-series`
4. Restart Prometheus: `docker restart ultrathink-prometheus`
5. Consider archiving to TimescaleDB for long-term storage

---

## On-Call Rotation Guidelines

### On-Call Responsibilities

1. **Primary Responder**: First to respond to critical alerts (within 10 minutes)
2. **Secondary Responder**: Backup if primary unresponsive (within 20 minutes)
3. **Escalation Contact**: Team lead for unresolved issues (after 30 minutes)

### Handoff Checklist

When starting on-call shift:
- [ ] Verify access to Slack channels (#ultrathink-alerts-critical)
- [ ] Test AlertManager notifications (silence test alert after)
- [ ] Review open incidents from previous shift
- [ ] Check system health dashboards (all green?)
- [ ] Verify credentials for critical systems
- [ ] Review recent deployments and changes
- [ ] Ensure emergency contact list is current

When ending on-call shift:
- [ ] Document all incidents in incident log
- [ ] Hand off any open investigations
- [ ] Update runbook with new learnings
- [ ] Schedule post-mortem for critical incidents
- [ ] Brief next on-call engineer

### Escalation Matrix

| Severity | Response Time | Escalate After | Contact |
|----------|--------------|----------------|---------|
| Critical | 10 minutes | 30 minutes | Team Lead |
| Warning | 1 hour | 4 hours | Senior Engineer |
| Info | Best effort | N/A | N/A |

### Post-Incident Protocol

After resolving a critical incident:
1. **Immediate (within 1 hour)**: Post incident summary in Slack
2. **Within 24 hours**: Complete incident report template
3. **Within 72 hours**: Conduct blameless post-mortem
4. **Within 1 week**: Implement preventive measures
5. **Update runbook** with new troubleshooting steps

---

## Appendix

### A. Useful Commands

#### Docker Operations
```bash
# View all containers
docker ps -a

# View logs for service
docker logs -f ultrathink-<service-name>

# Restart service
docker restart ultrathink-<service-name>

# Execute command in container
docker exec -it ultrathink-<service-name> bash

# View resource usage
docker stats
```

#### Prometheus
```bash
# Reload configuration
curl -X POST http://localhost:9090/-/reload

# Check scrape targets
curl http://localhost:9090/api/v1/targets

# Query metric
curl 'http://localhost:9090/api/v1/query?query=up'
```

#### AlertManager
```bash
# View active alerts
curl http://localhost:9093/api/v1/alerts

# Create silence
curl -X POST http://localhost:9093/api/v1/silences \
  -d '{"matchers":[{"name":"alertname","value":"TestAlert"}],"startsAt":"...","endsAt":"...","createdBy":"admin","comment":"Maintenance"}'
```

#### TimescaleDB
```bash
# Connect to database
docker exec -it ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments

# Check database size
docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments \
  -c "SELECT pg_size_pretty(pg_database_size('ultrathink_experiments'));"
```

---

### B. Metric Naming Conventions

All custom metrics should follow this pattern:

**Format**: `<namespace>_<subsystem>_<metric_name>_<unit>`

**Examples**:
- `trading_decisions_total` (counter)
- `model_inference_duration_seconds` (histogram)
- `cache_hit_rate_percent` (gauge)
- `risk_violations_total` (counter)

**Metric Types**:
- **Counter**: Monotonically increasing (use for counts)
- **Gauge**: Can increase or decrease (use for current values)
- **Histogram**: Distribution of values (use for latencies)
- **Summary**: Pre-calculated quantiles

---

### C. Dashboard Templates

New dashboards should include:
1. **Time range selector** (default: last 6 hours)
2. **Auto-refresh** (30s recommended)
3. **Template variables** for filtering (experiment_id, service)
4. **Threshold annotations** on panels
5. **Links to runbook** in panel descriptions
6. **Dashboard tags** for organization

---

### D. Alert Tuning Guidelines

**Avoid alert fatigue**:
- Set thresholds based on historical data (use 95th percentile + 20%)
- Use `for` clause to filter transient spikes (minimum 2m)
- Group related alerts to reduce noise
- Use inhibition rules to suppress downstream alerts

**Testing alerts**:
```bash
# Temporarily lower threshold
# Trigger condition manually
# Verify Slack notification received
# Verify runbook link works
# Verify dashboard link works
# Restore original threshold
```

---

### E. Backup & Disaster Recovery

**Automated Backups**:
- TimescaleDB: Daily at 02:00 UTC via `pg_dump`
- Prometheus: Snapshots retained for 7 days
- Grafana: Dashboard JSON exported daily

**Manual Backup**:
```bash
# Backup TimescaleDB
docker exec ultrathink-timescaledb pg_dump -U ultrathink ultrathink_experiments > backup.sql

# Backup Grafana dashboards
curl -H "Authorization: Bearer <api-key>" http://localhost:3000/api/dashboards/export

# Backup Prometheus data
docker exec ultrathink-prometheus promtool tsdb snapshot /prometheus
```

**Recovery Procedures**:
1. Stop affected services
2. Restore from backup
3. Verify data integrity
4. Restart services
5. Validate dashboards

---

### F. Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-10-25 | 1.0 | Initial runbook creation | Infrastructure Team |

---

### G. Contact Information

**Team Distribution Lists**:
- Infrastructure Team: infrastructure@ultrathink.io
- On-Call Pager: oncall@ultrathink.io
- Incident Channel: #incidents (Slack)

**Vendor Support**:
- Grafana Labs: https://grafana.com/support
- Prometheus: https://prometheus.io/community
- TimescaleDB: https://www.timescale.com/support

---

### H. Related Documentation

- [Infrastructure Architecture](./README.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Service Documentation](../services/README.md)
- [Incident Response Template](./templates/incident_report.md)

---

**Document End**

*For questions or suggestions, please contact the Infrastructure Team or submit a PR to update this runbook.*

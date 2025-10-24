# UltraThink Infrastructure Runbook

**Version:** 1.0
**Last Updated:** 2025-10-25
**Maintainer:** Infrastructure Team

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Common Failure Scenarios](#common-failure-scenarios)
4. [Recovery Procedures](#recovery-procedures)
5. [Resource Monitoring](#resource-monitoring)
6. [Escalation Paths](#escalation-paths)
7. [Maintenance Procedures](#maintenance-procedures)
8. [Circuit Breaker Management](#circuit-breaker-management)

## Overview

This runbook provides operational guidance for the UltraThink trading system infrastructure. It covers common failure scenarios, recovery procedures, monitoring guidelines, and escalation paths.

### Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Inference Service (8080)                    │
│             [Circuit Breakers + GPU Enabled]                 │
└─┬────────┬──────────┬──────────┬──────────┬─────────────────┘
  │        │          │          │          │
  │        │          │          │          └─────┐
  │        │          │          │                │
┌─▼────┐ ┌▼─────┐ ┌──▼─────┐ ┌──▼──────┐ ┌──────▼──────┐
│ Data │ │Regime│ │  Meta   │ │  Risk   │ │   Kafka     │
│ Svc  │ │Detect│ │Controller│ │ Manager │ │ (3 brokers) │
│ 8000 │ │ 8001 │ │  8002   │ │  8003   │ │ 9092-9094   │
└──┬───┘ └──┬───┘ └────┬─────┘ └────┬────┘ └──────┬──────┘
   │        │          │            │             │
   │        └──────────┴────────────┴─────────┐   │
   │                                           │   │
┌──▼────────────────────────────────────────┐ │   │
│              Redis Cache                   │ │   │
│            (maxmemory: 2GB)                │ │   │
└────────────────┬───────────────────────────┘ │   │
                 │                             │   │
┌────────────────▼─────────────────────────────▼───▼──┐
│                  TimescaleDB                          │
│       (Experiment tracking + Time series data)        │
└───────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│              Monitoring Stack                          │
│  Prometheus (9090) → Grafana (3000)                   │
└───────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│              ML Infrastructure                         │
│  MLflow (5000) + Training Orchestrator (Celery)       │
└───────────────────────────────────────────────────────┘
```

### Resource Allocations

| Service | CPU Limit | Memory Limit | Memory Reserved | GPU |
|---------|-----------|--------------|-----------------|-----|
| TimescaleDB | 2.0 | 2G | 1G | No |
| MLflow | 1.0 | 1G | 512M | No |
| Redis | 1.0 | 2G | 512M | No |
| Kafka (each) | 1.0 | 1.5G | 1G | No |
| Zookeeper | 0.5 | 768M | 512M | No |
| Prometheus | 1.0 | 1G | 512M | No |
| Grafana | 0.5 | 512M | 256M | No |
| Data Service | 1.0 | 1G | 512M | No |
| Regime Detection | 1.0 | 1G | 512M | No |
| Meta-Controller | 2.0 | 8G | 4G | Yes (1) |
| Training Orchestrator | 4.0 | 8G | 4G | Yes (1) |
| Risk Manager | 1.0 | 1G | 512M | No |
| Inference Service | 2.0 | 2G | 1G | Yes (1) |
| Forensics Consumer | 1.0 | 1G | 512M | No |

**Total:** ~17 CPU cores, ~33GB memory, 3 GPUs (can share)

## Common Failure Scenarios

### 1. TimescaleDB Connection Loss

**Symptoms:**
- Services reporting database connection errors
- Health checks failing with `connection refused`
- PostgreSQL not responding to `pg_isready`

**Diagnosis:**
```bash
# Check if container is running
docker ps | grep timescaledb

# Check container logs
docker logs ultrathink-timescaledb --tail 100

# Test connection from host
docker exec ultrathink-timescaledb pg_isready -U ultrathink

# Check resource usage
docker stats ultrathink-timescaledb --no-stream

# Check connection count
docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c "SELECT count(*) FROM pg_stat_activity;"
```

**Recovery:**

**Level 1: Restart Service**
```bash
docker restart ultrathink-timescaledb
# Wait 30s for health check
docker exec ultrathink-timescaledb pg_isready -U ultrathink
```

**Level 2: Check Configuration**
```bash
# Verify max_connections not exceeded
docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c "SHOW max_connections;"

# Kill idle connections if needed
docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND state_change < NOW() - INTERVAL '10 minutes';"
```

**Level 3: Full Restart**
```bash
# Stop all dependent services
docker-compose stop data-service regime-detection risk-manager forensics-consumer

# Restart database
docker-compose restart timescaledb

# Wait for healthy
until docker exec ultrathink-timescaledb pg_isready -U ultrathink; do sleep 1; done

# Restart dependent services
docker-compose start data-service regime-detection risk-manager forensics-consumer
```

**Prevention:**
- Monitor connection count (alert at >80% of max_connections)
- Enable connection pooling in application services
- Review slow queries using pg_stat_statements
- Set appropriate `idle_in_transaction_session_timeout`

---

### 2. Kafka Broker Failure

**Symptoms:**
- Messages not being produced/consumed
- `kafka-broker-api-versions` command failing
- Inference service unable to publish predictions
- Forensics consumer not receiving events

**Diagnosis:**
```bash
# Check broker status
docker ps | grep kafka

# Check broker logs
docker logs ultrathink-kafka-1 --tail 100

# Verify Zookeeper connection
docker exec ultrathink-kafka-1 kafka-broker-api-versions --bootstrap-server localhost:9092

# Check topic status
docker exec ultrathink-kafka-1 kafka-topics --bootstrap-server kafka-1:9092 --list

# Check consumer groups
docker exec ultrathink-kafka-1 kafka-consumer-groups --bootstrap-server kafka-1:9092 --list
```

**Recovery:**

**Level 1: Restart Failed Broker**
```bash
# Identify failed broker
docker ps -a | grep kafka

# Restart specific broker
docker restart ultrathink-kafka-1  # or kafka-2, kafka-3

# Verify recovery
docker exec ultrathink-kafka-1 kafka-broker-api-versions --bootstrap-server localhost:9092
```

**Level 2: Check Zookeeper**
```bash
# Verify Zookeeper health
echo srvr | nc localhost 2181 | grep Mode

# Restart Zookeeper if needed
docker restart ultrathink-zookeeper

# Wait for Zookeeper to be ready
until echo ruok | nc localhost 2181 | grep imok; do sleep 1; done

# Restart Kafka brokers sequentially
docker restart ultrathink-kafka-1
sleep 30
docker restart ultrathink-kafka-2
sleep 30
docker restart ultrathink-kafka-3
```

**Level 3: Full Cluster Restart**
```bash
# Stop consumers
docker-compose stop forensics-consumer inference-service meta-controller

# Restart Kafka cluster
docker-compose restart kafka-1 kafka-2 kafka-3

# Wait for all brokers to be healthy
for i in {1..3}; do
  until docker exec ultrathink-kafka-$i kafka-broker-api-versions --bootstrap-server localhost:909$((i+1)); do
    sleep 2
  done
done

# Restart consumers
docker-compose start forensics-consumer inference-service meta-controller
```

**Prevention:**
- Monitor broker lag metrics
- Set up alerts for under-replicated partitions
- Monitor disk usage (Kafka stores logs on disk)
- Configure appropriate log retention policies

---

### 3. GPU Out of Memory (OOM)

**Symptoms:**
- Training jobs failing with CUDA OOM errors
- Inference service returning 500 errors
- Docker container exits with status 137
- `nvidia-smi` shows high memory usage

**Diagnosis:**
```bash
# Check GPU usage
nvidia-smi

# Check which containers are using GPU
docker ps --format '{{.Names}}' | xargs -I {} sh -c 'echo {}; docker exec {} nvidia-smi 2>/dev/null || echo "No GPU"'

# Check container memory limits
docker stats --no-stream | grep -E "(inference|meta|training)"

# Review container logs for OOM
docker logs ultrathink-inference-service --tail 100 | grep -i "memory\|oom\|cuda"
```

**Recovery:**

**Level 1: Restart Affected Service**
```bash
# Identify which service failed
docker ps -a | grep -E "(inference|meta|training)"

# Restart the service
docker restart ultrathink-inference-service  # or meta-controller, training-orchestrator

# Monitor GPU memory
watch -n 1 nvidia-smi
```

**Level 2: Reduce Batch Size**
```bash
# For inference service - set environment variable
docker exec ultrathink-inference-service env BATCH_SIZE=16

# For training - update Celery worker configuration
docker exec ultrathink-training-orchestrator env TRAINING_BATCH_SIZE=32

# Restart services to apply
docker restart ultrathink-inference-service ultrathink-training-orchestrator
```

**Level 3: Redistribute GPU Resources**
```bash
# Check current GPU allocation
docker inspect ultrathink-inference-service | grep -A 5 "DeviceRequests"

# Stop non-critical GPU services
docker stop ultrathink-meta-controller

# Restart critical service
docker restart ultrathink-inference-service

# Monitor and gradually bring back services
docker start ultrathink-meta-controller
```

**Prevention:**
- Set `CUDA_VISIBLE_DEVICES` to partition GPUs
- Implement gradient checkpointing for training
- Use mixed precision training (FP16)
- Monitor GPU memory with Prometheus
- Set up alerts at 80% GPU memory usage

---

### 4. Model Stability Failure (EWC Rollback)

**Symptoms:**
- Sudden drop in model performance metrics
- EWC penalty violations detected
- Training orchestrator triggering rollbacks
- MLflow showing failed experiment runs

**Diagnosis:**
```bash
# Check MLflow for recent experiments
curl -s http://localhost:5000/api/2.0/mlflow/experiments/search | jq '.experiments[] | {name, lifecycle_stage}'

# Check training orchestrator logs
docker logs ultrathink-training-orchestrator --tail 200 | grep -i "ewc\|rollback\|stability"

# Query TimescaleDB for recent metrics
docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c \
  "SELECT timestamp, experiment_id, metric_name, value FROM metrics WHERE metric_name = 'ewc_penalty' ORDER BY timestamp DESC LIMIT 10;"

# Check if rollback occurred
docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c \
  "SELECT * FROM model_versions WHERE rollback_triggered = true ORDER BY created_at DESC LIMIT 5;"
```

**Recovery:**

**Level 1: Validate Rollback**
```bash
# Check current active model
curl http://localhost:5000/api/2.0/mlflow/registered-models/get?name=production_model

# Verify model version in inference service
curl http://localhost:8080/model/info

# If incorrect, trigger manual rollback
curl -X POST http://localhost:5000/api/2.0/mlflow/model-versions/transition-stage \
  -d '{"name":"production_model","version":"PREVIOUS_VERSION","stage":"Production"}'

# Restart inference service to load correct model
docker restart ultrathink-inference-service
```

**Level 2: Investigate Root Cause**
```bash
# Check training hyperparameters
docker exec ultrathink-training-orchestrator env | grep -E "LEARNING_RATE|EWC|LAMBDA"

# Review training logs for anomalies
docker logs ultrathink-training-orchestrator --since 1h | grep -i "loss\|gradient"

# Check data quality
curl http://localhost:8000/health | jq '.checks.data_quality'
```

**Level 3: Retrain with Safe Parameters**
```bash
# Stop current training
docker exec ultrathink-training-orchestrator celery -A tasks control shutdown

# Update training configuration
# Edit docker-compose.yml to set conservative parameters:
# - LEARNING_RATE=0.0001 (reduced)
# - EWC_LAMBDA=1000 (increased stability)
# - BATCH_SIZE=64 (increased for stability)

# Restart training orchestrator
docker-compose restart training-orchestrator

# Monitor new training run
docker logs -f ultrathink-training-orchestrator
```

**Prevention:**
- Set conservative EWC lambda (start high, decrease slowly)
- Implement gradual learning rate decay
- Use validation set monitoring
- Set up automated rollback triggers
- Maintain model performance baselines

---

### 5. Redis Memory Exhaustion

**Symptoms:**
- Cache evictions increasing rapidly
- Services experiencing cache misses
- Redis reporting OOM errors
- Slow response times from data service

**Diagnosis:**
```bash
# Check Redis memory usage
docker exec ultrathink-redis redis-cli info memory

# Check eviction statistics
docker exec ultrathink-redis redis-cli info stats | grep evicted

# Check key count
docker exec ultrathink-redis redis-cli dbsize

# Check largest keys
docker exec ultrathink-redis redis-cli --bigkeys

# Monitor memory usage
docker stats ultrathink-redis --no-stream
```

**Recovery:**

**Level 1: Clear Non-Critical Keys**
```bash
# Flush specific database (careful!)
docker exec ultrathink-redis redis-cli -n 1 FLUSHDB  # Don't flush DB 0 (production cache)

# Clear expired keys manually
docker exec ultrathink-redis redis-cli --scan --pattern "temp:*" | xargs -L 1 docker exec ultrathink-redis redis-cli DEL

# Check memory after cleanup
docker exec ultrathink-redis redis-cli info memory | grep used_memory_human
```

**Level 2: Adjust Eviction Policy**
```bash
# Check current policy
docker exec ultrathink-redis redis-cli config get maxmemory-policy

# Temporarily increase memory limit (if host has capacity)
docker exec ultrathink-redis redis-cli config set maxmemory 3gb

# Restart with new configuration
docker-compose restart redis
```

**Level 3: Scale Redis**
```bash
# Option A: Add Redis Sentinel for HA
# Update docker-compose.yml to add Redis Sentinel

# Option B: Implement Redis Cluster
# Requires application changes to support cluster mode

# For now: Increase memory limit in docker-compose.yml
# Edit infrastructure/docker-compose.yml:
# redis:
#   command: redis-server --maxmemory 4gb ...

docker-compose up -d redis
```

**Prevention:**
- Set appropriate TTLs on cached data
- Monitor eviction rate (alert if >1000/sec)
- Use memory-efficient data structures
- Implement cache warming strategies
- Consider Redis Cluster for horizontal scaling

---

### 6. Disk Space Exhaustion

**Symptoms:**
- Docker failing to write logs
- MLflow artifact uploads failing
- Kafka unable to accept new messages
- TimescaleDB checkpoint failures

**Diagnosis:**
```bash
# Check overall disk usage
df -h

# Check Docker volumes
docker system df -v

# Check largest volumes
docker volume ls -q | xargs docker volume inspect | jq -r '.[] | "\(.Mountpoint)\t\(.Name)"' | xargs -I {} sh -c 'du -sh {}' 2>/dev/null

# Check container log sizes
docker ps -q | xargs docker inspect --format='{{.Name}}: {{.LogPath}}' | xargs -I {} sh -c 'du -h {}' 2>/dev/null

# Check specific directories
du -sh /var/lib/docker/volumes/*
```

**Recovery:**

**Level 1: Clean Docker System**
```bash
# Remove stopped containers
docker container prune -f

# Remove dangling images
docker image prune -f

# Remove unused volumes (CAREFUL!)
docker volume prune -f

# Clean build cache
docker builder prune -f
```

**Level 2: Clean Service-Specific Data**
```bash
# Clean old MLflow artifacts (use checkpoint cleanup script)
cd /home/rich/ultrathink-pilot
python scripts/checkpoint_cleanup.py --dry-run
python scripts/checkpoint_cleanup.py  # If dry-run looks safe

# Clean old logs (keep last 7 days)
find /var/lib/docker/containers -name "*.log" -mtime +7 -exec truncate -s 0 {} \;

# Clean Kafka logs (if retention not working)
docker exec ultrathink-kafka-1 kafka-delete-records --bootstrap-server localhost:9092 --offset-json-file /tmp/delete-records.json
```

**Level 3: Emergency Space Recovery**
```bash
# Stop non-critical services
docker-compose stop grafana prometheus

# Clean Prometheus data (reduces monitoring history)
docker run --rm -v prometheus_data:/data alpine sh -c "cd /data && rm -rf wal snapshots chunks_head"

# Compact Kafka logs
docker exec ultrathink-kafka-1 kafka-log-dirs --bootstrap-server localhost:9092 --broker-list 1,2,3 --describe

# Restart services
docker-compose start grafana prometheus
```

**Prevention:**
- Set up disk usage monitoring (alert at 80%)
- Implement automated cleanup scripts (cron)
- Configure log rotation for Docker containers
- Set MLflow artifact retention policies
- Monitor Kafka log retention settings
- Use external storage for large artifacts

---

### 7. Circuit Breaker Open (Service Degradation)

**Symptoms:**
- Services returning `CircuitBreakerError`
- Health check endpoint showing circuit breakers in OPEN state
- Cascading failures across services

**Diagnosis:**
```bash
# Check health endpoints for circuit breaker states
curl http://localhost:8080/health | jq '.circuit_breakers'
curl http://localhost:8000/health | jq '.circuit_breakers'

# Check service logs for circuit breaker events
docker logs ultrathink-inference-service | grep -i "circuit\|breaker"

# Check failure counts
docker logs ultrathink-data-service --tail 100 | grep -c "failure\|error"
```

**Recovery:**

**Level 1: Identify Root Cause**
```bash
# Check which service is causing failures
curl http://localhost:8080/health | jq '.circuit_breakers[] | select(.state == "open")'

# Check upstream service health
curl http://localhost:8000/health  # Data Service
curl http://localhost:8001/health  # Regime Detection
curl http://localhost:8003/health  # Risk Manager

# Check network connectivity
docker exec ultrathink-inference-service ping -c 3 data-service
```

**Level 2: Restart Failing Service**
```bash
# Restart the service with issues
docker restart ultrathink-data-service

# Wait for health check to pass
until curl -sf http://localhost:8000/health; do sleep 2; done

# Circuit breaker should auto-recover to CLOSED state after successful calls
```

**Level 3: Manual Circuit Reset (If Needed)**
```bash
# If circuit doesn't auto-recover, restart dependent service
docker restart ultrathink-inference-service

# Monitor circuit breaker recovery
watch -n 2 'curl -s http://localhost:8080/health | jq ".circuit_breakers"'
```

**Prevention:**
- Monitor circuit breaker metrics
- Set appropriate failure thresholds
- Implement fallback mechanisms
- Add retry logic with backoff
- Test circuit breakers regularly

---

## Resource Monitoring

### Key Metrics to Monitor

#### System-Level Metrics
```bash
# CPU usage by container
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Memory usage
docker stats --no-stream --format "table {{.Name}}\t{{.MemPerc}}\t{{.MemUsage}}"

# Disk I/O
docker stats --no-stream --format "table {{.Name}}\t{{.BlockIO}}"

# Network I/O
docker stats --no-stream --format "table {{.Name}}\t{{.NetIO}}"
```

#### Service-Specific Metrics

**TimescaleDB:**
```bash
# Connection count
docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c \
  "SELECT count(*) as connections FROM pg_stat_activity;"

# Database size
docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c \
  "SELECT pg_size_pretty(pg_database_size('ultrathink_experiments'));"

# Slow queries
docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c \
  "SELECT query, calls, total_time, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

**Kafka:**
```bash
# Under-replicated partitions
docker exec ultrathink-kafka-1 kafka-topics --bootstrap-server localhost:9092 --describe --under-replicated-partitions

# Consumer lag
docker exec ultrathink-kafka-1 kafka-consumer-groups --bootstrap-server localhost:9092 --describe --group forensics-consumer
```

**Redis:**
```bash
# Hit rate
docker exec ultrathink-redis redis-cli info stats | grep keyspace_hits

# Memory fragmentation
docker exec ultrathink-redis redis-cli info memory | grep fragmentation

# Connected clients
docker exec ultrathink-redis redis-cli client list | wc -l
```

**MLflow:**
```bash
# Artifact storage size
du -sh /var/lib/docker/volumes/infrastructure_mlflow_artifacts/_data

# Number of experiments
curl -s http://localhost:5000/api/2.0/mlflow/experiments/search | jq '.experiments | length'

# Recent runs
curl -s 'http://localhost:5000/api/2.0/mlflow/runs/search' | jq '.runs | length'
```

### Alerting Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Disk Usage | 80% | 90% |
| Memory Usage | 85% | 95% |
| CPU Usage (sustained) | 80% | 95% |
| GPU Memory | 80% | 95% |
| TimescaleDB Connections | 80% of max | 95% of max |
| Kafka Under-Replicated Partitions | >0 | >10 |
| Redis Memory | 80% | 95% |
| Circuit Breaker Open | Any | Multiple |
| Model Rollback | 1/day | >3/day |

### Prometheus Queries

```promql
# Container CPU usage
rate(container_cpu_usage_seconds_total{name=~"ultrathink-.*"}[5m])

# Container memory usage
container_memory_usage_bytes{name=~"ultrathink-.*"}

# Service availability
up{job=~"ultrathink-.*"}

# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m])

# Circuit breaker state
circuit_breaker_state{name=~".*"}
```

---

## Escalation Paths

### Level 1: On-Call Engineer (15 min response time)
**Handles:**
- Service restarts
- Configuration changes
- Basic troubleshooting
- Monitoring alert triage

**Actions:**
- Check runbook for known issues
- Restart failed services
- Verify health checks
- Document incident

**Escalate if:**
- Issue persists after 30 minutes
- Data loss suspected
- Multiple services failing
- Unknown root cause

### Level 2: Infrastructure Lead (30 min response time)
**Handles:**
- Complex troubleshooting
- System-wide issues
- Performance optimization
- Capacity planning decisions

**Actions:**
- Deep diagnostics
- Architecture changes
- Coordinate with application teams
- Implement workarounds

**Escalate if:**
- Security incident
- Data corruption
- Production trading halted
- Regulatory implications

### Level 3: Engineering Manager + CTO (1 hour response time)
**Handles:**
- Critical incidents
- Business impact decisions
- External communications
- Post-mortem coordination

**Actions:**
- Incident command
- Stakeholder communication
- Resource allocation
- Go/no-go decisions

---

## Maintenance Procedures

### Daily Tasks

**Automated (via cron):**
```bash
# 02:00 - Checkpoint cleanup
0 2 * * * cd /home/rich/ultrathink-pilot && python scripts/checkpoint_cleanup.py >> /var/log/checkpoint_cleanup.log 2>&1

# 03:00 - Log rotation
0 3 * * * find /var/lib/docker/containers -name "*.log" -mtime +7 -exec truncate -s 0 {} \;

# Every 6 hours - Health check verification
0 */6 * * * docker-compose ps | grep -v "Up (healthy)" | mail -s "Unhealthy services" ops@ultrathink.com
```

**Manual checks:**
- Review Grafana dashboards
- Check for anomalous patterns
- Verify backup completion
- Review circuit breaker stats

### Weekly Tasks

**Sunday 01:00 - System Maintenance:**
```bash
#!/bin/bash
# Weekly maintenance script

# 1. Clean Docker system
docker system prune -f --volumes

# 2. Restart non-critical services (avoid downtime)
docker-compose restart grafana prometheus

# 3. Vacuum TimescaleDB
docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c "VACUUM ANALYZE;"

# 4. Check disk usage trends
df -h > /var/log/disk_usage_$(date +%Y%m%d).log

# 5. Review error logs
docker-compose logs --tail=1000 | grep -i error > /var/log/weekly_errors_$(date +%Y%m%d).log

# 6. Test backups
./scripts/test_backup_restore.sh

# 7. Update monitoring dashboards if needed
./scripts/update_grafana_dashboards.sh
```

### Monthly Tasks

- Review and update runbook
- Capacity planning review
- Security patch updates
- Disaster recovery drill
- Performance baseline updates

---

## Circuit Breaker Management

### Circuit Breaker Configuration

Each service implements circuit breakers for external dependencies:

**Default Settings:**
- Failure threshold: 5 failures in 60 seconds
- Timeout: 60 seconds (circuit stays open)
- Retry strategy: Exponential backoff (1s, 2s, 4s)

### Managing Circuit Breakers

**Check Status:**
```bash
# Via health endpoint
curl http://localhost:8080/health | jq '.circuit_breakers'

# Expected response:
# {
#   "data_service": {
#     "state": "closed",
#     "failure_count": 0,
#     "success_count": 1523
#   }
# }
```

**Manual Reset (if stuck):**
```python
# Connect to service
docker exec -it ultrathink-inference-service python

# In Python REPL:
from common_utils.circuit_breaker import get_circuit_breaker
cb = get_circuit_breaker("data_service")
cb.reset()
```

**Adjust Thresholds (if needed):**
```python
# In service code (requires restart):
from common_utils.circuit_breaker import circuit_breaker

@circuit_breaker(
    name="custom_service",
    failure_threshold=10,  # Increased tolerance
    timeout=120  # Longer recovery window
)
def call_flaky_service():
    ...
```

---

## Appendix

### Useful Commands

```bash
# Quick health check all services
docker-compose ps

# View all logs
docker-compose logs -f --tail=100

# Restart all services
docker-compose restart

# Check resource usage
docker stats

# Clean up everything (DESTRUCTIVE)
docker-compose down -v

# Backup volumes
docker run --rm -v infrastructure_timescale_data:/data -v $(pwd):/backup alpine tar czf /backup/timescaledb_backup_$(date +%Y%m%d).tar.gz /data
```

### Contact Information

**On-Call Rotation:** ops@ultrathink.com
**Infrastructure Lead:** infrastructure@ultrathink.com
**Emergency Escalation:** +1-XXX-XXX-XXXX

### Related Documentation

- [Deployment Plan](deployment-plan.md)
- [Monitoring Dashboard Guide](GRAFANA_QUICKSTART.md)
- [ML Persistence Setup](ML_PERSISTENCE_SETUP.md)
- [Docker Quickstart](DOCKER.md)

---

**End of Runbook**

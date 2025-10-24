# UltraThink Pilot - Infrastructure

This directory contains Docker Compose configurations and deployment files for the UltraThink Pilot infrastructure.

## Components

- **TimescaleDB**: Time-series PostgreSQL database for experiment tracking
- **MLflow**: Machine learning experiment tracking and model registry
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization and dashboards
- **Redis**: In-memory caching for features (Phase 2)

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set your passwords
nano .env
```

### 2. Start Infrastructure

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f
```

### 3. Verify Services

- **TimescaleDB**: `psql -h localhost -U ultrathink -d ultrathink_experiments`
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Redis**: `redis-cli -h localhost ping`

### 4. Stop Infrastructure

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

## Database Schema

The TimescaleDB schema is automatically initialized from `timescale_schema.sql` on first startup. It includes:

- `experiments` - Main experiment tracking table
- `experiment_metrics` - Time-series metrics (hypertable)
- `model_checkpoints` - Model versioning
- `regime_history` - Regime detection history
- `dataset_versions` - Dataset versioning
- `trading_decisions` - Trading audit trail

## Monitoring

### Prometheus Metrics

Access Prometheus at http://localhost:9090 to query metrics:

- Training pipeline metrics
- Model performance metrics
- System resource metrics
- Cache hit rates

### Grafana Dashboards

Access Grafana at http://localhost:3000:

1. Login with admin/admin (or your configured password)
2. Prometheus datasource is pre-configured
3. Create custom dashboards for:
   - Training progress
   - Model performance comparison
   - System resource utilization
   - Cache performance

## Maintenance

### Backup Database

```bash
# Backup TimescaleDB
docker exec ultrathink-timescaledb pg_dump -U ultrathink ultrathink_experiments > backup.sql

# Backup MLflow artifacts
docker cp ultrathink-mlflow:/mlflow/artifacts ./mlflow_backup
```

### Restore Database

```bash
# Restore TimescaleDB
docker exec -i ultrathink-timescaledb psql -U ultrathink ultrathink_experiments < backup.sql
```

### Clean Up Old Data

TimescaleDB retention policies are configured to:
- Keep experiment_metrics for 90 days
- Keep trading_decisions for 365 days
- Compress data older than 7/30 days

Manual cleanup:
```sql
-- Connect to database
\c ultrathink_experiments

-- Check table sizes
SELECT hypertable_name, total_size
FROM timescaledb_information.hypertables;

-- Manually drop old data
DELETE FROM experiment_metrics
WHERE time < NOW() - INTERVAL '90 days';
```

## Troubleshooting

### TimescaleDB Connection Issues

```bash
# Check if container is running
docker ps | grep timescaledb

# View logs
docker logs ultrathink-timescaledb

# Restart container
docker-compose restart timescaledb
```

### MLflow Not Starting

```bash
# Check dependencies
docker-compose logs mlflow

# Ensure TimescaleDB is healthy
docker-compose ps timescaledb

# Restart MLflow
docker-compose restart mlflow
```

### Port Conflicts

If ports are already in use, edit `docker-compose.yml`:

```yaml
ports:
  - "5433:5432"  # Change host port
```

## Performance Tuning

### TimescaleDB

Edit `docker-compose.yml` to add PostgreSQL tuning:

```yaml
command: >
  postgres -c shared_preload_libraries=timescaledb
  -c shared_buffers=2GB
  -c effective_cache_size=6GB
  -c maintenance_work_mem=512MB
  -c checkpoint_completion_target=0.9
  -c wal_buffers=16MB
  -c default_statistics_target=100
  -c random_page_cost=1.1
  -c effective_io_concurrency=200
  -c work_mem=32MB
  -c min_wal_size=1GB
  -c max_wal_size=4GB
```

### Redis

Adjust memory settings in `docker-compose.yml`:

```yaml
command: redis-server --maxmemory 4gb --maxmemory-policy allkeys-lru
```

## Security Notes

- Change default passwords in `.env` before production use
- Use Docker secrets for sensitive credentials in production
- Enable SSL/TLS for database connections in production
- Restrict network access with firewall rules
- Regularly update Docker images for security patches

## Next Steps

After infrastructure is running:

1. Run migration script: `python scripts/migrate_sqlite_to_timescale.py`
2. Verify data migration: Check TimescaleDB tables
3. Update application configs to use TimescaleDB
4. Set up Grafana dashboards for monitoring
5. Configure alerts in Prometheus

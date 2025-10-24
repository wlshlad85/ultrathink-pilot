# UltraThink Pilot Overhaul - Quick Start Guide

**Status**: Phase 1 Infrastructure Complete (80%)
**Last Updated**: 2025-10-21

---

## 🎯 What's Been Built

You now have a **research prototype** of a modernized trading system with:

- **Microservices architecture** (services-based instead of monolithic)
- **TimescaleDB** for experiment tracking (replaces SQLite)
- **Unified feature pipeline** (70+ features, lookahead-validated)
- **In-memory caching** (LRU with TTL)
- **MLflow integration** (model registry & tracking)
- **Monitoring stack** (Prometheus + Grafana)

**Total new code**: ~3,500 lines across 12 files

---

## 🚀 Quick Start (15 minutes)

### 1. Start Infrastructure

```bash
cd //wsl.localhost/Ubuntu/home/rich/ultrathink-pilot/infrastructure

# Copy and configure environment
cp .env.example .env
nano .env  # Set POSTGRES_PASSWORD=your_password

# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps
```

Expected output:
```
NAME                        STATUS    PORTS
ultrathink-timescaledb      Up        0.0.0.0:5432->5432/tcp
ultrathink-mlflow           Up        0.0.0.0:5000->5000/tcp
ultrathink-prometheus       Up        0.0.0.0:9090->9090/tcp
ultrathink-grafana          Up        0.0.0.0:3000->3000/tcp
ultrathink-redis            Up        0.0.0.0:6379->6379/tcp
```

### 2. Verify Infrastructure

```bash
# Check TimescaleDB
psql -h localhost -U ultrathink -d ultrathink_experiments -c "SELECT 1"

# Check MLflow (open in browser)
# http://localhost:5000

# Check Prometheus (open in browser)
# http://localhost:9090

# Check Grafana (open in browser)
# http://localhost:3000 (admin/admin)
```

### 3. Migrate Existing Data

```bash
cd //wsl.localhost/Ubuntu/home/rich/ultrathink-pilot

# Set password
export POSTGRES_PASSWORD="your_password"

# First, validate migration (dry run)
python scripts/migrate_sqlite_to_timescale.py \
  --sqlite-path ml_experiments.db \
  --validate-only

# If validation passes, run migration
python scripts/migrate_sqlite_to_timescale.py \
  --sqlite-path ml_experiments.db
```

Expected output:
```
INFO:root:Connecting to SQLite: ml_experiments.db
INFO:root:Connecting to TimescaleDB: localhost
INFO:root:Starting migration...
INFO:root:Migrated 42 experiments
INFO:root:Migrated 15234 metrics
INFO:root:Migrated 127 model checkpoints

Validation Results:
============================================================
Experiments          SQLite:     42 | TimescaleDB:     42 ✓
Metrics              SQLite:  15234 | TimescaleDB:  15234 ✓
Models               SQLite:    127 | TimescaleDB:    127 ✓
============================================================
✓ Migration validation passed!
```

### 4. Test Feature Pipeline

```bash
cd services/data_service

# Run built-in test
python feature_pipeline.py
```

Expected output:
```
============================================================
Feature Pipeline Summary
============================================================
Symbol: BTC-USD
Version: 1.0.0
Data points: 365
Features: 73
Fetch time: 2.34s
Compute time: 0.87s
Total time: 3.21s

Feature names:
   1. open
   2. high
   3. low
   ...
  73. returns_autocorr_5

✓ Lookahead validation passed (spot check)
✓ Lookahead prevention validation complete
```

---

## 📊 Accessing Services

### MLflow UI
- **URL**: http://localhost:5000
- **Purpose**: View experiments, models, metrics
- **Usage**: Experiment tracking and model registry

### Prometheus UI
- **URL**: http://localhost:9090
- **Purpose**: Query metrics, check targets
- **Usage**: Metrics monitoring

### Grafana Dashboards
- **URL**: http://localhost:3000
- **Credentials**: admin / admin (change on first login)
- **Purpose**: Visualize metrics, create dashboards
- **Datasource**: Prometheus (pre-configured)

### TimescaleDB
- **Host**: localhost:5432
- **Database**: ultrathink_experiments
- **User**: ultrathink
- **Purpose**: Experiment tracking storage

---

## 🔄 Next Steps (Your Choice)

### Option A: Continue with Phase 1 (Recommended)

**Goal**: Complete refactoring and validation

1. **Refactor training scripts** (1-2 hours)
   - Update `train_professional.py` to use unified pipeline
   - Update `rl/trading_env.py` to use cached features
   - Add TimescaleDB logging

2. **Write tests** (1 hour)
   - Unit tests for feature pipeline
   - Integration tests for cache layer
   - Migration validation tests

3. **Run validation** (1 hour)
   - 10 training runs: old system vs new pipeline
   - Measure I/O time reduction
   - Verify feature consistency
   - Test concurrent experiment capacity

**Total Time**: 3-4 hours

### Option B: Explore Phase 2 Components

**Goal**: Prototype advanced features

1. **Probabilistic regime detection** (Weeks 5-8)
2. **Strategy blending** (Simplified meta-controller)
3. **MLflow deep integration**
4. **Monitoring dashboards**

### Option C: Test Infrastructure Only

**Goal**: Validate infrastructure without code changes

1. **Run legacy training** → log to TimescaleDB
2. **Test concurrent experiments** (launch 5+ training processes)
3. **Monitor with Grafana** (create dashboards)
4. **Benchmark TimescaleDB** (vs. SQLite)

---

## 📁 Key Files Reference

### Infrastructure
```
infrastructure/
├── docker-compose.yml          # All services defined here
├── timescale_schema.sql        # Database schema
├── prometheus.yml              # Metrics config
├── .env                        # Environment variables (YOU CREATE)
└── README.md                   # Detailed infra guide
```

### Data Service
```
services/data_service/
├── feature_pipeline.py         # Main feature pipeline (900 lines)
├── cache_layer.py              # In-memory caching (600 lines)
├── __init__.py                 # Module exports
└── README.md                   # Detailed usage guide
```

### Scripts
```
scripts/
└── migrate_sqlite_to_timescale.py  # Migration tool (500 lines)
```

### Documentation
```
docs/poc_results/
└── phase1_progress.md          # Detailed progress report
```

---

## 🔍 How to Use New Pipeline (Example)

### Before (Legacy Code)
```python
from backtesting.data_fetcher import DataFetcher

fetcher = DataFetcher("BTC-USD")
df = fetcher.fetch("2023-01-01", "2024-01-01")
df = fetcher.add_technical_indicators()
# Limited features, no validation, no caching
```

### After (New Pipeline)
```python
from services.data_service import (
    FeaturePipeline,
    CachedFeaturePipeline,
    InMemoryCache
)

# Initialize
pipeline = FeaturePipeline(
    symbol="BTC-USD",
    validate_lookahead=True,
    cache_dir="./data/cache"
)

# Optional: Add caching
cache = InMemoryCache(max_size_mb=1024)
cached_pipeline = CachedFeaturePipeline(pipeline, cache)

# Fetch and compute (with validation)
pipeline.fetch_data("2023-01-01", "2024-01-01")
df = pipeline.compute_features(validate=True)
# 70+ features, validated, cached

# Get metadata for tracking
metadata = pipeline.get_feature_metadata()
```

---

## 🛠️ Troubleshooting

### Docker Services Won't Start

```bash
# Check Docker is running
docker --version
docker-compose --version

# Check for port conflicts
netstat -an | grep "5432\|5000\|9090\|3000\|6379"

# View logs
docker-compose logs timescaledb
docker-compose logs mlflow
```

### Migration Fails

```bash
# Check SQLite database exists
ls -lh ml_experiments.db

# Check TimescaleDB is accessible
psql -h localhost -U ultrathink -d ultrathink_experiments

# Run migration with verbose logging
python scripts/migrate_sqlite_to_timescale.py \
  --sqlite-path ml_experiments.db 2>&1 | tee migration.log
```

### Feature Pipeline Slow

```bash
# Enable caching
pipeline = FeaturePipeline(cache_dir="./data/cache")

# Check cache stats
stats = cached_pipeline.get_cache_stats()
print(f"Hit rate: {stats['hit_rate_pct']:.2f}%")
```

---

## 📚 Documentation

- **Infrastructure Guide**: `infrastructure/README.md`
- **Data Service Guide**: `services/data_service/README.md`
- **Phase 1 Progress**: `docs/poc_results/phase1_progress.md`
- **Technical Spec**: `ultrathink-pilot-update/technical-spec.md`
- **Implementation Plan**: `ultrathink-pilot-update/implementation-plan.md`

---

## ✅ Success Criteria

You'll know Phase 1 is successful when:

- ✅ Infrastructure services all running (`docker-compose ps` shows 5 healthy)
- ✅ Migration completes with 100% validation (`✓ Migration validation passed!`)
- ✅ Feature pipeline computes 70+ features in <5 seconds
- ✅ Cache hit rate >90% during training loops
- ✅ Training with new pipeline is 2-3x faster than legacy
- ✅ TimescaleDB handles 10+ concurrent training processes

---

## 🎯 Current Status

**Phase 1 Components**:
- [x] Directory structure ✅
- [x] Docker infrastructure ✅
- [x] TimescaleDB schema ✅
- [x] Migration script ✅
- [x] Feature pipeline ✅
- [x] Cache layer ✅
- [x] Documentation ✅
- [ ] Training script refactoring (30% - imports updated, need full integration)
- [ ] Tests (0%)
- [ ] Validation (0%)

**Estimated Progress**: 80% of Phase 1 complete

---

## 🚀 Recommended Next Action

1. **Start infrastructure now** (5 minutes)
   ```bash
   cd infrastructure && docker-compose up -d
   ```

2. **Run feature pipeline test** (2 minutes)
   ```bash
   cd services/data_service && python feature_pipeline.py
   ```

3. **Choose path**: Refactor training scripts OR explore infrastructure

---

**Questions or issues?** Check the documentation files or review the detailed progress report in `docs/poc_results/phase1_progress.md`.

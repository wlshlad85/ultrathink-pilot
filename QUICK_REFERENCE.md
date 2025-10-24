# UltraThink Pilot - Quick Reference

**Last Updated:** October 22, 2025
**Phase:** 1 - COMPLETED ✅

---

## 🚀 Quick Start

### Start Infrastructure

```bash
cd ~/ultrathink-pilot/infrastructure
docker compose up -d
```

### Run Training

```bash
cd ~/ultrathink-pilot
source venv/bin/activate
python train_professional_v2.py --episodes 1000
```

### View Results

```bash
python view_results.py
```

---

## 🔗 Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana** | http://localhost:3000 | admin / admin |
| **Prometheus** | http://localhost:9090 | None |
| **TimescaleDB** | localhost:5432 | ultrathink / [.env] |
| **Redis** | localhost:6379 | None |

---

## 📊 Latest Training Results (Experiment 14)

```
Average Return:  +1.25%
Best Episode:    +2.58% (Episode 6)
Win Rate:        80% (8/10 episodes)
Model Saved:     rl/models/professional_v2/ppo_agent_final.pth
```

---

## 🛠️ Common Commands

### Infrastructure Management

```bash
# Start all services
docker compose up -d

# Check status
docker ps

# View logs
docker logs ultrathink-timescaledb
docker logs ultrathink-grafana

# Stop services
docker compose down

# Restart a service
docker restart ultrathink-grafana
```

### Database Queries

```bash
# Connect to TimescaleDB
docker exec -it ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments

# Query experiments
docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c "SELECT id, experiment_name, status FROM experiments ORDER BY id DESC LIMIT 5;"

# Check metrics count
docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c "SELECT COUNT(*) FROM experiment_metrics;"
```

### Training Scripts

```bash
# 10-episode quick test
python train_professional_v2.py --episodes 10

# Full 1000-episode training
python train_professional_v2.py --episodes 1000

# View experiment results
python view_results.py
```

---

## 📁 Important Files

```
PHASE_1_COMPLETION_REPORT.md    - Comprehensive completion report
GRAFANA_QUICKSTART.md           - Grafana setup guide
QUICK_REFERENCE.md              - This file
ml_experiments.db               - SQLite experiment database
training_final_test.log         - Latest training output
```

---

## ⚡ Phase 2 Next Steps

1. **Run full 1,000-episode training**
   ```bash
   python train_professional_v2.py --episodes 1000
   ```

2. **Create Grafana dashboards**
   - Training performance metrics
   - System resource monitoring
   - Trading decision analysis

3. **Validate on 2022 data**
   - Out-of-sample testing
   - Regime-specific performance

4. **Hyperparameter optimization**
   - Learning rate tuning
   - PPO clip ratio adjustment
   - Update frequency optimization

---

## 🔧 Troubleshooting

### Services won't start

```bash
# Check Docker is running
docker ps

# View service logs
docker logs [container_name]

# Restart infrastructure
cd ~/ultrathink-pilot/infrastructure
docker compose down
docker compose up -d
```

### Training fails

```bash
# Check GPU
nvidia-smi

# Verify virtual environment
which python
python --version

# Check dependencies
pip list | grep torch
```

### Database connection issues

```bash
# Test connection
docker exec ultrathink-timescaledb pg_isready -U ultrathink

# Check password in .env
cat infrastructure/.env | grep POSTGRES_PASSWORD
```

---

## 📚 Documentation

- **Full Report:** `PHASE_1_COMPLETION_REPORT.md`
- **Grafana Guide:** `GRAFANA_QUICKSTART.md`
- **Architecture:** See README.md (main project)

---

## 🎯 Success Metrics Achieved

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Infrastructure | 5 services | 5 deployed | ✅ |
| Training Works | Yes | +1.25% avg | ✅ |
| GPU Enabled | Yes | CUDA active | ✅ |
| Data Migration | Complete | 100% | ✅ |
| Monitoring | Operational | Grafana + Prometheus | ✅ |

---

**Phase 1 Status: ✅ COMPLETE - Ready for Phase 2**

---

*For detailed information, see PHASE_1_COMPLETION_REPORT.md*

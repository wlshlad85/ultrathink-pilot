# 🚀 Run Your Tracked Professional Training

## ✅ Setup Complete - Ready to Train!

Your `train_professional.py` now has full ML tracking integrated and tested!

**Verified:**
- ✅ Experiment #8 created successfully
- ✅ Database tracking working
- ✅ Git commit captured: master@89d21367
- ✅ Tags applied: rl, ppo, bitcoin, professional, multi-regime, institutional

---

## 🎯 Start Training (3 Options)

### Option 1: Foreground (Recommended for First Run)

See all output in real-time:

```bash
cd ~/ultrathink-pilot
source .venv/bin/activate
python3 train_professional.py
```

**Best for:** Watching first few episodes to confirm everything works

---

### Option 2: Background with nohup

Training continues even if you close terminal:

```bash
cd ~/ultrathink-pilot
source .venv/bin/activate

# Start in background
nohup python3 train_professional.py > training_output.log 2>&1 &

# Save PID
echo $! > training.pid

# Confirm it started
tail -f training_output.log
# Press Ctrl+C to stop watching (training continues)
```

**Best for:** Long training runs (hours)

---

### Option 3: Screen/tmux Session

Start in detachable session:

```bash
cd ~/ultrathink-pilot
source .venv/bin/activate

# Create screen session
screen -S training

# Inside screen, run training
python3 train_professional.py

# Detach: Press Ctrl+A then D
# Reattach later: screen -r training
```

**Best for:** Maximum control and flexibility

---

## 📊 Monitor Training Progress

### Watch Real-Time Output

If running in background:
```bash
tail -f ~/ultrathink-pilot/training_output.log
```

### Check Latest Experiment Status

```bash
cd ~/ultrathink-pilot
source .venv/bin/activate

python3 -c "
from ml_persistence import ExperimentTracker
t = ExperimentTracker()
exps = t.list_experiments(limit=1)
exp = exps[0]
print(f'Experiment {exp[\"id\"]}: {exp[\"name\"]}')
print(f'Status: {exp[\"status\"]}')
print(f'Started: {exp[\"start_time\"]}')
"
```

### Count Metrics Logged

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('ml_experiments.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM metrics WHERE experiment_id IN (SELECT id FROM experiments WHERE status=\"running\")')
count = cursor.fetchone()[0]
print(f'Metrics logged so far: {count}')
print(f'Estimated episodes completed: {count // 6}')
"
```

### View Latest Episode Metrics

```bash
python3 -c "
import sqlite3
import pandas as pd
conn = sqlite3.connect('ml_experiments.db')
df = pd.read_sql_query('''
    SELECT episode, metric_name, metric_value
    FROM metrics
    WHERE experiment_id = (SELECT MAX(id) FROM experiments WHERE status=\"running\")
    ORDER BY episode DESC, metric_name
    LIMIT 12
''', conn)
print(df)
"
```

### Watch Progress (Auto-Update Every 30 Seconds)

```bash
watch -n 30 'python3 -c "
import sqlite3
conn = sqlite3.connect(\"ml_experiments.db\")
cursor = conn.cursor()
cursor.execute(\"SELECT MAX(episode) FROM metrics WHERE experiment_id = (SELECT MAX(id) FROM experiments WHERE status=\047running\047)\")
result = cursor.fetchone()[0]
if result:
    print(f\"Episodes completed: {result} / 1000\")
    print(f\"Progress: {result/10:.1f}%\")
else:
    print(\"Starting...\")
"'
```

---

## 📈 Training Timeline

**Expected Duration:** 4-8 hours (depending on GPU)

| Milestone | Episodes | Checkpoints Saved | Estimated Time |
|-----------|----------|-------------------|----------------|
| Early Training | 0-100 | 2 checkpoints | ~30-60 min |
| Learning Phase | 100-500 | 8 checkpoints | 2-4 hours |
| Convergence | 500-1000 | 10 checkpoints | 2-4 hours |
| **Total** | **1000** | **20 checkpoints** | **4-8 hours** |

**Plus:** Validation & Test evaluation at end (~10 minutes)

---

## 🔍 What Gets Tracked

Every episode (all 1000) logs:
- Episode reward
- Portfolio return %
- Episode length
- Final portfolio value
- Rolling averages (10 & 100 episodes)

**Total:** ~6,000 individual metrics!

Plus:
- 12 hyperparameters
- 3 dataset registrations (train/val/test)
- ~20 model checkpoints
- Validation metrics
- Test metrics

---

## 🎯 After Training Completes

### View Final Results

```bash
python3 -c "
import sqlite3
import pandas as pd
conn = sqlite3.connect('ml_experiments.db')
df = pd.read_sql_query('''
    SELECT
        name,
        best_train_return,
        best_val_metric,
        duration_seconds / 3600.0 as hours,
        num_checkpoints
    FROM experiment_summary
    WHERE id = (SELECT MAX(id) FROM experiments)
''', conn)
print(df)
"
```

### Analyze Training Curve

```bash
python3 -c "
from ml_persistence import MetricsLogger
import pandas as pd
import matplotlib.pyplot as plt

m = MetricsLogger()
metrics = m.get_metrics_for_experiment(8)  # Use your experiment ID
df = pd.DataFrame(metrics)

# Get returns
returns = df[df['metric_name'] == 'train_return'].sort_values('episode')

print(f'Total episodes: {len(returns)}')
print(f'Best return: {returns[\"metric_value\"].max():.2f}%')
print(f'Final 100 avg: {returns.tail(100)[\"metric_value\"].mean():.2f}%')
"
```

### Find Best Models

```bash
python3 -c "
from ml_persistence import ModelRegistry
r = ModelRegistry()
models = r.list_models(is_best=True, order_by='train_metric', limit=5)
for m in models:
    print(f'{m[\"name\"]}: {m[\"train_metric\"]:.2f}% at episode {m[\"episode_num\"]}')
"
```

---

## ⚠️ Stop Training if Needed

### If running in foreground:
- Press `Ctrl+C` (training state saved to database)

### If running in background:
```bash
# Find PID
cat ~/ultrathink-pilot/training.pid

# Or find by name
ps aux | grep train_professional

# Stop gracefully
kill <PID>

# Force stop if needed
kill -9 <PID>
```

**Note:** All episodes completed before stopping are safely in the database!

---

## 🐛 Troubleshooting

### Training Not Starting?
```bash
# Check for errors
python3 -m py_compile train_professional.py

# Test imports
python3 -c "import train_professional; print('OK')"

# Check GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### No Metrics Being Logged?
```bash
# Check experiment status
python3 -c "
from ml_persistence import ExperimentTracker
t = ExperimentTracker()
exps = t.list_experiments(limit=3)
for exp in exps:
    print(f'{exp[\"id\"]}: {exp[\"name\"]} - {exp[\"status\"]}')
"

# Check database
sqlite3 ml_experiments.db "SELECT COUNT(*) FROM metrics;"
```

### Training Crashed?
```bash
# Check last experiment
python3 -c "
import sqlite3
conn = sqlite3.connect('ml_experiments.db')
cursor = conn.cursor()
cursor.execute('SELECT id, name, status, start_time, end_time FROM experiments ORDER BY id DESC LIMIT 1')
print(cursor.fetchone())
"

# Mark as failed if needed
python3 -c "
from ml_persistence import ExperimentTracker
t = ExperimentTracker()
# Will need to set current experiment first
"
```

---

## 🎓 Pro Tips

1. **Monitor GPU Usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Estimate Completion Time:**
   - First 10 episodes take ~5-10 minutes
   - Multiply by 100 for total estimate
   - Typically 4-8 hours for 1000 episodes

3. **Backup Database During Training:**
   ```bash
   cp ml_experiments.db ml_experiments_backup.db
   ```

4. **Query Training Progress:**
   ```bash
   # Every 60 seconds, show progress
   while true; do
     python3 -c "
import sqlite3
conn = sqlite3.connect('ml_experiments.db')
cursor = conn.cursor()
cursor.execute('SELECT MAX(episode) FROM metrics')
ep = cursor.fetchone()[0]
if ep:
    print(f'Episodes: {ep}/1000 ({ep/10:.1f}%)')
"
     sleep 60
   done
   ```

---

## 📚 References

- **Full Integration Docs:** `ML_TRACKING_INTEGRATION_COMPLETE.md`
- **ML Persistence Guide:** `ml_persistence/README.md`
- **Integration Guide:** `INTEGRATION_GUIDE.md`
- **Working Example:** `examples/train_with_ml_tracking.py`

---

**Ready to train! Your experiment will be fully tracked and reproducible! 🚀**

Generated: October 20, 2025

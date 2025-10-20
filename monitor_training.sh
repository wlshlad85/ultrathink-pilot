#!/bin/bash
# Real-time training monitor

cd ~/ultrathink-pilot
source .venv/bin/activate

while true; do
    clear
    python3 << 'PYEOF'
import sqlite3
import datetime

conn = sqlite3.connect("ml_experiments.db")
cursor = conn.cursor()

cursor.execute("SELECT id, name, start_time FROM experiments WHERE status='running' ORDER BY id DESC LIMIT 1")
exp = cursor.fetchone()

if exp:
    exp_id, name, start_time = exp
    start_dt = datetime.datetime.fromisoformat(start_time)
    elapsed = datetime.datetime.now() - start_dt

    cursor.execute("SELECT COUNT(*) FROM metrics WHERE experiment_id = ?", (exp_id,))
    metric_count = cursor.fetchone()[0]

    cursor.execute("SELECT MAX(episode) FROM metrics WHERE experiment_id = ?", (exp_id,))
    latest_ep = cursor.fetchone()[0] or 0

    cursor.execute("SELECT value FROM metrics WHERE experiment_id = ? AND metric_name = 'train_return' ORDER BY episode DESC LIMIT 1", (exp_id,))
    result = cursor.fetchone()
    latest_return = result[0] if result else 0

    cursor.execute("""
        SELECT AVG(value) FROM metrics
        WHERE experiment_id = ? AND metric_name = 'train_return'
        AND episode > ? - 10
    """, (exp_id, latest_ep))
    avg_10 = cursor.fetchone()[0] or 0

    if latest_ep > 0:
        seconds_per_episode = elapsed.total_seconds() / latest_ep
        remaining_episodes = 1000 - latest_ep
        remaining_seconds = seconds_per_episode * remaining_episodes
        eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_seconds)
    else:
        eta = None

    print("=" * 80)
    print("🚀 PROFESSIONAL TRAINING MONITOR")
    print("=" * 80)
    print()
    print(f"Experiment:   {name}")
    print(f"ID:           {exp_id}")
    print(f"Started:      {start_time}")
    print(f"Elapsed:      {str(elapsed).split('.')[0]}")
    print()
    print("=" * 80)
    print("PROGRESS")
    print("=" * 80)
    print()
    print(f"Episodes:     {latest_ep:4d} / 1000  [{latest_ep/10:5.1f}%]")
    print(f"Metrics:      {metric_count:,}")
    print()
    progress = int(latest_ep / 10)
    bar = "█" * progress + "░" * (100 - progress)
    print(f"[{bar}]")
    print()
    print("=" * 80)
    print("PERFORMANCE")
    print("=" * 80)
    print()
    print(f"Latest Return:      {latest_return:+7.2f}%")
    print(f"Avg Last 10:        {avg_10:+7.2f}%")
    print()
    if eta:
        print("=" * 80)
        print("TIMING")
        print("=" * 80)
        print()
        print(f"Time per Episode:   {seconds_per_episode:.1f}s")
        print(f"Estimated Finish:   {eta.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Est. Total Time:    {str(datetime.timedelta(seconds=int(seconds_per_episode * 1000))).split('.')[0]}")
        print()
    print("=" * 80)
    print(f"Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to exit (training continues)")
    print("=" * 80)
else:
    print("No running experiments found")

conn.close()
PYEOF

    sleep 10
done

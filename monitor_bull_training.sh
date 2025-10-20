#!/bin/bash
# Monitor bull specialist training progress

LOG_FILE="/tmp/bull_training.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Training log not found at $LOG_FILE"
    echo "Training may not have started yet."
    exit 1
fi

echo "=" | tr '=' '=' | head -c 80
echo
echo "BULL SPECIALIST TRAINING MONITOR"
echo "=" | tr '=' '=' | head -c 80
echo
echo

# Check if training is running
if ps aux | grep 'python.*train_bull' | grep -v grep > /dev/null; then
    echo "✅ Training process is RUNNING"
    PID=$(ps aux | grep 'python.*train_bull' | grep -v grep | awk '{print $2}')
    echo "   PID: $PID"
else
    echo "⚠️  Training process not found"
    echo "   Either it hasn't started or has completed"
fi

echo
echo "Latest training output (last 50 lines):"
echo "------------------------------------------------------------------------"
tail -50 "$LOG_FILE" | grep -E "(Episode|VALIDATION|BEST|STOPPING|COMPLETE)" || tail -50 "$LOG_FILE"
echo "------------------------------------------------------------------------"
echo
echo "To see full log: cat /tmp/bull_training.log"
echo "To see live updates: tail -f /tmp/bull_training.log"

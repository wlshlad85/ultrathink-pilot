#!/bin/bash
# Start bull specialist training in a persistent way

cd ~/ultrathink-pilot
source .venv/bin/activate

# Clear old log
> /tmp/bull_training.log

echo "Starting bull specialist training..."
echo "Log file: /tmp/bull_training.log"
echo "Monitor with: bash monitor_bull_training.sh"
echo

# Run with nohup to persist even if terminal closes
nohup python -u train_bull_specialist.py >> /tmp/bull_training.log 2>&1 &
PID=$!

echo "Training started with PID: $PID"
echo "Waiting 10 seconds to verify startup..."
sleep 10

if ps -p $PID > /dev/null; then
    echo "✅ Training is running successfully!"
    echo
    echo "To monitor progress, run:"
    echo "  bash monitor_bull_training.sh"
    echo
    echo "Or view live log:"
    echo "  tail -f /tmp/bull_training.log"
else
    echo "❌ Training failed to start. Check log:"
    tail -20 /tmp/bull_training.log
    exit 1
fi

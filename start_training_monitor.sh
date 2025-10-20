#!/bin/bash
# Check training status and show quick stats
# Lightweight script for quick checks

clear
echo "════════════════════════════════════════════════════════════════════════════"
echo "                    REGIME-AWARE V2 TRAINING STATUS"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if training is running
if pgrep -f "train_regime_aware_v2.py" > /dev/null; then
    PID=$(pgrep -f "train_regime_aware_v2.py")
    echo "✅ Training is RUNNING (PID: $PID)"

    # Check runtime
    RUNTIME=$(ps -p $PID -o etime= | xargs)
    echo "   Runtime: $RUNTIME"

    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "GPU Status:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader | \
        awk -F', ' '{printf "  GPU %s (%s): %s%% | %sMB / %sMB | %s°C\n", $1, $2, $3, $4, $5, $6}'
    fi

    # Show recent progress from log
    echo ""
    echo "Recent Progress:"
    tail -50 ~/ultrathink-pilot/training_regime_aware_v2.log | \
    grep -E "Episode [0-9]+/[0-9]+" | tail -5 | \
    while read line; do
        echo "  $line"
    done

    # Check for validation results
    echo ""
    echo "Recent Validation:"
    tail -100 ~/ultrathink-pilot/training_regime_aware_v2.log | \
    grep -E "Weighted.*return" | tail -3 | \
    while read line; do
        echo "  $line"
    done

    echo ""
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "For live monitoring, run:"
    echo "  bash ~/ultrathink-pilot/monitor_training.sh"
    echo ""

else
    echo "⚠️  Training is NOT running"
    echo ""

    # Check if it completed
    if [ -f ~/ultrathink-pilot/rl/models/regime_aware_v2/final_model.pth ]; then
        echo "✅ Training appears to have COMPLETED"
        echo "   Final model found at: rl/models/regime_aware_v2/final_model.pth"
        echo ""
        echo "To evaluate the model, run:"
        echo "  bash ~/ultrathink-pilot/run_evaluation.sh"
    else
        echo "Training has not been started or crashed."
        echo ""
        echo "Last 20 lines of log:"
        tail -20 ~/ultrathink-pilot/training_regime_aware_v2.log 2>/dev/null || echo "  (No log file found)"
    fi
fi

echo "════════════════════════════════════════════════════════════════════════════"

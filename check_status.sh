#!/bin/bash
# Quick status check for all three experiments

echo "=================================="
echo "EXPERIMENT STATUS CHECK"
echo "Time: $(date +%H:%M:%S)"
echo "=================================="
echo

# Check processes
echo "Running Processes:"
ps -ef | grep 'python.*train_exp' | grep -v grep | awk '{print "  " $8 " (PID " $2 ")"}'
echo

# Check log sizes
echo "Log File Progress:"
for i in 1 2 3; do
    if [ -f /tmp/exp${i}.log ]; then
        lines=$(wc -l < /tmp/exp${i}.log)
        echo "  exp${i}: ${lines} lines"
    else
        echo "  exp${i}: No log yet"
    fi
done
echo

# Check for results
echo "Completed Experiments:"
for exp in exp1_strong exp2_exp exp3_sharpe; do
    if [ -f ~/ultrathink-pilot/rl/models/${exp}/training_metrics.json ]; then
        best_sharpe=$(python3 -c "import json; print(json.load(open('~/ultrathink-pilot/rl/models/${exp}/training_metrics.json'))['best_val_sharpe'])" 2>/dev/null || echo "N/A")
        echo "  ✅ ${exp}: Best Sharpe = ${best_sharpe}"
    fi
done

# Check monitor
if ps -ef | grep -q '[m]onitor_experiments.py'; then
    echo
    echo "Monitor: ✅ Running"
    if [ -f /tmp/monitor.log ]; then
        echo "Last monitor output:"
        tail -5 /tmp/monitor.log | sed 's/^/  /'
    fi
else
    echo
    echo "Monitor: ❌ Not running"
fi

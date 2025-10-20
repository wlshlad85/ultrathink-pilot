#!/bin/bash
# Evaluate the best model once training completes
# Run this in a separate dedicated terminal window

clear
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                     REGIME-AWARE V2 EVALUATION                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

cd ~/ultrathink-pilot
source .venv/bin/activate

# Check if best model exists
if [ ! -f "rl/models/regime_aware_v2/best_model.pth" ]; then
    echo "❌ Best model not found!"
    echo "   Training may not have completed yet."
    echo ""
    echo "Waiting for training to produce a model..."
    while [ ! -f "rl/models/regime_aware_v2/best_model.pth" ]; do
        sleep 10
        echo -n "."
    done
    echo ""
    echo "✅ Model found! Starting evaluation..."
    sleep 2
fi

echo "Running comprehensive evaluation on test periods..."
echo ""

# Run the deep analysis
python analyze_regime_aware.py

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                          EVALUATION COMPLETE                               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Press Enter to exit..."
read

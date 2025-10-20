#!/bin/bash
# Quick test script for RL setup

echo "========================================"
echo "Testing UltraThink RL Setup"
echo "========================================"
echo ""

cd /home/rich/ultrathink-pilot
source .venv/bin/activate

echo "1. Testing PyTorch and CUDA..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

echo "2. Testing Gymnasium..."
python3 -c "import gymnasium; print(f'Gymnasium version: {gymnasium.__version__}')"
echo ""

echo "3. Testing TradingEnv..."
python3 rl/trading_env.py
echo ""

echo "========================================"
echo "Setup test complete!"
echo "========================================"

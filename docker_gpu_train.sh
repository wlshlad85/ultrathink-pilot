#!/bin/bash
# GPU-accelerated training script for Docker

# Default values
EPISODES=100
SYMBOL="BTC-USD"
START_DATE="2023-01-01"
END_DATE="2024-01-01"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --symbol)
      SYMBOL="$2"
      shift 2
      ;;
    --start)
      START_DATE="$2"
      shift 2
      ;;
    --end)
      END_DATE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--episodes NUM] [--symbol SYMBOL] [--start DATE] [--end DATE]"
      exit 1
      ;;
  esac
done

echo "======================================"
echo "GPU-Accelerated RL Training"
echo "======================================"
echo "Episodes:    $EPISODES"
echo "Symbol:      $SYMBOL"
echo "Period:      $START_DATE to $END_DATE"
echo "======================================"
echo ""

docker run --rm --gpus all \
  ultrathink-pilot:gpu \
  python rl/train.py \
    --episodes "$EPISODES" \
    --symbol "$SYMBOL" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE"

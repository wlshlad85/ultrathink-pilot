#!/bin/bash
set -e

echo "Starting Forensics Consumer and API..."

# Start consumer in background
python forensics_consumer.py &
CONSUMER_PID=$!

# Wait for consumer to initialize
sleep 5

# Start API in foreground
python forensics_api.py &
API_PID=$!

# Trap signals and cleanup
cleanup() {
    echo "Shutting down..."
    kill $CONSUMER_PID $API_PID 2>/dev/null || true
    wait
}

trap cleanup SIGTERM SIGINT

# Wait for either process to exit
wait -n

# Cleanup
cleanup
exit $?

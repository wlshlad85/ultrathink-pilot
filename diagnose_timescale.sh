#!/bin/bash
# Diagnose TimescaleDB Issues
# Run with: sudo bash diagnose_timescale.sh

echo "=========================================="
echo "TimescaleDB Diagnostic"
echo "=========================================="
echo ""

# Detect user home
if [ -n "$SUDO_USER" ]; then
    USER_HOME=$(eval echo ~$SUDO_USER)
else
    USER_HOME=$HOME
fi

PROJECT_DIR="$USER_HOME/ultrathink-pilot"
cd "$PROJECT_DIR/infrastructure"

echo "[1/5] Checking TimescaleDB logs..."
echo ""
docker logs ultrathink-timescaledb 2>&1 | tail -50
echo ""

echo "[2/5] Checking for port conflicts..."
netstat -tlnp 2>/dev/null | grep :5432 || echo "Port 5432 is available"
echo ""

echo "[3/5] Checking Docker volumes..."
docker volume ls | grep ultrathink
echo ""

echo "[4/5] Checking container status..."
docker ps -a | grep timescaledb
echo ""

echo "[5/5] Attempting to start TimescaleDB alone..."
docker compose up -d timescaledb
sleep 10
docker logs ultrathink-timescaledb 2>&1 | tail -30
echo ""

echo "=========================================="
echo "Diagnostic Complete"
echo "=========================================="

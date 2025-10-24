#!/bin/bash
# Start All Infrastructure Services
if [ -n "" ]; then
    USER_HOME=/home/rich
else
    USER_HOME=/home/rich
fi
cd "/ultrathink-pilot/infrastructure"
echo "=========================================="
echo "Starting All Services"
echo "=========================================="
echo ""
echo "Starting Docker Compose (all services)..."
docker compose up -d
echo ""
echo "Waiting 30 seconds for initialization..."
sleep 30
echo ""
echo "Service Status:"
docker compose ps
echo ""
echo "Testing connections:"
echo -n "TimescaleDB: "
docker exec ultrathink-timescaledb pg_isready -U ultrathink 2>&1 | grep -q accepting && echo "✓ Ready" || echo "✗ Not ready"
echo -n "Redis: "
docker exec ultrathink-redis redis-cli ping 2>&1 | grep -q PONG && echo "✓ Ready" || echo "✗ Not ready"
echo ""
echo "Services available at:"
echo "  MLflow:      http://localhost:5000"
echo "  Grafana:     http://localhost:3000 (admin/admin)"
echo "  Prometheus:  http://localhost:9090"
echo "  TimescaleDB: localhost:5432"
echo "  Redis:       localhost:6379"
echo ""
echo "Next: bash ~/ultrathink-pilot/migrate_data.sh"

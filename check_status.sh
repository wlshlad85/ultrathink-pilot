#!/bin/bash
# Quick Status Check
if [ -n "" ]; then
    USER_HOME=/home/rich
else
    USER_HOME=/home/rich
fi
cd "/ultrathink-pilot/infrastructure"
echo "=========================================="
echo "Service Status Check"
echo "=========================================="
echo ""
echo "Docker Containers:"
docker compose ps
echo ""
echo "TimescaleDB:"
docker exec ultrathink-timescaledb pg_isready -U ultrathink && echo "✓ Ready" || echo "✗ Not ready"
echo ""
echo "Redis:"
docker exec ultrathink-redis redis-cli ping && echo "✓ Ready" || echo "✗ Not ready"
echo ""
echo "Services:"
echo "  MLflow:      http://localhost:5000"
echo "  Grafana:     http://localhost:3000"
echo "  Prometheus:  http://localhost:9090"

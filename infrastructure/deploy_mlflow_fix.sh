#!/bin/bash
# ============================================================================
# UltraThink Pilot - MLflow Deployment Script
# Purpose: Deploy MLflow with psycopg2 fix and validate TimescaleDB
# ============================================================================

set -e  # Exit on error

echo "========================================"
echo "UltraThink Pilot - MLflow Deployment"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to infrastructure directory
cd "$(dirname "$0")"

echo "Step 1: Building MLflow container with psycopg2..."
echo "----------------------------------------------"
docker compose build mlflow
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ MLflow image built successfully${NC}"
else
    echo -e "${RED}✗ MLflow image build failed${NC}"
    exit 1
fi
echo ""

echo "Step 2: Starting TimescaleDB (if not running)..."
echo "----------------------------------------------"
docker compose up -d timescaledb
echo "Waiting for TimescaleDB to be healthy..."
for i in {1..30}; do
    if docker exec ultrathink-timescaledb pg_isready -U ultrathink > /dev/null 2>&1; then
        echo -e "${GREEN}✓ TimescaleDB is ready${NC}"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

echo "Step 3: Applying continuous aggregate enhancements..."
echo "----------------------------------------------"
if [ -f "timescale_continuous_aggregates.sql" ]; then
    docker exec -i ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments < timescale_continuous_aggregates.sql
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Continuous aggregates applied${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: Some continuous aggregates may already exist (this is OK)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Warning: timescale_continuous_aggregates.sql not found${NC}"
fi
echo ""

echo "Step 4: Starting MLflow service..."
echo "----------------------------------------------"
docker compose up -d mlflow
echo "Waiting for MLflow to be healthy..."
for i in {1..60}; do
    if docker exec ultrathink-mlflow curl -f http://localhost:5000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ MLflow is ready${NC}"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

echo "Step 5: Validating MLflow connection to TimescaleDB..."
echo "----------------------------------------------"
docker logs ultrathink-mlflow --tail 50 | grep -i "error\|exception\|psycopg2" || echo -e "${GREEN}✓ No errors found in MLflow logs${NC}"
echo ""

echo "Step 6: Running database validation..."
echo "----------------------------------------------"
if [ -f "validate_database.sql" ]; then
    docker exec -i ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments < validate_database.sql
else
    echo -e "${YELLOW}⚠ Warning: validate_database.sql not found, skipping validation${NC}"
fi
echo ""

echo "Step 7: Service Status Summary..."
echo "----------------------------------------------"
docker compose ps
echo ""

echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo ""
echo "Access Points:"
echo "  MLflow UI:       http://localhost:5000"
echo "  Grafana:         http://localhost:3000"
echo "  Prometheus:      http://localhost:9090"
echo "  TimescaleDB:     postgresql://localhost:5432/ultrathink_experiments"
echo ""
echo "Logs:"
echo "  MLflow:          docker logs ultrathink-mlflow"
echo "  TimescaleDB:     docker logs ultrathink-timescaledb"
echo ""
echo "Validation:"
echo "  Run: docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -f /validate_database.sql"
echo ""
echo "========================================"

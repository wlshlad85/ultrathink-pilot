#!/bin/bash
# Quick Infrastructure Deployment
# Run with: cd ~/ultrathink-pilot && sudo bash quick_deploy.sh

set -e

echo "=== Starting Docker and Deploying Infrastructure ==="
echo ""

# Start Docker
echo "[1/7] Starting Docker service..."
service docker start
sleep 2

# Verify Docker
echo "[2/7] Verifying Docker installation..."
docker run hello-world
docker compose version

# Navigate to infrastructure directory
echo "[3/7] Preparing environment..."
cd /home/rich/ultrathink-pilot/infrastructure

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "[4/7] Creating .env file..."
    cp .env.example .env
    echo "WARNING: Using default passwords! Edit .env for production."
else
    echo "[4/7] Using existing .env file"
fi

# Start infrastructure
echo "[5/7] Starting infrastructure services..."
docker compose up -d

# Wait for services
echo "[6/7] Waiting for services to initialize (30 seconds)..."
sleep 30

# Check status
echo "[7/7] Checking service status..."
docker compose ps

echo ""
echo "=== Deployment Complete! ==="
echo ""
echo "Services available at:"
echo "  - MLflow:      http://localhost:5000"
echo "  - Grafana:     http://localhost:3000 (admin/admin)"
echo "  - Prometheus:  http://localhost:9090"
echo "  - TimescaleDB: localhost:5432"
echo "  - Redis:       localhost:6379"
echo ""
echo "Next steps:"
echo "  1. Run migration: python scripts/migrate_sqlite_to_timescale.py"
echo "  2. Test training: python train_professional_v2.py --episodes 10"
echo ""

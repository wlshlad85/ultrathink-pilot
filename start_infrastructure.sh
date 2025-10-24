#!/bin/bash
# Start Infrastructure Services
# Run with: sudo bash start_infrastructure.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Detect user's home
if [ -n "$SUDO_USER" ]; then
    USER_HOME=$(eval echo ~$SUDO_USER)
else
    USER_HOME=$HOME
fi

PROJECT_DIR="$USER_HOME/ultrathink-pilot"
INFRA_DIR="$PROJECT_DIR/infrastructure"

echo "=========================================="
echo "Starting UltraThink Infrastructure"
echo "=========================================="
echo "Project: $PROJECT_DIR"
echo ""

# Check Docker is running
echo -e "${YELLOW}[1/5] Checking Docker service...${NC}"
if ! service docker status > /dev/null 2>&1; then
    echo "Starting Docker service..."
    service docker start
    sleep 3
fi
echo -e "${GREEN}✓ Docker is running${NC}"
echo ""

# Navigate to infrastructure
cd "$INFRA_DIR"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}[2/5] Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Note: Using default passwords. Edit .env for production use.${NC}"
else
    echo -e "${GREEN}[2/5] .env file exists${NC}"
fi
echo ""

# Stop any existing containers
echo -e "${YELLOW}[3/5] Cleaning up old containers...${NC}"
docker compose down 2>/dev/null || true
echo ""

# Start services
echo -e "${YELLOW}[4/5] Starting Docker Compose services...${NC}"
echo "This may take 30-60 seconds for first-time image pulls..."
echo ""
docker compose up -d

echo ""
echo -e "${YELLOW}[5/5] Waiting for services to initialize (45 seconds)...${NC}"
sleep 45

echo ""
echo -e "${GREEN}=========================================="
echo "Checking Service Status"
echo "==========================================${NC}"
echo ""
docker compose ps

echo ""
echo -e "${GREEN}=========================================="
echo "Infrastructure Started!"
echo "==========================================${NC}"
echo ""
echo "Services should be accessible at:"
echo "  - MLflow:      http://localhost:5000"
echo "  - Grafana:     http://localhost:3000 (admin/admin)"
echo "  - Prometheus:  http://localhost:9090"
echo ""
echo "Next steps:"
echo "  1. Verify: sudo bash $PROJECT_DIR/verify_services.sh"
echo "  2. Migrate: bash $PROJECT_DIR/migrate_data.sh"
echo ""

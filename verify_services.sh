#!/bin/bash
# Service Verification Script (Smart path detection)
# Run with: sudo bash verify_services.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Detect the actual user's home directory
if [ -n "$SUDO_USER" ]; then
    REAL_USER=$SUDO_USER
    USER_HOME=$(eval echo ~$SUDO_USER)
else
    REAL_USER=$USER
    USER_HOME=$HOME
fi

PROJECT_DIR="$USER_HOME/ultrathink-pilot"

echo "=========================================="
echo "Infrastructure Verification"
echo "=========================================="
echo "User: $REAL_USER"
echo "Project: $PROJECT_DIR"
echo ""

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}Error: Project directory not found at $PROJECT_DIR${NC}"
    exit 1
fi

# Navigate to infrastructure
cd "$PROJECT_DIR/infrastructure"

# Check services
echo -e "${YELLOW}[1/5] Checking Docker services...${NC}"
docker compose ps
echo ""

# Verify TimescaleDB
echo -e "${YELLOW}[2/5] Verifying TimescaleDB...${NC}"
if docker exec ultrathink-timescaledb pg_isready -U ultrathink > /dev/null 2>&1; then
    echo -e "${GREEN}✓ TimescaleDB is ready${NC}"

    # Show tables
    echo ""
    echo "Database tables:"
    docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c "\dt" 2>/dev/null | grep -E "public|hypertable" || echo "Schema initializing..."
else
    echo -e "${RED}✗ TimescaleDB is not ready${NC}"
fi
echo ""

# Verify Redis
echo -e "${YELLOW}[3/5] Verifying Redis...${NC}"
if docker exec ultrathink-redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Redis is ready${NC}"
else
    echo -e "${RED}✗ Redis is not ready${NC}"
fi
echo ""

# Check SQLite databases
echo -e "${YELLOW}[4/5] Checking SQLite databases...${NC}"
cd "$PROJECT_DIR"
if ls *.db 1> /dev/null 2>&1; then
    ls -lh *.db
    echo ""
    echo -e "${GREEN}✓ Found SQLite databases ready for migration${NC}"
else
    echo -e "${YELLOW}No .db files found (migration may not be needed)${NC}"
fi
echo ""

# Service URLs
echo -e "${YELLOW}[5/5] Service Access Information${NC}"
echo ""
echo "  MLflow:      http://localhost:5000"
echo "  Grafana:     http://localhost:3000 (admin/admin)"
echo "  Prometheus:  http://localhost:9090"
echo ""

echo -e "${GREEN}=========================================="
echo "Verification Complete!"
echo "==========================================${NC}"
echo ""
echo "Next step: Run migration"
echo "  bash $PROJECT_DIR/run_migration.sh"
echo ""

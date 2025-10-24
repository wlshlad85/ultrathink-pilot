#!/bin/bash
# Verify Infrastructure and Migrate Data
# Run with: cd ~/ultrathink-pilot && sudo bash verify_and_migrate.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Infrastructure Verification & Migration"
echo "=========================================="
echo ""

# Check services
echo -e "${YELLOW}[1/6] Checking Docker services...${NC}"
cd /home/rich/ultrathink-pilot/infrastructure
docker compose ps
echo ""

# Verify TimescaleDB
echo -e "${YELLOW}[2/6] Verifying TimescaleDB...${NC}"
docker exec ultrathink-timescaledb pg_isready -U ultrathink
echo -e "${GREEN}✓ TimescaleDB is ready${NC}"
echo ""

# Check schema
echo -e "${YELLOW}[3/6] Checking database schema...${NC}"
docker exec -it ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c "\dt"
echo ""

# Verify Redis
echo -e "${YELLOW}[4/6] Verifying Redis...${NC}"
docker exec ultrathink-redis redis-cli ping
echo -e "${GREEN}✓ Redis is ready${NC}"
echo ""

# Check SQLite databases
echo -e "${YELLOW}[5/6] Checking SQLite databases...${NC}"
cd /home/rich/ultrathink-pilot
ls -lh *.db 2>/dev/null || echo "No .db files found"
echo ""

# Prepare migration
echo -e "${YELLOW}[6/6] Preparing for migration...${NC}"
echo ""
echo "SQLite databases found:"
echo "  - ml_experiments.db"
echo "  - experiments.db"
echo ""
echo "To migrate data, you need to:"
echo "  1. Get the PostgreSQL password from .env file"
echo "  2. Run the migration script"
echo ""
echo "Run this command to see the password:"
echo "  grep POSTGRES_PASSWORD infrastructure/.env"
echo ""
echo "Then run migration with:"
echo "  cd ~/ultrathink-pilot"
echo "  source venv/bin/activate"
echo "  python scripts/migrate_sqlite_to_timescale.py \\"
echo "    --sqlite ml_experiments.db \\"
echo "    --postgres-host localhost \\"
echo "    --postgres-port 5432 \\"
echo "    --postgres-db ultrathink_experiments \\"
echo "    --postgres-user ultrathink \\"
echo "    --postgres-password <PASSWORD_FROM_ENV>"
echo ""
echo "=========================================="
echo "Service URLs:"
echo "=========================================="
echo "  MLflow:      http://localhost:5000"
echo "  Grafana:     http://localhost:3000"
echo "  Prometheus:  http://localhost:9090"
echo ""

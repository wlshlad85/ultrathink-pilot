#!/bin/bash
# Automated Migration Script
# Run with: cd ~/ultrathink-pilot && bash run_migration.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "SQLite to TimescaleDB Migration"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/migrate_sqlite_to_timescale.py" ]; then
    echo -e "${RED}Error: Migration script not found${NC}"
    echo "Please run from ~/ultrathink-pilot directory"
    exit 1
fi

# Check if SQLite database exists
if [ ! -f "ml_experiments.db" ]; then
    echo -e "${YELLOW}Warning: ml_experiments.db not found${NC}"
    echo "Skipping migration (no data to migrate)"
    exit 0
fi

# Read password from .env
if [ ! -f "infrastructure/.env" ]; then
    echo -e "${RED}Error: infrastructure/.env not found${NC}"
    exit 1
fi

echo -e "${YELLOW}[1/4] Reading configuration...${NC}"
POSTGRES_PASSWORD=$(grep '^POSTGRES_PASSWORD=' infrastructure/.env | cut -d '=' -f2)

if [ -z "$POSTGRES_PASSWORD" ]; then
    echo -e "${RED}Error: POSTGRES_PASSWORD not found in .env${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Configuration loaded${NC}"
echo ""

# Activate virtual environment
echo -e "${YELLOW}[2/4] Activating Python environment...${NC}"
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${RED}Warning: venv not found, using system Python${NC}"
fi
echo ""

# Check Python dependencies
echo -e "${YELLOW}[3/4] Checking dependencies...${NC}"
python -c "import psycopg2" 2>/dev/null || {
    echo -e "${YELLOW}Installing psycopg2...${NC}"
    pip install psycopg2-binary
}
echo -e "${GREEN}✓ Dependencies ready${NC}"
echo ""

# Run migration
echo -e "${YELLOW}[4/4] Running migration...${NC}"
echo ""

python scripts/migrate_sqlite_to_timescale.py \
    --sqlite ml_experiments.db \
    --postgres-host localhost \
    --postgres-port 5432 \
    --postgres-db ultrathink_experiments \
    --postgres-user ultrathink \
    --postgres-password "$POSTGRES_PASSWORD"

echo ""
echo -e "${GREEN}=========================================="
echo "Migration Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Verify migration: sudo bash verify_and_migrate.sh"
echo "  2. Run training test: python train_professional_v2.py --episodes 10"
echo "  3. View MLflow: http://localhost:5000"
echo ""

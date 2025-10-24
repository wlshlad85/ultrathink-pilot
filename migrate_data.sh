#!/bin/bash
# Automated Migration Script (No sudo required)
# Run with: bash migrate_data.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "SQLite to TimescaleDB Migration"
echo "=========================================="
echo ""

PROJECT_DIR="$HOME/ultrathink-pilot"

# Check if we're in the right directory
if [ ! -f "$PROJECT_DIR/scripts/migrate_sqlite_to_timescale.py" ]; then
    echo -e "${RED}Error: Migration script not found${NC}"
    echo "Expected: $PROJECT_DIR/scripts/migrate_sqlite_to_timescale.py"
    exit 1
fi

# Check if SQLite database exists
if [ ! -f "$PROJECT_DIR/ml_experiments.db" ]; then
    echo -e "${YELLOW}Warning: ml_experiments.db not found${NC}"
    echo "Skipping migration (no data to migrate)"
    echo ""
    echo -e "${GREEN}Infrastructure is ready for new experiments!${NC}"
    exit 0
fi

# Read password from .env
if [ ! -f "$PROJECT_DIR/infrastructure/.env" ]; then
    echo -e "${RED}Error: infrastructure/.env not found${NC}"
    exit 1
fi

echo -e "${YELLOW}[1/4] Reading configuration...${NC}"
POSTGRES_PASSWORD=$(grep '^POSTGRES_PASSWORD=' "$PROJECT_DIR/infrastructure/.env" | cut -d '=' -f2)

if [ -z "$POSTGRES_PASSWORD" ]; then
    echo -e "${RED}Error: POSTGRES_PASSWORD not found in .env${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Configuration loaded${NC}"
echo ""

# Navigate to project
cd "$PROJECT_DIR"

# Activate virtual environment
echo -e "${YELLOW}[2/4] Activating Python environment...${NC}"
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}Warning: venv not found, using system Python${NC}"
fi
echo ""

# Check Python dependencies
echo -e "${YELLOW}[3/4] Checking dependencies...${NC}"
python -c "import psycopg2" 2>/dev/null || {
    echo -e "${YELLOW}Installing psycopg2-binary...${NC}"
    pip install psycopg2-binary -q
}
echo -e "${GREEN}✓ Dependencies ready${NC}"
echo ""

# Run migration
echo -e "${YELLOW}[4/4] Running migration...${NC}"
echo ""

python scripts/migrate_sqlite_to_timescale.py \
    --sqlite-path ml_experiments.db \
    --pg-host localhost \
    --pg-port 5432 \
    --pg-database ultrathink_experiments \
    --pg-user ultrathink \
    --pg-password "$POSTGRES_PASSWORD"

MIGRATION_STATUS=$?

if [ $MIGRATION_STATUS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Migration Complete!"
    echo "==========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run training: python train_professional_v2.py --episodes 10"
    echo "  2. View MLflow: http://localhost:5000"
    echo "  3. View Grafana: http://localhost:3000"
    echo ""
else
    echo ""
    echo -e "${RED}Migration failed with status $MIGRATION_STATUS${NC}"
    echo "Check the error messages above"
    exit 1
fi

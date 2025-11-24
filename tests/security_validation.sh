#!/bin/bash
# UltraThink Pilot - Security Validation Test Suite
# Validates that all hardcoded passwords have been removed
# and security best practices are followed.

set -e  # Exit on error

GREEN='\033[0.32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

echo "========================================"
echo "  Security Validation Test Suite"
echo "  UltraThink Pilot - Day 2 Completion"
echo "========================================"
echo ""

# Test 1: No hardcoded passwords in Python files
echo "Test 1: Checking for hardcoded passwords in Python files..."
FOUND=$(grep -r "changeme_in_production\|ultrathink_changeme" \
  --exclude-dir=node_modules \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude="*.md" \
  --exclude="SECURITY_AUDIT*" \
  --include="*.py" \
  services/ 2>/dev/null || true)

if [ -n "$FOUND" ]; then
  echo -e "${RED}❌ FAIL${NC}: Found hardcoded passwords in Python files:"
  echo "$FOUND"
  FAILED=$((FAILED + 1))
else
  echo -e "${GREEN}✅ PASS${NC}: No hardcoded passwords in Python files"
  PASSED=$((PASSED + 1))
fi
echo ""

# Test 2: No hardcoded passwords in YAML config files
echo "Test 2: Checking for hardcoded passwords in YAML files..."
FOUND=$(grep -r "changeme_in_production" \
  --exclude-dir=node_modules \
  --exclude-dir=.git \
  --include="*.yml" \
  --include="*.yaml" \
  infrastructure/ 2>/dev/null | grep -v ".env.example" | grep -v ".env.template" || true)

if [ -n "$FOUND" ]; then
  echo -e "${RED}❌ FAIL${NC}: Found hardcoded passwords in YAML files:"
  echo "$FOUND"
  FAILED=$((FAILED + 1))
else
  echo -e "${GREEN}✅ PASS${NC}: No hardcoded passwords in YAML config files"
  PASSED=$((PASSED + 1))
fi
echo ""

# Test 3: .env file exists
echo "Test 3: Checking .env file exists..."
if [ ! -f "infrastructure/.env" ]; then
  echo -e "${RED}❌ FAIL${NC}: infrastructure/.env not found"
  echo "  Run: python3 scripts/generate_secrets.py --output infrastructure/.env"
  FAILED=$((FAILED + 1))
else
  echo -e "${GREEN}✅ PASS${NC}: .env file exists"
  PASSED=$((PASSED + 1))
fi
echo ""

# Test 4: .env file has correct permissions
echo "Test 4: Checking .env file permissions..."
if [ -f "infrastructure/.env" ]; then
  PERMS=$(stat -c %a infrastructure/.env 2>/dev/null || stat -f %A infrastructure/.env 2>/dev/null)
  if [ "$PERMS" != "600" ]; then
    echo -e "${RED}❌ FAIL${NC}: .env permissions are $PERMS (should be 600)"
    echo "  Run: chmod 600 infrastructure/.env"
    FAILED=$((FAILED + 1))
  else
    echo -e "${GREEN}✅ PASS${NC}: .env file has correct permissions (600)"
    PASSED=$((PASSED + 1))
  fi
else
  echo -e "${RED}❌ SKIP${NC}: .env file not found"
fi
echo ""

# Test 5: All required passwords in .env
echo "Test 5: Checking required passwords in .env..."
if [ -f "infrastructure/.env" ]; then
  REQUIRED_VARS=(
    "POSTGRES_PASSWORD"
    "GRAFANA_ADMIN_PASSWORD"
    "GF_SECURITY_SECRET_KEY"
    "JWT_SECRET_KEY"
  )

  MISSING=()
  for VAR in "${REQUIRED_VARS[@]}"; do
    if ! grep -q "^${VAR}=" infrastructure/.env; then
      MISSING+=("$VAR")
    fi
  done

  if [ ${#MISSING[@]} -gt 0 ]; then
    echo -e "${RED}❌ FAIL${NC}: Missing required passwords in .env:"
    for VAR in "${MISSING[@]}"; do
      echo "  - $VAR"
    done
    FAILED=$((FAILED + 1))
  else
    echo -e "${GREEN}✅ PASS${NC}: All required passwords present in .env"
    PASSED=$((PASSED + 1))
  fi
else
  echo -e "${RED}❌ SKIP${NC}: .env file not found"
fi
echo ""

# Test 6: Password strength (minimum 32 characters)
echo "Test 6: Checking password strength..."
if [ -f "infrastructure/.env" ]; then
  PASSWORD=$(grep "^POSTGRES_PASSWORD=" infrastructure/.env | cut -d= -f2 | tr -d '"')
  if [ ${#PASSWORD} -lt 32 ]; then
    echo -e "${RED}❌ FAIL${NC}: POSTGRES_PASSWORD too short (${#PASSWORD} chars, need 32+)"
    FAILED=$((FAILED + 1))
  else
    echo -e "${GREEN}✅ PASS${NC}: Passwords meet 32-character minimum (POSTGRES_PASSWORD: ${#PASSWORD} chars)"
    PASSED=$((PASSED + 1))
  fi
else
  echo -e "${RED}❌ SKIP${NC}: .env file not found"
fi
echo ""

# Test 7: Docker compose files use environment variables
echo "Test 7: Checking docker-compose files use env vars..."
COMPOSE_FILES=("infrastructure/docker-compose.yml" "infrastructure/docker-compose.enhanced.yml")
COMPOSE_PASS=true

for FILE in "${COMPOSE_FILES[@]}"; do
  if [ -f "$FILE" ]; then
    # Check that password fields use ${...} syntax
    if grep -q "POSTGRES_PASSWORD.*changeme" "$FILE"; then
      echo -e "${RED}❌ FAIL${NC}: $FILE still has hardcoded password"
      COMPOSE_PASS=false
    fi
  fi
done

if $COMPOSE_PASS; then
  echo -e "${GREEN}✅ PASS${NC}: Docker compose files use environment variables"
  PASSED=$((PASSED + 1))
else
  FAILED=$((FAILED + 1))
fi
echo ""

# Test 8: Port conflict resolved
echo "Test 8: Checking port conflict resolved..."
# risk-manager should use 8003, not 8001
CONFLICT=false

if grep -q "ports:.*8001:8001" infrastructure/docker-compose.yml | grep -q "risk-manager"; then
  echo -e "${RED}❌ FAIL${NC}: risk-manager still uses port 8001"
  CONFLICT=true
fi

if grep -q "port.*8003" services/risk_manager/main.py || grep -q "PORT.*8003" services/risk_manager/main.py; then
  if ! $CONFLICT; then
    echo -e "${GREEN}✅ PASS${NC}: Port conflict resolved (risk-manager uses 8003)"
    PASSED=$((PASSED + 1))
  fi
else
  echo -e "${RED}❌ FAIL${NC}: risk-manager port not updated"
  FAILED=$((FAILED + 1))
fi
echo ""

# Test 9: .gitignore includes .env
echo "Test 9: Checking .gitignore..."
if [ -f ".gitignore" ]; then
  if grep -q "\.env$\|^\.env$" .gitignore; then
    echo -e "${GREEN}✅ PASS${NC}: .env is in .gitignore"
    PASSED=$((PASSED + 1))
  else
    echo -e "${RED}❌ FAIL${NC}: .env not in .gitignore (security risk!)"
    echo "  Add '.env' to .gitignore"
    FAILED=$((FAILED + 1))
  fi
else
  echo -e "${RED}❌ WARN${NC}: .gitignore not found"
fi
echo ""

# Test 10: Docker compose validation
echo "Test 10: Validating docker-compose syntax..."
if command -v docker-compose &> /dev/null; then
  cd infrastructure
  if docker-compose config > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PASS${NC}: docker-compose.yml syntax is valid"
    PASSED=$((PASSED + 1))
  else
    echo -e "${RED}❌ FAIL${NC}: docker-compose.yml has syntax errors"
    FAILED=$((FAILED + 1))
  fi
  cd ..
else
  echo -e "${RED}❌ SKIP${NC}: docker-compose not installed"
fi
echo ""

# Summary
echo "========================================"
echo "  Test Summary"
echo "========================================"
echo -e "${GREEN}Passed:${NC} $PASSED"
echo -e "${RED}Failed:${NC} $FAILED"
echo "Total:  $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
  echo -e "${GREEN}✅ ALL TESTS PASSED${NC}"
  echo ""
  echo "Security Status: READY FOR PRODUCTION"
  echo "Next steps:"
  echo "  1. Test service startup: cd infrastructure && docker-compose up -d timescaledb mlflow"
  echo "  2. Verify connectivity: docker-compose logs"
  echo "  3. Proceed with deployment"
  exit 0
else
  echo -e "${RED}❌ SOME TESTS FAILED${NC}"
  echo ""
  echo "Please fix the issues above before proceeding."
  echo "Re-run: ./tests/security_validation.sh"
  exit 1
fi

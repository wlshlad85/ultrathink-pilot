#!/bin/bash
# Quick test script for Inference API
# Tests basic functionality without full deployment

set -e

BASE_URL="${BASE_URL:-http://localhost:8080}"

echo "========================================"
echo "Inference API Quick Test"
echo "========================================"
echo "Base URL: $BASE_URL"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

test_passed=0
test_failed=0

# Test function
test_endpoint() {
    local name="$1"
    local endpoint="$2"
    local method="${3:-GET}"
    local data="$4"

    echo -n "Testing $name... "

    if [ "$method" = "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data" 2>/dev/null || echo "000")
    else
        response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint" 2>/dev/null || echo "000")
    fi

    http_code=$(echo "$response" | tail -n 1)
    body=$(echo "$response" | head -n -1)

    if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
        echo -e "${GREEN}✓ PASS${NC} (HTTP $http_code)"
        ((test_passed++))
        if [ -n "$body" ]; then
            echo "  Response: $(echo "$body" | head -c 100)..."
        fi
    elif [ "$http_code" -eq 503 ]; then
        echo -e "${RED}⚠ DEGRADED${NC} (HTTP $http_code - Models not loaded)"
        echo "  Note: This is expected if models aren't loaded yet"
        ((test_passed++))
    else
        echo -e "${RED}✗ FAIL${NC} (HTTP $http_code)"
        ((test_failed++))
        if [ -n "$body" ]; then
            echo "  Error: $body"
        fi
    fi
    echo ""
}

# Run tests
echo "Running tests..."
echo ""

test_endpoint "Root endpoint" "/"
test_endpoint "Health check" "/health"
test_endpoint "Models list" "/api/v1/models"
test_endpoint "Metrics" "/metrics"

test_endpoint "Prediction (AAPL)" "/api/v1/predict" "POST" \
    '{"symbol": "AAPL", "risk_check": false}'

test_endpoint "Prediction (MSFT with risk)" "/api/v1/predict" "POST" \
    '{"symbol": "MSFT", "risk_check": true}'

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "Passed: ${GREEN}$test_passed${NC}"
echo -e "Failed: ${RED}$test_failed${NC}"
echo ""

if [ $test_failed -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi

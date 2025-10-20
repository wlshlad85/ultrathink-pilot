#!/bin/bash
# Master test runner for UltraThink system
# Runs all tests and generates validation report

echo "======================================================================="
echo "ULTRATHINK SYSTEM - COMPREHENSIVE TEST SUITE"
echo "======================================================================="
echo ""

cd /home/rich/ultrathink-pilot
source .venv/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_RESULTS_DIR="test_results_${TIMESTAMP}"
mkdir -p "$TEST_RESULTS_DIR"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test and capture result
run_test() {
    local test_name="$1"
    local test_command="$2"

    echo "-----------------------------------------------------------------------"
    echo "Running: $test_name"
    echo "-----------------------------------------------------------------------"

    if eval "$test_command" > "${TEST_RESULTS_DIR}/${test_name}.log" 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}: $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}: $test_name"
        echo "  See: ${TEST_RESULTS_DIR}/${test_name}.log"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

echo "======================================================================="
echo "PHASE 1: DEPENDENCY CHECKS"
echo "======================================================================="
echo ""

run_test "check_pytorch" "python3 -c 'import torch; print(f\"PyTorch {torch.__version__}\")'"
run_test "check_cuda" "python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
run_test "check_gymnasium" "python3 -c 'import gymnasium; print(f\"Gymnasium {gymnasium.__version__}\")'"
run_test "check_yfinance" "python3 -c 'import yfinance; print(\"yfinance OK\")'"
run_test "check_pandas" "python3 -c 'import pandas; print(f\"Pandas {pandas.__version__}\")'"
run_test "check_matplotlib" "python3 -c 'import matplotlib; print(f\"Matplotlib {matplotlib.__version__}\")'"

echo ""
echo "======================================================================="
echo "PHASE 2: MODULE IMPORTS"
echo "======================================================================="
echo ""

run_test "import_backtesting" "python3 -c 'from backtesting import DataFetcher, Portfolio, PerformanceMetrics, BacktestEngine; print(\"Backtesting imports OK\")'"
run_test "import_rl" "python3 -c 'from rl import TradingEnv, PPOAgent; print(\"RL imports OK\")'"
run_test "import_agents" "python3 -c 'from agents import model_backends, schema; print(\"Agents imports OK\")'"

echo ""
echo "======================================================================="
echo "PHASE 3: UNIT TESTS"
echo "======================================================================="
echo ""

run_test "test_backtesting_unit" "pytest tests/test_backtesting.py -v"
run_test "test_rl_unit" "pytest tests/test_rl.py -v"

echo ""
echo "======================================================================="
echo "PHASE 4: INTEGRATION TESTS"
echo "======================================================================="
echo ""

run_test "test_data_fetcher_integration" "python3 backtesting/data_fetcher.py"
run_test "test_trading_env_integration" "python3 rl/trading_env.py"
run_test "test_ppo_agent_integration" "python3 rl/ppo_agent.py"

echo ""
echo "======================================================================="
echo "PHASE 5: END-TO-END TESTS"
echo "======================================================================="
echo ""

run_test "test_backtest_short" "python3 run_backtest.py --start 2024-05-01 --end 2024-06-01 --skip-days 20 --output ${TEST_RESULTS_DIR}/backtest_test.json"

# Only run short RL training test if GPU available
if python3 -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "CUDA detected - running RL training test..."
    run_test "test_rl_training_short" "timeout 300 python3 rl/train.py --episodes 2 --update-freq 50 --model-dir ${TEST_RESULTS_DIR}/rl_test"
else
    echo -e "${YELLOW}⊘ SKIPPED${NC}: RL training test (no CUDA available)"
fi

echo ""
echo "======================================================================="
echo "PHASE 6: STRESS TESTS"
echo "======================================================================="
echo ""

run_test "test_large_dataset" "python3 -c '
from backtesting import DataFetcher
fetcher = DataFetcher(\"BTC-USD\")
df = fetcher.fetch(\"2023-01-01\", \"2024-01-01\")
fetcher.add_technical_indicators()
print(f\"Processed {len(df)} data points\")
'"

run_test "test_multiple_envs" "python3 -c '
from rl import TradingEnv
envs = [TradingEnv(\"BTC-USD\", \"2024-01-01\", \"2024-02-01\") for _ in range(5)]
for env in envs:
    env.reset()
print(\"Created and reset 5 environments\")
'"

echo ""
echo "======================================================================="
echo "TEST SUMMARY"
echo "======================================================================="
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))

echo "Total Tests:   $TOTAL_TESTS"
echo -e "${GREEN}Tests Passed:  $TESTS_PASSED${NC}"
echo -e "${RED}Tests Failed:  $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}ALL TESTS PASSED! ✓${NC}"
    echo -e "${GREEN}======================================${NC}"
    EXIT_CODE=0
else
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}SOME TESTS FAILED ✗${NC}"
    echo -e "${RED}======================================${NC}"
    echo ""
    echo "Check logs in: $TEST_RESULTS_DIR/"
    EXIT_CODE=1
fi

echo ""
echo "Detailed logs saved to: $TEST_RESULTS_DIR/"
echo ""

# Generate summary report
cat > "${TEST_RESULTS_DIR}/summary.md" << EOF
# UltraThink Test Results

**Date**: $(date)
**Total Tests**: $TOTAL_TESTS
**Passed**: $TESTS_PASSED
**Failed**: $TESTS_FAILED

## Test Phases

1. **Dependency Checks**: Verify all required packages
2. **Module Imports**: Test all modules can be imported
3. **Unit Tests**: Test individual components
4. **Integration Tests**: Test module interactions
5. **End-to-End Tests**: Test complete workflows
6. **Stress Tests**: Test with large datasets

## Results

$(if [ $TESTS_FAILED -eq 0 ]; then echo "✅ All tests passed!"; else echo "❌ Some tests failed. See logs for details."; fi)

## Logs

All test logs are available in this directory.

EOF

echo "Summary report: ${TEST_RESULTS_DIR}/summary.md"
echo ""

exit $EXIT_CODE

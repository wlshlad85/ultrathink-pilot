# UltraThink Backtesting Framework

A comprehensive backtesting engine for the UltraThink trading agent system (MR-SR + ERS).

## Features

- **Historical Data Fetching**: Automatically downloads market data via yfinance
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Moving Averages, etc.
- **Portfolio Simulation**: Realistic trade execution with commission, position tracking, P&L
- **Agent Integration**: Runs your MR-SR and ERS agents on historical data
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, profit factor, etc.
- **Risk Metrics**: VaR, CVaR, volatility, Sortino ratio, Calmar ratio

## Quick Start

### 1. Run a basic backtest

```bash
# From project root
python run_backtest.py
```

This runs a 1-year BTC-USD backtest with $100k initial capital.

### 2. Customize parameters

```bash
python run_backtest.py \
  --symbol BTC-USD \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --capital 100000 \
  --commission 0.001 \
  --output my_backtest.json
```

### 3. Use with OpenAI API

```bash
# Set your API key first
export OPENAI_API_KEY="your-key-here"

python run_backtest.py --use-openai
```

## Architecture

```
┌─────────────────┐
│  DataFetcher    │  Fetch OHLCV + calculate indicators
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BacktestEngine  │  Main orchestrator
└────────┬────────┘
         │
         ├──► MR-SR Agent  (Strategy recommendation)
         │
         ├──► ERS Agent    (Risk validation)
         │
         ▼
┌─────────────────┐
│   Portfolio     │  Execute trades, track P&L
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│PerformanceMetrics│  Calculate Sharpe, drawdown, etc.
└─────────────────┘
```

## Module Details

### DataFetcher (`data_fetcher.py`)
- Downloads historical data from yfinance
- Calculates 15+ technical indicators
- Provides market context for agents

### Portfolio (`portfolio.py`)
- Simulates realistic trading with commissions
- Tracks positions, cash, P&L
- Records equity curve
- Position sizing and risk management

### PerformanceMetrics (`metrics.py`)
- Sharpe & Sortino ratios
- Maximum drawdown
- VaR & CVaR
- Profit factor
- Calmar ratio

### BacktestEngine (`backtest_engine.py`)
- Orchestrates entire backtest
- Calls MR-SR and ERS agents
- Executes trades based on recommendations
- Generates comprehensive reports

## Output

The backtest generates a JSON report with:
- Portfolio performance (return, P&L, win rate)
- Risk-adjusted metrics (Sharpe, max drawdown, volatility)
- Complete trade history
- Agent decision log
- Equity curve data

## Example Output

```
======================================================================
BACKTEST REPORT
======================================================================

--- Configuration ---
Symbol:              BTC-USD
Period:              2023-01-01 to 2024-01-01
Initial Capital:     $100,000.00
Commission Rate:     0.10%

--- Portfolio Performance ---
Final Value:         $127,450.00
Total P&L:           $27,450.00
Total Return:        27.45%

--- Trading Activity ---
Total Trades:        45
Winning Trades:      28
Losing Trades:       17
Win Rate:            62.22%

--- Risk-Adjusted Metrics ---
Sharpe Ratio:        1.85
Sortino Ratio:       2.34
Max Drawdown:        -12.34%
Volatility:          18.45%
======================================================================
```

## Next Steps: RL Integration

This backtesting framework provides the foundation for reinforcement learning:

1. **Environment**: Use `Portfolio` + `DataFetcher` as RL gym environment
2. **State Space**: Market indicators + portfolio state
3. **Action Space**: BUY/SELL/HOLD decisions
4. **Reward Function**: Returns, Sharpe ratio, or risk-adjusted metrics
5. **Training**: Use your CUDA GPU with PyTorch/stable-baselines3

See the upcoming RL integration guide for details.

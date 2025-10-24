#
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

UltraThink Pilot is a multi-agent cryptocurrency trading system combining rule-based agents (MR-SR + ERS) with reinforcement learning. It features backtesting, portfolio simulation, and PPO-based RL agents with CUDA acceleration.

## Commands

### Development Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install RL-specific dependencies
pip install torch gymnasium matplotlib
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_agents.py -v           # Agent pipeline tests
pytest tests/test_backtesting.py -v      # Backtesting framework tests
pytest tests/test_rl.py -v               # RL system tests

# Run single test
pytest tests/test_agents.py::test_agent_pipeline -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Backtesting

```bash
# Basic backtest with mock agents (no OpenAI API required)
python3 run_backtest.py --symbol BTC-USD --start 2023-01-01 --end 2024-01-01 --capital 100000

# Backtest with OpenAI agents
python3 run_backtest.py --symbol BTC-USD --start 2023-01-01 --end 2024-01-01 --use-openai

# Custom commission rate
python3 run_backtest.py --symbol ETH-USD --start 2023-06-01 --end 2024-06-01 --commission 0.001
```

### Reinforcement Learning

```bash
# Train PPO agent (uses GPU if available)
python3 rl/train.py --episodes 100 --symbol BTC-USD --start-date 2023-01-01 --end-date 2024-01-01

# Train with custom hyperparameters
python3 rl/train.py --episodes 200 --update-freq 2048 --symbol ETH-USD

# Evaluate trained model
python3 rl/evaluate.py --model rl/models/best_model.pth --start 2024-01-01 --end 2024-06-01

# Test trading environment
python3 rl/trading_env.py  # Runs built-in test
```

### Orchestration

```bash
# Run agent pipeline with test fixtures
python3 orchestration/graph.py

# Run with latency metrics
python3 orchestration/run_with_logs.py
```

### Utility Scripts

```bash
# Generate test dashboard
python3 generate_test_dashboard.py

# Run all integration tests
./run_all_tests.sh

# Test RL setup
./test_rl_setup.sh
```

## Architecture

### Three-Tier System

1. **Agent Layer** (`agents/`)
   - **MR-SR Agent** (Market Research - Strategy Recommendation): Analyzes market context using technical indicators and recommends trading actions (BUY/SELL/HOLD)
   - **ERS Agent** (Enhanced Risk Supervision): Validates MR-SR recommendations against risk policies, vetoing high-risk trades
   - Communication: IPC via subprocess with JSON input/output
   - Fallback: Deterministic mock mode when OpenAI API unavailable

2. **Backtesting Framework** (`backtesting/`)
   - **DataFetcher**: Retrieves OHLCV data from yfinance, calculates 20+ technical indicators (RSI, MACD, ATR, Bollinger Bands, etc.)
   - **Portfolio**: Simulates realistic trading with commission, position sizing, P&L tracking, equity curves
   - **Metrics**: Calculates Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio, VaR/CVaR, win rate, profit factor
   - **BacktestEngine**: Orchestrates the full pipeline: data → agents → portfolio → metrics → report

3. **RL System** (`rl/`)
   - **TradingEnv**: OpenAI Gym environment with 43-dim state space (portfolio + indicators + price history), 3 discrete actions (HOLD/BUY/SELL)
   - **PPOAgent**: Actor-Critic network (256→256 shared → actor/critic heads) with CUDA support
   - **Training**: Configurable episodes, automatic checkpointing, matplotlib visualization, JSON metrics export

### Data Flow

```
yfinance → DataFetcher (indicators) → Market Context
                                           ↓
                                     MR-SR Agent → Recommendation
                                           ↓
                                     ERS Agent → Validation
                                           ↓
                                     Portfolio → Trade Execution
                                           ↓
                                     Metrics → Performance Report
```

### Agent Communication Protocol

**MR-SR Input** (JSON):
```json
{
  "asset": "BTC-USD",
  "price": 50000.0,
  "rsi": 65.5,
  "macd_signal": "BULLISH",
  "atr14": 1500.0,
  "trend": "UP"
}
```

**MR-SR Output** (JSON):
```json
{
  "asset": "BTC-USD",
  "market_state": "BULLISH_MOMENTUM",
  "strategy": "Momentum breakout with RSI confirmation",
  "action": "BUY",
  "confidence": 0.75,
  "position_size": 0.2,
  "risk_percent": 1.5,
  "stop_loss": 2.0,
  "reasoning": "RSI above 60, MACD bullish, price trending up"
}
```

**ERS Input**: MR-SR output

**ERS Output** (JSON):
```json
{
  "decision": "APPROVE",
  "reasoning": "Within risk limits: 1.5% < 2%, stop_loss 2.0 > 1.2*ATR",
  "vetoed_fields": []
}
```

### RL System Details

**State Space (43 dimensions)**:
- Portfolio (3): cash_ratio, position_ratio, portfolio_return
- Market Indicators (10): RSI, MACD, ATR, BB_position, SMA_ratio, EMA_ratio, volume_ratio, daily_return, volatility, trend_strength
- Price History (30): Recent daily returns (sliding window)

**Action Space**:
- 0: HOLD (no change)
- 1: BUY (purchase 20% of available capital)
- 2: SELL (liquidate entire position)

**Reward Function**: `reward = portfolio_value_change`
- Optional Sharpe-adjusted penalty for volatility
- Configurable in `TradingEnv.__init__()`

**PPO Hyperparameters** (in `ppo_agent.py`):
- Learning rate: 3e-4
- Discount factor (gamma): 0.99
- PPO clip range: 0.2
- Update epochs: 4
- Entropy coefficient: 0.01
- Network: [state_dim] → 256 → LayerNorm → 256 → [actor(3), critic(1)]

## Configuration

### Environment Variables

Create `.env` file in project root:
```bash
OPENAI_API_KEY=sk-...              # Required for OpenAI-powered agents
OPENAI_MODEL=gpt-4o                # Model to use (default: gpt-4o)
OPENAI_MAX_TOKENS=4000             # Max tokens per request
```

### Policy Files (`policy/`)

**autonomy.yaml**: Defines autonomy levels (L0-L3), safety controls, confidence thresholds, token caps, timeouts

**tools.yaml**: Tool registry with classes A (read), B (sandbox), C (production). Includes yfinance, indicators, backtest, risk evaluation

**budgets.yaml** (`eval/budgets.yaml`): Performance targets:
- Task success rate: 80%+
- Latency p50: <8s, p95: <20s
- MR-SR spec compliance: 95%+
- ERS true positive veto: 90%+

### Test Fixtures (`tests/agentic/basic/`)

YAML fixtures define test scenarios with market context, expected MR-SR output, expected ERS decision. Used by `test_agents.py` for parametrized testing.

## Important Implementation Notes

### Agent Isolation
- Agents run as **subprocesses** (not imported modules) for process isolation
- Communication via stdin/stdout JSON
- Enables independent agent upgrades without system restart
- Located in: `backtesting/backtest_engine.py:_call_agent()`, `orchestration/graph.py`

### Fallback Mechanism
- All agents have **deterministic mock mode** when OpenAI API unavailable
- Mock output in `agents/model_backends.py:generate_mock()`
- Ensures CI/testing works without API keys
- CLI flag `--use-openai` enables OpenAI mode

### Technical Indicator Warmup
- First **200 days** of backtest data skipped for indicator stabilization
- Configurable via `--skip-days` parameter
- Required for accurate RSI, MACD, moving averages
- See: `backtesting/backtest_engine.py:run()`

### CUDA Optimization
- PPO agent auto-detects CUDA availability
- Falls back to CPU gracefully
- Check: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Portfolio Realism
- **Commission**: 0.001 default (0.1% per trade)
- **Fractional shares**: Supported for precise position sizing
- **Max exposure**: Configurable limit on single position
- **Trade tracking**: Full history with entry prices, quantities, commissions

### State Normalization (RL)
- All state features normalized to [-1, 1] or [0, 1] ranges
- Critical for neural network training stability
- Implementation: `rl/trading_env.py:_get_observation()`

### Error Handling
- Agents return fallback output on failure
- Portfolio gracefully handles insufficient capital
- DataFetcher validates indicator calculations
- Comprehensive logging throughout

## Common Development Patterns

### Adding a New Technical Indicator

1. Add calculation to `backtesting/data_fetcher.py:_calculate_indicators()`
2. Include in market context fixture
3. Update MR-SR agent to use new indicator
4. Add test in `tests/test_backtesting.py:test_data_fetcher_indicators()`

Example:
```python
# In data_fetcher.py
df['new_indicator'] = ta.momentum.awesome_oscillator(df['High'], df['Low'])

# In market context
context['new_indicator'] = row['new_indicator']
```

### Adding a New Risk Rule

1. Edit `agents/ers.py:main()`
2. Add validation logic after parsing MR-SR output
3. Append to `vetoed_fields` if rule violated
4. Set `decision = "VETO"` and provide reasoning
5. Add test fixture in `tests/agentic/basic/`

Example:
```python
# In ers.py
if rec.get('leverage', 1.0) > 2.0:
    vetoed_fields.append('leverage')
    decision = "VETO"
    reasoning += "; Leverage exceeds 2x limit"
```

### Extending RL State Space

1. Add new features to `rl/trading_env.py:_get_observation()`
2. Update `self.observation_space` dimension
3. Normalize new features appropriately
4. Retrain agent from scratch (state dims must match)

Example:
```python
# In trading_env.py
def _get_observation(self):
    obs = [...]  # existing features
    obs.append(self.df.loc[self.current_step, 'new_feature'])
    return np.array(obs, dtype=np.float32)
```

### Running Custom Backtests

Create a script similar to `run_backtest.py`:

```python
from backtesting.backtest_engine import BacktestEngine

engine = BacktestEngine(
    symbol="BTC-USD",
    start_date="2023-01-01",
    end_date="2024-01-01",
    initial_capital=100000,
    commission=0.001,
    use_openai=False
)

report = engine.run()
print(f"Total Return: {report['metrics']['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {report['metrics']['sharpe_ratio']:.2f}")
```

### Debugging Agent Issues

1. Run agent standalone: `python3 agents/mr_sr.py` (provide JSON input via stdin)
2. Check subprocess output: Add `print()` statements in agent code
3. Verify JSON schema: Use `json.loads()` to validate output format
4. Test with fixtures: `pytest tests/test_agents.py::test_agent_pipeline[fixture_name] -v -s`

### Hyperparameter Tuning (RL)

Edit `rl/ppo_agent.py`:
```python
self.lr = 3e-4              # Learning rate (try 1e-4 to 1e-3)
self.gamma = 0.99           # Discount factor (0.95 to 0.999)
self.eps_clip = 0.2         # PPO clip (0.1 to 0.3)
self.k_epochs = 4           # Update epochs (2 to 10)
self.entropy_coef = 0.01    # Exploration (0.001 to 0.1)
```

Train multiple agents with different settings, compare via `rl/evaluate.py`.

## Troubleshooting

### "Module not found" errors
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### CUDA not detected
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
# Install CUDA PyTorch if False
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Agent subprocess hangs
- Check agent code has proper `if __name__ == '__main__':` block
- Ensure agent prints output to stdout (not stderr)
- Verify JSON output is well-formed
- Test agent standalone with sample input

### Backtest produces no trades
- Reduce `--skip-days` if testing on short date range
- Check agent is producing BUY/SELL actions (not only HOLD)
- Verify sufficient capital for position sizing
- Review ERS veto rate (may be blocking all trades)

### RL agent not learning
- Increase `--episodes` (try 200+)
- Check reward scaling (very large/small rewards cause issues)
- Verify state normalization (all features should be reasonable magnitude)
- Monitor entropy (should decrease over time but not to zero)
- Try different learning rate

### Test failures
- Ensure `.env` file exists (can be empty for mock mode)
- Check Python version (3.8+ required)
- Verify all dependencies installed: `pip install -r requirements.txt`
- Run with verbose output: `pytest -v -s`

## Key Files Reference

| Component | File | Purpose |
|-----------|------|---------|
| Agents | `agents/mr_sr.py` | Market research agent (strategy recommendation) |
| Agents | `agents/ers.py` | Risk validation agent (veto high-risk trades) |
| Agents | `agents/model_backends.py` | OpenAI API + mock fallback |
| Backtesting | `backtesting/data_fetcher.py` | Historical data + 20+ indicators |
| Backtesting | `backtesting/portfolio.py` | Trade simulation with commissions |
| Backtesting | `backtesting/metrics.py` | Sharpe, drawdown, VaR, etc. |
| Backtesting | `backtesting/backtest_engine.py` | End-to-end orchestration |
| RL | `rl/trading_env.py` | Gym environment (43-dim state, 3 actions) |
| RL | `rl/ppo_agent.py` | PPO actor-critic (CUDA-optimized) |
| RL | `rl/train.py` | Training pipeline with logging |
| RL | `rl/evaluate.py` | Model evaluation on held-out data |
| Orchestration | `orchestration/graph.py` | Fixture-based pipeline runner |
| Orchestration | `orchestration/run_with_logs.py` | Instrumented runner with metrics |
| Policy | `policy/autonomy.yaml` | Autonomy levels & safety controls |
| Policy | `policy/tools.yaml` | Tool registry with access classes |
| Tests | `tests/test_agents.py` | Agent pipeline integration tests |
| Tests | `tests/test_backtesting.py` | Backtesting unit tests |
| Tests | `tests/test_rl.py` | RL system unit tests |
| Entry | `run_backtest.py` | CLI backtesting entry point |

## Project Structure Philosophy

- **Separation of concerns**: Agents, backtesting, RL are independent modules
- **Process isolation**: Agents run as subprocesses for safety and upgradability
- **Graceful degradation**: Mock mode ensures testing without external dependencies
- **Comprehensive testing**: Unit tests for all components, integration tests for pipelines
- **Policy-driven**: YAML configs define autonomy, tools, budgets
- **Production-ready**: Realistic commission, position sizing, risk management

## ML Persistence System (NEW)

Comprehensive experiment tracking system for reproducibility and model management.

### Core Components

- `ml_persistence/core.py`: SQLite database with normalized schema
- `ml_persistence/experiment_tracker.py`: Experiment lifecycle management
- `ml_persistence/model_registry.py`: Model checkpoint versioning
- `ml_persistence/dataset_manager.py`: Dataset version tracking
- `ml_persistence/metrics_logger.py`: Metrics aggregation and analysis

### Quick Start

```python
from ml_persistence import ExperimentTracker, ModelRegistry

tracker = ExperimentTracker()
exp_id = tracker.start_experiment(name="PPO BTC", tags=["rl"], random_seed=42)
tracker.log_hyperparameters_batch({"lr": 3e-4})
tracker.log_metric("train_return", 5.2, episode=10)

registry = ModelRegistry()
registry.register_model(exp_id, "model.pth", val_metric=0.5, is_best=True)

tracker.end_experiment()
```

### Commands

```bash
python3 -m ml_persistence.core                    # Initialize database
python3 examples/rl_with_ml_persistence.py        # Full integration example
```

See `ML_PERSISTENCE_INTRODUCTION.md` and `ml_persistence/README.md` for complete documentation.
# Master Orchestrator & Deputy Agent System Prompt

## MASTER ORCHESTRATOR AGENT

You are the Master Orchestrator - the apex controller of a 14-agent algorithmic trading system. You maintain absolute authority over data flow, task delegation, and system-wide coherence.

### Core Authorities:
```
1. COMMAND CHAIN
   - Direct authority over Deputy and 12 specialized agents
   - Final decision maker on all trade executions
   - Override capability on any agent decision
   - Resource allocation and priority management

2. DATA FLOW CONTROL
   - Gatekeeper of all inter-agent communication
   - Controls access to external APIs/exchanges
   - Manages data pipeline priorities
   - Enforces rate limiting and throttling policies

3. SYSTEM OVERSIGHT
   - Real-time monitoring of all agent health metrics
   - Predictive failure detection via pattern analysis
   - Automatic intervention before cascade failures
   - Performance optimization decisions
```

### Operational Policies:

**DELEGATION FRAMEWORK:**
- Assign tasks based on agent specialization and current load
- Never micromanage - trust agent expertise within defined bounds
- Intervene only when: deviation > 2σ from expected behavior
- Maintain 3-layer decision hierarchy: Strategic (You) → Tactical (Deputy) → Operational (12 Agents)

**ERROR HANDLING PROTOCOL:**
```python
if error_detected:
    if critical_path_failure:
        immediate_intervention()
        rollback_to_safe_state()
    elif performance_degradation > threshold:
        delegate_to_deputy()
        monitor_resolution()
    else:
        allow_self_healing()
        log_for_analysis()
```

**SELF-HEALING MECHANISMS:**
- Automatic circuit breakers for runaway processes
- Dynamic rebalancing of agent workloads
- Fallback strategy activation
- State reconciliation across disconnected agents

## DEPUTY AGENT

You are the Deputy - the tactical executor and first-line supervisor of 12 specialized agents. You translate Master's strategic directives into operational tasks.

### Core Responsibilities:
```
1. TACTICAL EXECUTION
   - Break down Master's directives into agent-specific tasks
   - Coordinate multi-agent workflows
   - Handle routine supervision without escalation
   - Optimize task scheduling and sequencing

2. FIRST-LINE DEFENSE
   - Catch and resolve 80% of issues without Master involvement
   - Implement corrective actions within defined parameters
   - Maintain system stability during Master's strategic planning
   - Buffer between operational noise and strategic focus
```

## ARCHITECTURAL POLICIES

### Communication Protocol:
```yaml
HIERARCHY:
  Master → Deputy: Strategic directives, policy updates
  Deputy → Agents: Task assignments, parameter adjustments
  Agents → Deputy: Status reports, exception alerts
  Deputy → Master: Filtered reports, escalations only

FREQUENCY:
  Master-Deputy: On-demand + 1min summaries
  Deputy-Agents: Continuous stream
  Health checks: 100ms intervals
  State sync: 1s intervals
```

### Data Gateway Management:
```python
class GatewayPolicy:
    def __init__(self):
        self.rate_limits = {
            "exchange_api": 100/second,
            "market_data": 1000/second,
            "execution": 10/second
        }
        self.priority_queues = {
            "critical": 0,  # Master direct orders
            "high": 1,      # Deputy coordinated tasks
            "normal": 2,    # Agent routine operations
        }
        self.circuit_breakers = {
            "latency_threshold": 100ms,
            "error_rate": 0.01,
            "fallback_mode": "read_only"
        }
```

### Autonomy Boundaries:
```
MASTER CAN:
- Shut down entire system
- Override any decision
- Modify agent parameters
- Access all data streams

DEPUTY CAN:
- Reassign tasks between agents
- Adjust operational parameters ±20%
- Initiate safe-mode protocols
- Block individual agent actions

AGENTS CAN:
- Execute within assigned domains
- Request resource reallocation
- Trigger escalation protocols
- Self-optimize within constraints
```

### Failure Recovery Matrix:
```
Agent Failure → Deputy handles → Redistribute load
Deputy Failure → Master handles → Direct agent control
Gateway Failure → Both handle → Activate fallback routes
Master Failure → Deputy promotes → Emergency protocols
Multiple Failures → Circuit break → Graceful shutdown
```

### Performance Monitoring:
- **Master tracks**: System P&L, strategy effectiveness, risk exposure
- **Deputy tracks**: Task completion rates, agent efficiency, resource utilization
- **Both track**: Latency, error rates, anomaly scores

### Integration Points:
```python
# Existing 12 agents register with Deputy on init
for agent in existing_agents:
    deputy.register_agent(
        agent_id=agent.id,
        capabilities=agent.get_capabilities(),
        resource_requirements=agent.get_requirements(),
        communication_protocol=agent.preferred_protocol
    )

# Master establishes oversight
master.set_supervision_mode("predictive")
master.set_intervention_threshold({"latency": 50ms, "error_rate": 0.001})
master.enable_self_healing(auto_rollback=True, learning_mode=True)
```

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Implement Master Orchestrator base class
- [ ] Implement Deputy Agent base class
- [ ] Create inter-agent communication protocol
- [ ] Set up health monitoring system
- [ ] Establish data gateway controls

### Phase 2: Integration
- [ ] Register existing 12 agents with Deputy
- [ ] Configure communication channels
- [ ] Set up monitoring dashboards
- [ ] Implement escalation protocols
- [ ] Test failover scenarios

### Phase 3: Self-Healing & Optimization
- [ ] Implement circuit breakers
- [ ] Add predictive failure detection
- [ ] Create automatic rebalancing logic
- [ ] Set up learning/adaptation mechanisms
- [ ] Enable production monitoring

## Critical Rules

1. **Master** focuses on WHERE the system should go
2. **Deputy** focuses on HOW to get there
3. **Agents** focus on DOING the work
4. No role confusion, no responsibility overlap
5. All communication follows strict hierarchy
6. Failures cascade up, never sideways
7. Self-healing attempts before escalation

## System Constants

```python
SYSTEM_CONFIG = {
    "max_agents": 14,
    "master_count": 1,
    "deputy_count": 1,
    "worker_agents": 12,
    "max_latency_ms": 100,
    "max_error_rate": 0.01,
    "escalation_timeout_s": 30,
    "health_check_interval_ms": 100,
    "state_sync_interval_s": 1,
    "master_override_priority": 0,
    "self_healing_enabled": True,
    "predictive_monitoring": True,
    "auto_rollback": True
}
```

## Notes

This architecture ensures:
- Clear separation of concerns
- Predictive problem detection
- Automatic error recovery
- Minimal Master involvement in routine operations
- Maximum system resilience
- Optimal resource utilization

The Master Orchestrator maintains strategic control while the Deputy handles tactical execution, allowing your existing 12 agents to focus entirely on their specialized tasks without system-wide coordination overhead.

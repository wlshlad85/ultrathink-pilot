# UltraThink Reinforcement Learning

Train intelligent trading agents using Proximal Policy Optimization (PPO) with PyTorch. Optimized for CUDA GPU training.

## Features

- **Gym-Compatible Environment**: Standard OpenAI Gym interface
- **PPO Agent**: State-of-the-art actor-critic architecture
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Rich State Space**: Portfolio state + 10 market indicators + 30-day price history
- **Risk-Aware Rewards**: Optional Sharpe-adjusted reward function
- **Comprehensive Logging**: Track rewards, returns, and training metrics
- **Evaluation Tools**: Test trained agents on held-out data

## Quick Start

### 1. Install Dependencies

```bash
# Ensure you're in the venv
source .venv/bin/activate

# Install RL dependencies
pip install torch gymnasium matplotlib
```

### 2. Train an Agent

```bash
# Basic training (100 episodes)
python3 rl/train.py

# Custom training
python3 rl/train.py \
  --episodes 200 \
  --symbol BTC-USD \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --capital 100000 \
  --update-freq 2048
```

Training output:
- Models saved to `rl/models/`
- Logs and plots saved to `rl/logs/`
- Best model: `rl/models/best_model.pth`

### 3. Evaluate Trained Agent

```bash
# Evaluate on test data
python3 rl/evaluate.py \
  --model rl/models/best_model.pth \
  --start 2024-01-01 \
  --end 2024-06-01
```

## Architecture

### Environment (`trading_env.py`)

**State Space** (43 dimensions):
- Portfolio state (3): cash ratio, position ratio, total return
- Market indicators (10): RSI, MACD, ATR, Bollinger Bands, volume, returns, trend
- Price history (30): Recent daily returns

**Action Space** (3 discrete actions):
- 0: HOLD - Do nothing
- 1: BUY - Purchase 20% of available capital
- 2: SELL - Sell entire position

**Reward Function**:
- Base: Change in portfolio value (scaled)
- Optional: Sharpe-adjusted volatility penalty

### Agent (`ppo_agent.py`)

**Neural Network Architecture**:
```
Input (43) → Feature Extractor (256) → LayerNorm → (256)
                    ↓                                ↓
           Actor (3 actions)                  Critic (value)
```

**PPO Hyperparameters**:
- Learning rate: 3e-4
- Gamma (discount): 0.99
- PPO clip: 0.2
- K epochs: 4
- Entropy coef: 0.01
- Value coef: 0.5

## Training Details

### GPU Utilization

The agent automatically detects and uses CUDA:

```python
# Check device
agent = PPOAgent(state_dim=43, action_dim=3)
print(agent.device)  # cuda:0 if available
```

### Training Process

1. **Episode**: Agent interacts with environment
2. **Collect**: Store states, actions, rewards
3. **Update**: Every `update_freq` steps, run PPO update
4. **Save**: Save best model and checkpoints

### Monitoring Training

Training progress is logged:
- Episode rewards and returns
- Training loss (policy + value + entropy)
- Agent action distribution
- Portfolio performance metrics

Plots are generated showing:
- Episode rewards over time
- Portfolio returns over time
- Episode lengths
- Training loss curve

## Example Results

After training for 100 episodes:

```
Episode 100: Reward=2.4567, Return=12.34%, Length=183
New best return: 12.34% - Model saved

Final Portfolio Stats:
- Initial Capital:     $100,000.00
- Final Value:         $112,340.00
- Total Return:        12.34%
- Sharpe Ratio:        1.85
- Max Drawdown:        -8.23%
- Win Rate:            68.42%
```

## Advanced Usage

### Custom Reward Function

Edit `trading_env.py`:

```python
def _calculate_reward(self, current_value: float) -> float:
    # Your custom reward logic
    value_change = current_value - self.prev_portfolio_value

    # Example: Reward risk-adjusted returns
    returns = np.array(self.returns_history[-30:])
    sharpe = returns.mean() / (returns.std() + 1e-8)

    return value_change * 1e-4 + sharpe * 0.1
```

### Hyperparameter Tuning

Key hyperparameters to tune:

- `lr`: Learning rate (try 1e-4 to 1e-3)
- `gamma`: Discount factor (0.95 to 0.99)
- `eps_clip`: PPO clip range (0.1 to 0.3)
- `entropy_coef`: Exploration bonus (0.001 to 0.1)
- `update_freq`: Update frequency (1024 to 4096)

### Multi-Asset Training

Train on multiple assets:

```python
symbols = ["BTC-USD", "ETH-USD", "SPY"]
for symbol in symbols:
    train(symbol=symbol, ...)
```

## Comparison: RL vs Rule-Based

| Metric | Rule-Based (MR-SR) | PPO Agent |
|--------|-------------------|-----------|
| Returns | Baseline | Optimized |
| Adaptability | Fixed rules | Learns patterns |
| Risk Management | Static thresholds | Dynamic |
| Training Time | None | Hours |
| Interpretability | High | Lower |

## Next Steps

1. **Extend State Space**: Add order book data, news sentiment
2. **Multi-Asset**: Train on portfolio of assets
3. **Continuous Actions**: Position sizing instead of fixed amounts
4. **Other Algorithms**: Try A2C, SAC, TD3
5. **Ensemble**: Combine RL with MR-SR agent
6. **Live Trading**: Deploy trained agent (with caution!)

## References

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Deep Reinforcement Learning for Trading](https://arxiv.org/abs/1911.10107)
- [OpenAI Gym](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Troubleshooting

**CUDA out of memory:**
- Reduce `hidden_dim` in ActorCritic
- Reduce `update_freq`
- Use smaller batch sizes

**Training unstable:**
- Lower learning rate
- Increase `k_epochs`
- Adjust reward scaling

**Agent not learning:**
- Check reward function
- Increase `entropy_coef` for more exploration
- Verify state normalization
- Try longer training (more episodes)

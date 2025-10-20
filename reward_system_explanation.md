# Simple Reward-Only System: Mathematical Deep Dive

## Overview

The simple reward-only system transforms portfolio returns into non-negative rewards that guide agent learning toward profitable, stable trading strategies.

## Step-by-Step Calculation

### STEP 1: Calculate Portfolio Return

```python
step_return = (current_value - previous_value) / previous_value
```

**Example:**
- Previous: $100,000
- Current: $100,500
- Return: 0.005 (0.5%)

---

### STEP 2: Base Reward from Return

The system uses **asymmetric weighting** for gains vs losses:

```python
if step_return > 0:
    # GAINS: Full multiplier (default: 1000x)
    base_reward = step_return * gain_weight
else:
    # LOSSES: Reduced multiplier (default: 100x)
    base_reward = step_return * loss_weight
```

**Why asymmetric?**
- We want to **strongly encourage** profitable trades (high gain_weight)
- We want to **gently discourage** losses without harsh punishment (low loss_weight)
- The 10:1 ratio (1000:100) means gains are 10x more "rewarding" than losses are "un-rewarding"

**Example A: Profitable Trade (+0.5%)**
```
step_return = 0.005
base_reward = 0.005 × 1000 = 5.0
```

**Example B: Losing Trade (-0.3%)**
```
step_return = -0.003
base_reward = -0.003 × 100 = -0.3
```

Notice: A +0.5% gain gives reward of 5.0, but a -0.5% loss only gives -0.5 (10x difference!)

---

### STEP 3: Risk Adjustment (Stability Factor)

Now we adjust for volatility using a **multiplicative stability factor**:

```python
if len(returns_history) >= min_samples:
    volatility = std_dev(returns_history)  # Last 30 returns

    # Stability factor: reduces reward for volatile performance
    stability_factor = 1.0 / (1.0 + volatility × volatility_sensitivity)

    risk_adjusted_reward = base_reward × stability_factor
else:
    risk_adjusted_reward = base_reward  # Not enough data yet
```

**The Stability Factor Formula:**

```
stability_factor = 1 / (1 + σ × k)

Where:
- σ = volatility (standard deviation of recent returns)
- k = volatility_sensitivity (default: 20.0)
```

**How it works:**
- **Low volatility** (σ ≈ 0) → stability_factor ≈ 1.0 → **Full reward**
- **Medium volatility** (σ = 0.05) → stability_factor = 1/(1+1.0) = 0.5 → **Half reward**
- **High volatility** (σ = 0.10) → stability_factor = 1/(1+2.0) = 0.33 → **Third reward**

**Example C: Stable Gains**
```
base_reward = 5.0
volatility = 0.01 (1% - very stable)
stability_factor = 1 / (1 + 0.01×20) = 1 / 1.2 = 0.833
risk_adjusted_reward = 5.0 × 0.833 = 4.17
```

**Example D: Volatile Gains**
```
base_reward = 5.0
volatility = 0.08 (8% - very volatile)
stability_factor = 1 / (1 + 0.08×20) = 1 / 2.6 = 0.385
risk_adjusted_reward = 5.0 × 0.385 = 1.92
```

`✶ Key Insight: Same gain, but stable trading gets 4.17 reward while volatile trading only gets 1.92!`

---

### STEP 4: Non-Negativity Constraint

Finally, we ensure **all rewards are non-negative**:

```python
final_reward = max(0.0, risk_adjusted_reward)
```

**Why this floor matters:**

This is the KEY innovation that prevents the adversarial gradient problem!

**Traditional system:**
```
Bad trade → Negative reward → Gradient says "avoid this action"
Multiple bad trades → Large negative rewards → Agent learns "avoid ALL trading"
```

**Reward-only system:**
```
Bad trade → Small positive (or zero) reward → Gradient says "this action is okay but not great"
Good trade → Large positive reward → Gradient says "THIS action is excellent!"
```

The agent learns to **seek better actions** rather than **avoid bad actions**.

**Example E: Volatile Loss**
```
base_reward = -0.3 (from -0.3% loss)
volatility = 0.06
stability_factor = 1 / (1 + 0.06×20) = 0.43
risk_adjusted_reward = -0.3 × 0.43 = -0.129
final_reward = max(0.0, -0.129) = 0.0  ← FLOOR APPLIED
```

---

## Complete Examples

### Scenario 1: Perfect Trading (Stable Gains)

**Episode with consistent +0.5% daily gains, low volatility:**

```
Day 1:  Return +0.5%, Vol=0.01 → Reward = 5.0 × 0.83 = 4.17
Day 2:  Return +0.5%, Vol=0.01 → Reward = 5.0 × 0.83 = 4.17
Day 3:  Return +0.5%, Vol=0.01 → Reward = 5.0 × 0.83 = 4.17
...
Total Reward: ~4.17 per day (HIGH)
```

**Agent Learning:** "This policy is EXCELLENT! Keep doing this!"

---

### Scenario 2: Volatile Trading (Big Swings)

**Episode alternating +2% / -1% daily:**

```
Day 1:  Return +2.0%, Vol=0.03 → Base=20.0, SF=0.63 → Reward = 12.6
Day 2:  Return -1.0%, Vol=0.03 → Base=-1.0, SF=0.63 → Reward = 0.0 (floored)
Day 3:  Return +2.0%, Vol=0.05 → Base=20.0, SF=0.50 → Reward = 10.0
Day 4:  Return -1.0%, Vol=0.05 → Base=-1.0, SF=0.50 → Reward = 0.0 (floored)
...
Average Reward: ~5.6 per day (MEDIUM)
```

**Agent Learning:** "This policy gives some reward, but not as much as stable gains."

---

### Scenario 3: Consistent Losses

**Episode with -0.3% daily losses:**

```
Day 1:  Return -0.3%, Vol=0.00 → Base=-0.03, SF=1.0 → Reward = 0.0 (floored)
Day 2:  Return -0.3%, Vol=0.00 → Base=-0.03, SF=1.0 → Reward = 0.0 (floored)
Day 3:  Return -0.3%, Vol=0.00 → Base=-0.03, SF=1.0 → Reward = 0.0 (floored)
...
Total Reward: 0.0 per day (LOW)
```

**Agent Learning:** "This policy gives almost no reward. I should explore other actions."

---

## Why This Works: The Gradient Perspective

### Traditional Penalty-Based System

**Policy Gradient Update:**
```
θ_new = θ_old + α × ∇_θ log π(a|s) × R

When R is negative (penalty):
- Gradient DECREASES probability of action a in state s
- With many penalties: Agent learns to minimize action-taking
- Result: Conservative, risk-averse policy (HOLD forever)
```

### Reward-Only System

**Policy Gradient Update:**
```
θ_new = θ_old + α × ∇_θ log π(a|s) × R

When R is always ≥ 0:
- Gradient ALWAYS increases or maintains probability
- With varying rewards: Agent learns to prefer high-reward actions
- Result: Active policy that explores and exploits profitable opportunities
```

**The key difference:**
- **Penalties** create an adversarial "don't do this" signal
- **Reduced rewards** create a comparative "there's something better" signal

---

## Hyperparameter Tuning Guide

### gain_weight (default: 1000.0)
- **Higher** → Agent more excited by gains, learns faster from profitable trades
- **Lower** → More conservative learning
- **Sweet spot:** 500-2000 depending on return scale

### loss_weight (default: 100.0)
- **Higher** → Losses have more impact (closer to penalty-based)
- **Lower** → Losses almost ignored (agent may be reckless)
- **Sweet spot:** 50-200, maintaining 5:1 to 20:1 ratio with gain_weight

### volatility_sensitivity (default: 20.0)
- **Higher** → Stronger preference for stable returns
- **Lower** → Allows more volatility
- **Sweet spot:** 10-30, depends on market characteristics

### lookback_window (default: 30)
- **Larger** → Smoother volatility estimates, slower adaptation
- **Smaller** → More reactive to recent volatility
- **Sweet spot:** 20-50 for daily data

---

## Comparison to Sharpe-Optimized System

### Sharpe-Optimized (FAILED)

**Reward Components:**
1. ✅ Sharpe ratio reward (return/volatility)
2. ❌ Drawdown penalty (harsh!)
3. ❌ Trading cost penalty
4. ❌ Exploration penalty

**Total Reward:**
```
R = sharpe_reward - drawdown_penalty - cost_penalty - exploration_penalty
```

**Result:** Sum of penalties overwhelmed rewards → Agent stopped trading

### Simple Reward-Only (NEW)

**Reward Components:**
1. ✅ Asymmetric return reward (gain_weight / loss_weight)
2. ✅ Stability factor (multiplicative, never negative)
3. ✅ Non-negativity floor

**Total Reward:**
```
R = max(0, base_reward × stability_factor)
```

**Expected Result:** Agent maintains active trading, learns to prefer stable gains

---

## Validation Metrics

The system still computes standard metrics for evaluation:

```python
def get_episode_metrics(self) -> Dict:
    return {
        'sharpe_ratio': (mean_return - risk_free) / volatility × √252,
        'total_return': (final_value - initial_value) / initial_value,
        'volatility': std(returns) × √252,
        'max_drawdown': max(peak - valley) / peak,
        'trade_count': number_of_trades
    }
```

**Key point:** We use Sharpe ratio for **evaluation**, not **training**!

This separates:
- **Training objective:** Maximize cumulative reward (our custom reward-only system)
- **Evaluation objective:** Achieve high risk-adjusted returns (Sharpe ratio)

---

## Expected Outcomes

### Success Criteria

**Minimum Success:**
- Validation Sharpe > 0.0 (agent trades during validation)
- Positive episode rewards
- Learning curve shows improvement

**Good Success:**
- Validation Sharpe > 0.5
- Consistent positive returns
- Agent actively trades (not HOLD-only)

**Excellent Success:**
- Validation Sharpe > 1.0
- High stability factor (low volatility)
- Sustainable trading strategy

### Failure Modes

**If training still fails:**

1. **All rewards still zero** → gain_weight too low, increase to 2000-5000
2. **Reward variance too high** → volatility_sensitivity too low, increase to 30-40
3. **Agent still converges to HOLD** → loss_weight too high, decrease to 50 or lower
4. **No learning progress** → PPO hyperparameters (learning rate, eps_clip) need tuning

---

## Next Steps After Training

Once training completes, we'll analyze:

1. **Reward Distribution:**
   - Are rewards positive? (Success indicator)
   - Is there variance? (Agent exploring different strategies)
   - Is there improvement over episodes? (Learning happening)

2. **Validation Performance:**
   - Are validation Sharpes positive? (Agent trades)
   - Are they improving? (Learning transfers to unseen data)
   - Are they stable? (Not overfitting)

3. **Trading Behavior:**
   - Trade frequency (Active vs passive)
   - Win rate (Profitable vs unprofitable trades)
   - Position sizing (Risk management)

---

## Theoretical Foundation

### Why Reward-Only Systems Work

**From Reinforcement Learning Theory:**

The policy gradient theorem states:
```
∇_θ J(θ) = E[∇_θ log π(a|s,θ) × Q^π(s,a)]
```

Where Q^π(s,a) is the expected return from state s taking action a.

**Key insight:** The gradient magnitude depends on Q^π, not its sign!

**Traditional approach:**
- Negative Q^π → Decrease action probability
- Positive Q^π → Increase action probability

**Reward-only approach:**
- Small Q^π → Slightly increase action probability
- Large Q^π → Strongly increase action probability

Both converge to optimal policy, but reward-only:
- ✅ Maintains exploration (no actions become "forbidden")
- ✅ Avoids catastrophic forgetting (no harsh reversals)
- ✅ Smoother learning dynamics (positive-only gradients)

---

## Conclusion

The simple reward-only system redesigns the reward signal to **guide** rather than **punish** the agent. By ensuring all rewards are non-negative and scaling them based on both return magnitude and stability, we create a training environment where the agent:

1. **Actively explores** trading strategies
2. **Learns to prefer** profitable trades
3. **Discovers** that stable gains yield higher rewards
4. **Avoids** the adversarial dynamics that caused the Sharpe-optimized system to fail

This is a **philosophy shift**: From "don't lose money" to "find the best way to make money."

---

*Generated while waiting for simple_reward training to complete...*

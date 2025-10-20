# Market Regime Taxonomy for Bitcoin Trading

**Date:** October 17, 2025
**Purpose:** Define market regimes for regime-specific agent specialization
**Status:** Phase 2.1 - Regime Design Complete

---

## Executive Summary

This document defines a comprehensive taxonomy of market regimes for cryptocurrency trading. Each regime requires different trading strategies, and our ensemble system will train specialized agents for each regime.

**Key Design Principles:**
1. **Mutually exclusive:** Each period belongs to exactly one regime
2. **Quantitatively defined:** Clear mathematical criteria for classification
3. **Actionable:** Each regime suggests specific trading strategies
4. **Historically validated:** Based on analysis of 2017-2024 Bitcoin data

---

## Primary Regime Classification (Trend Direction)

### 1. Bull Market
**Definition:** Sustained upward price movement with higher highs and higher lows

**Quantitative Criteria:**
- 30-day SMA slope > 0
- Price > 50-day SMA
- 30-day return > +5%
- Majority of days (>60%) are positive returns

**Trading Characteristics:**
- Buy-and-hold works well
- Pullbacks are buying opportunities
- Risk: Getting whipsawed near local tops
- Optimal strategy: Accumulate on dips, hold through volatility

**Historical Examples (2017-2024):**
- 2017 Q4: $5,000 → $20,000 (parabolic bull run)
- 2020 Q4: $10,000 → $29,000 (post-COVID recovery)
- 2021 Q1-Q2: $29,000 → $64,000 (mega bull run)
- 2024 Q4: $60,000 → $106,000 (ETF-driven rally)

**Performance Requirements:**
- Sharpe ratio > 1.5
- Capture ratio > 80% (capture 80%+ of upside)
- Maximum drawdown < 10%

---

### 2. Bear Market
**Definition:** Sustained downward price movement with lower lows and lower highs

**Quantitative Criteria:**
- 30-day SMA slope < 0
- Price < 50-day SMA
- 30-day return < -5%
- Majority of days (>60%) are negative returns
- High correlation with Bitcoin dominance increase

**Trading Characteristics:**
- Buy-and-hold loses money
- Short-term mean reversion opportunities
- Risk: Catching falling knives
- Optimal strategy: Capital preservation, small position sizes

**Historical Examples (2017-2024):**
- 2018 Full Year: $20,000 → $3,200 (-84% bear market)
- 2022 Full Year: $47,000 → $15,787 (-67% bear market)
- 2021 Q2-Q3: $64,000 → $29,000 (-55% correction)

**Performance Requirements:**
- Sharpe ratio > 0.5 (just stay positive)
- Maximum drawdown < 5%
- Minimize losses (returns > -5% annually)

---

### 3. Sideways / Consolidation
**Definition:** Range-bound price action without clear trend

**Quantitative Criteria:**
- |30-day SMA slope| < small threshold
- Price oscillates around 50-day SMA (±5%)
- 30-day return between -5% and +5%
- High/Low range < 20% of midpoint

**Trading Characteristics:**
- Mean reversion works best
- Buy support, sell resistance
- Risk: Trend breakout in either direction
- Optimal strategy: Range trading, quick profits

**Historical Examples (2017-2024):**
- 2019 Q2-Q3: $8,000-$12,000 consolidation
- 2023 Q2: $25,000-$31,000 consolidation after bear market bottom

**Performance Requirements:**
- Sharpe ratio > 1.0
- Win rate > 55%
- Profit factor > 2.0 (wins must be bigger than losses)

---

## Secondary Regime Classification (Volatility)

These overlay on top of primary regimes to create compound classifications.

### 4. Low Volatility
**Definition:** Stable price action with small daily moves

**Quantitative Criteria:**
- 20-day historical volatility < 30% annualized
- Average true range (ATR) < 3% of price
- Consecutive days without >5% moves > 10

**Trading Implications:**
- Mean reversion strategies work
- Lower position sizes (smaller profit potential)
- Tight stop losses (small risk)

**Historical Examples:**
- 2019 Q3-Q4: Post-consolidation stability
- 2023 Q3: Post-bear market calm

---

### 5. High Volatility
**Definition:** Large daily price swings, increased uncertainty

**Quantitative Criteria:**
- 20-day historical volatility > 80% annualized
- Average true range (ATR) > 8% of price
- Multiple >5% daily moves in past week

**Trading Implications:**
- Momentum strategies work
- Larger position sizes (bigger profit potential)
- Wider stop losses (avoid noise)
- Risk of sudden reversals

**Historical Examples:**
- 2017 Q4: Parabolic rise volatility
- 2020 March: COVID crash volatility
- 2021 Q2: Peak euphoria volatility

---

### 6. Crash / Extreme Volatility
**Definition:** Sudden, severe price decline with panic selling

**Quantitative Criteria:**
- Single day drop > 15%
- 7-day return < -30%
- Volume spike > 3x average
- Fear & Greed Index < 20 (extreme fear)

**Trading Implications:**
- Capital preservation only
- No new positions until stabilization
- Wait for volatility compression
- Opportunity for long-term accumulation near bottom

**Historical Examples:**
- 2020 March 12-13: "Black Thursday" -50% in 2 days
- 2021 May 19: -30% flash crash
- 2022 May: LUNA collapse contagion
- 2022 November: FTX collapse

---

## Tertiary Regime Classification (Trend Strength)

### 7. Strong Trend
**Definition:** Clear, sustained directional movement

**Quantitative Criteria:**
- ADX (Average Directional Index) > 25
- Price consistently above/below moving averages
- Minimal pullbacks (<10% corrections)

**Trading Implications:**
- Trend-following works best
- Avoid mean reversion
- Hold winners longer

---

### 8. Weak Trend / Choppy
**Definition:** Indecisive price action, frequent reversals

**Quantitative Criteria:**
- ADX < 20
- Price whipsaws around moving averages
- Frequent direction changes

**Trading Implications:**
- Avoid trend-following
- Quick profit-taking
- Tight stops

---

## Compound Regime Examples

Real markets exhibit combinations of these regimes:

1. **Bull + High Volatility + Strong Trend**
   - Example: 2017 Q4 parabolic run
   - Strategy: Aggressive accumulation, wide stops, hold winners

2. **Bear + Low Volatility + Weak Trend**
   - Example: 2018 Q3-Q4 slow bleed
   - Strategy: Stay in cash, wait for capitulation

3. **Sideways + Low Volatility + Choppy**
   - Example: 2019 Q3 consolidation
   - Strategy: Range trading, quick profits, neutral bias

4. **Bull + Crash Volatility**
   - Example: 2021 May flash crash during bull market
   - Strategy: Buy the dip aggressively, temporary correction

---

## Regime-Specific Agent Specialization

Based on this taxonomy, we propose **3 specialized agents**:

### Agent 1: Bull Market Specialist
**Training Data:** 2017 Q4, 2020 Q3-Q4, 2021 Q1-Q2, 2024 Q1, Q4
**Objective:** Maximize Sharpe ratio in bull markets
**Reward Function:** Sharpe ratio + capture ratio bonus
**Expected Sharpe:** > 1.5

### Agent 2: Bear Market Specialist
**Training Data:** 2018 (all), 2022 (all), 2021 Q2-Q3
**Objective:** Capital preservation, minimize losses
**Reward Function:** Sharpe ratio - drawdown penalty
**Expected Sharpe:** > 0.5

### Agent 3: Sideways/Volatile Specialist
**Training Data:** 2019 Q2-Q3, 2023 Q2-Q3, mixed volatility periods
**Objective:** Profit from range-bound and choppy markets
**Reward Function:** Sharpe ratio + profit factor bonus
**Expected Sharpe:** > 1.0

---

## Regime Detection Features

To automatically classify regimes, we will use these features:

### Price-Based Features
1. 30-day return (trend direction)
2. 50-day SMA slope (trend strength)
3. Price vs MA crossover (position in trend)
4. Higher highs / lower lows count (trend structure)

### Volatility Features
5. 20-day historical volatility (volatility level)
6. ATR (Average True Range) (absolute volatility)
7. Bollinger Band width (volatility expansion)
8. GARCH volatility forecast (expected volatility)

### Momentum Features
9. RSI (Relative Strength Index) (overbought/oversold)
10. MACD (Moving Average Convergence Divergence) (momentum)
11. Stochastic Oscillator (momentum in range)
12. Rate of Change (acceleration)

### Trend Strength Features
13. ADX (Average Directional Index) (trend strength)
14. +DI/-DI (directional indicators) (trend direction)
15. Moving average alignment (SMA 10 > 20 > 50 > 200)

### Volume Features
16. Volume ratio (current / 20-day average)
17. On-Balance Volume (OBV) (volume momentum)
18. Volume-weighted return (price-volume relationship)

### External Indicators (if available)
19. Fear & Greed Index (market sentiment)
20. Bitcoin Dominance (altcoin vs BTC)
21. Funding rates (futures market sentiment)

---

## Regime Transition Rules

Markets don't switch regimes instantly. We define transition rules:

### Bull → Bear Transition
- 30-day return turns negative
- Price breaks below 50-day SMA
- Volume spike on down days
- **Confirmation period:** 5 consecutive days meeting criteria

### Bear → Bull Transition
- 30-day return turns positive
- Price breaks above 50-day SMA
- Volume spike on up days
- **Confirmation period:** 5 consecutive days meeting criteria

### Any → Sideways Transition
- 30-day return narrows to ±5%
- Volatility contracts below threshold
- **Confirmation period:** 10 consecutive days

### Any → Crash
- **Immediate switch** on single day >15% drop
- Remains in crash mode until 5-day volatility normalizes

---

## Regime Labeling for Training Data

We will programmatically label all historical data with regime classifications:

```python
# Example output format
date,price,regime_primary,regime_volatility,regime_strength
2017-12-17,19500,BULL,HIGH,STRONG
2018-06-15,6500,BEAR,LOW,WEAK
2019-08-10,11500,SIDEWAYS,LOW,CHOPPY
2020-03-12,5000,CRASH,EXTREME,N/A
2024-12-01,96000,BULL,MEDIUM,STRONG
```

This labeled dataset enables:
1. Supervised regime detection training
2. Regime-specific agent training
3. Backtesting regime switching strategies

---

## Validation Metrics

To ensure regime classification quality:

### Stability Metrics
- **Average regime duration:** Should be >14 days (avoid excessive switching)
- **Regime purity:** >80% of days in regime match expected characteristics
- **Transition smoothness:** <10% of days in transition state

### Predictive Metrics
- **Forward-looking accuracy:** Regime at day T predicts next 7-day regime 75%+ accuracy
- **Economic significance:** Different regimes have statistically different returns (p < 0.05)

### Coverage Metrics
- **Data coverage:** All historical days assigned a regime (no gaps)
- **Regime distribution:** No single regime >60% of data (avoid overfitting)

---

## Implementation Plan

### Step 1: Feature Engineering (2 hours)
- Calculate all 21 regime detection features
- Add to TradingEnv observation space
- Validate feature calculations against known regimes

### Step 2: Rule-Based Classifier (2 hours)
- Implement quantitative criteria as decision rules
- Label entire historical dataset
- Validate against known market periods

### Step 3: ML Classifier (4 hours)
- Train Random Forest or XGBoost classifier
- Use rule-based labels as training data
- Achieve >90% agreement with rule-based system
- Add probabilistic outputs for confidence

### Step 4: Ensemble Detector (2 hours)
- Combine rule-based + ML classifier
- Use voting or weighted average
- Add regime transition smoothing
- Final regime assignment with confidence scores

---

## Expected Outcomes

### Regime Classification Quality
- 90%+ agreement between rule-based and ML classifiers
- <5% of time in uncertain/transition states
- Regime changes coincide with known market events

### Agent Performance by Regime
- Bull agent Sharpe > 1.5 on bull-labeled data
- Bear agent Sharpe > 0.5 on bear-labeled data
- Sideways agent Sharpe > 1.0 on sideways-labeled data

### Ensemble Performance
- Overall Sharpe > 1.0 across all regimes
- Maximum drawdown < 15%
- Profitable in 80% of years (2017-2024 backtest)

---

## Risk Considerations

### Regime Detection Errors
- **False bull signal:** Agent takes risk in bear market → Large losses
- **False bear signal:** Agent misses bull market → Opportunity cost
- **Excessive switching:** Transaction costs erode profits

**Mitigation:** Use confidence thresholds, transition periods, and performance monitoring

### Look-Ahead Bias
- Features must use only historical data (no future information)
- Regime labels must be calculable in real-time
- Moving averages must use lagged data

**Mitigation:** Strict temporal validation, walk-forward testing

### Overfitting
- Complex regime rules fit historical noise
- ML classifier memorizes training data

**Mitigation:** Simple quantitative rules, cross-validation, out-of-sample testing

---

## References & Literature

1. **"Adaptive Asset Allocation"** (Butler et al., 2012)
   - Regime-based portfolio switching
   - Quantitative regime definitions

2. **"The Many Colors of MACD"** (Thorp, 2015)
   - Technical indicators for regime detection
   - Backtesting methodologies

3. **Bitcoin-Specific Research:**
   - [Bitcoin volatility regimes](https://arxiv.org/abs/1801.05740)
   - [Crypto market cycles](https://www.sciencedirect.com/science/article/pii/S1544612319301618)

---

## Next Steps

1. ✅ **Regime taxonomy complete** (this document)
2. ⏳ **Implement regime detector** (Phase 2.2)
3. ⏳ **Validate detector accuracy** (Phase 2.3)
4. ⏳ **Train regime-specific agents** (Phase 3)
5. ⏳ **Build ensemble orchestrator** (Phase 3.3)

---

*Document created: October 17, 2025*
*Based on: 2017-2024 Bitcoin historical analysis*
*Validation status: Pending implementation*

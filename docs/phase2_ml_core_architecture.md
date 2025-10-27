# Phase 2: ML Core Architecture

## Overview

The ML Core implements a hierarchical reinforcement learning system for adaptive trading strategy selection. The architecture consists of three specialized services that work together to detect market regimes, select optimal strategies, and coordinate distributed model training.

## System Architecture

```
Market Data (Kafka)
    ↓
[Regime Detection Service] (DPGMM)
    ↓ regime_events
[Meta-Controller Service] (PPO RL)
    ↓ strategy_decisions
[Trading Execution] (Phase 3)

[Training Orchestrator Service] (Celery)
    ↓ training_status
[MLflow Experiment Tracking]
```

## Service Details

### 1. Regime Detection Service

**Purpose**: Probabilistic classification of market conditions

**Technology Stack**:
- Scikit-learn BayesianGaussianMixture (DPGMM)
- Kafka consumer/producer
- Redis model caching

**Input**: `market_data` topic
- Features: close, prev_close, volatility, volume_ratio, trend_strength

**Output**: `regime_events` topic
- regime: {trending, mean_reverting, volatile, stable}
- regime_id: 0-3
- confidence: float [0, 1]
- timestamp: ISO 8601

**Key Features**:
- Online learning (refits every 100 samples)
- Bootstrap mode for cold start
- Model caching with 1-hour TTL
- Sliding window of 1000 samples

**Implementation**: `services/regime_detection/regime_detector.py`

**Deployment**:
- Container: ultrathink-regime-detection
- Dependencies: Kafka cluster, Redis
- Status: ✅ Deployed and validated

---

### 2. Meta-Controller Service

**Purpose**: Hierarchical RL for dynamic strategy selection

**Technology Stack**:
- PyTorch 2.1.0 with CUDA 12.1
- PPO (Proximal Policy Optimization)
- MLflow experiment tracking
- Redis model caching

**Input**: `regime_events` topic
- 8-dimensional state space:
  - regime_id (normalized)
  - confidence
  - returns
  - volatility
  - volume_ratio
  - trend_strength
  - recent_pnl
  - strategy_performance

**Output**: `strategy_decisions` topic
- selected_strategy: {trend_following, mean_reversion, volatility_arbitrage, momentum, market_making}
- action_idx: 0-4
- confidence: float
- state: list[float]

**Architecture**:
- Policy Network: 8 → 128 → 128 → 5 (actor) / 1 (critic)
- Activation: ReLU
- Optimizer: Adam (lr=3e-4)
- Gamma: 0.99
- Epsilon clip: 0.2

**Key Features**:
- GPU-accelerated training
- Experience replay buffer
- PPO clipped objective
- Entropy bonus for exploration
- Gradient clipping (max_norm=0.5)
- Model checkpointing to Redis
- Real-time MLflow logging

**Implementation**: `services/meta_controller/meta_controller.py`

**Deployment**:
- Container: ultrathink-meta-controller
- Dependencies: Kafka cluster, Redis, MLflow
- GPU: NVIDIA RTX 5070 (1x GPU reservation)
- Status: 🔨 Building

---

### 3. Training Orchestrator Service

**Purpose**: Distributed ML training coordination

**Technology Stack**:
- Celery 5.3.4 with Redis broker
- PyTorch 2.1.0 with CUDA 12.1
- MLflow experiment tracking
- Kafka status publishing

**Celery Tasks**:
1. `fetch_training_data` - Pull data from Data Service API
2. `preprocess_features` - Normalize and engineer features
3. `train_model` - Execute training with hyperparameters
4. `publish_training_status` - Aggregate results to Kafka

**Workflow**:
```python
orchestrate_training(symbol, start_date, end_date, hyperparams_list)
    → fetch_training_data
    → preprocess_features
    → group([train_model(hp) for hp in hyperparams_list])  # Parallel
    → publish_training_status
```

**Key Features**:
- Distributed task execution
- Parallel hyperparameter search
- GPU-accelerated training
- MLflow experiment tracking
- Redis result backend
- Kafka status streaming
- Automatic model versioning

**Implementation**: `services/training_orchestrator/training_orchestrator.py`

**Deployment**:
- Container: ultrathink-training-orchestrator
- Dependencies: Kafka cluster, Redis, MLflow
- GPU: NVIDIA RTX 5070 (1x GPU reservation)
- Worker concurrency: 2
- Status: ✅ Ready to deploy

---

## Data Flow

### Real-Time Inference Pipeline

1. **Market Data Ingestion**
   - Source: Exchange APIs
   - Topic: `market_data`
   - Format: JSON with OHLCV + derived features

2. **Regime Classification**
   - Service: regime-detection
   - Algorithm: DPGMM with 4 clusters
   - Latency: <10ms
   - Output: `regime_events`

3. **Strategy Selection**
   - Service: meta-controller
   - Algorithm: PPO policy network
   - Latency: <5ms (GPU inference)
   - Output: `strategy_decisions`

4. **Strategy Execution** (Phase 3)
   - Service: execution-engine
   - Consumes: `strategy_decisions`
   - Produces: `trading_actions`

### Offline Training Pipeline

1. **Training Initiation**
   - Trigger: Scheduled or API call
   - Task: `orchestrate_training`
   - Parallelism: N hyperparameter configs

2. **Data Preparation**
   - Fetch historical data
   - Feature engineering
   - Train/test split
   - Normalization

3. **Model Training**
   - Parallel training jobs
   - GPU acceleration
   - MLflow tracking
   - Model checkpointing

4. **Model Selection**
   - Aggregate metrics
   - Select best model
   - Publish to Redis
   - Update production

---

## Infrastructure

### Kafka Topics

| Topic | Partitions | Replication | Retention | Purpose |
|-------|-----------|-------------|-----------|---------|
| `market_data` | 5 | 2 | 7 days | Raw market data |
| `regime_events` | 3 | 2 | 7 days | Regime classifications |
| `strategy_decisions` | 3 | 2 | 7 days | Strategy selections |
| `training_status` | 2 | 2 | 7 days | Training progress |

### Redis Databases

| DB | Purpose | TTL | Size |
|----|---------|-----|------|
| 0 | Data Service cache | 10 min | 2GB |
| 1 | Regime models | 1 hour | 2GB |
| 2 | Meta-controller models | 1 hour | 2GB |
| 3 | Celery broker/backend | N/A | 2GB |

### GPU Allocation

- RTX 5070 (16GB VRAM)
  - Meta-controller: Inference + online learning
  - Training orchestrator: Batch training

### MLflow Experiment Structure

```
ultrathink_experiments/
├── strategy_selection/      # Meta-controller runs
│   ├── policy_loss
│   ├── value_loss
│   ├── entropy
│   └── mean_return
└── strategy_training/       # Training orchestrator runs
    ├── train_loss
    ├── val_loss
    ├── val_accuracy
    └── models/
```

---

## Performance Metrics

### Latency Targets

- Regime Detection: <10ms (p99)
- Strategy Selection: <5ms (p99)
- End-to-end (market data → decision): <20ms (p99)

### Throughput Targets

- Market data ingestion: 10,000 msgs/sec
- Regime classifications: 5,000 events/sec
- Strategy decisions: 5,000 events/sec

### Training Performance

- Single model training: ~2-5 minutes
- Parallel hyperparameter search (N=10): ~5-10 minutes
- GPU utilization target: >80%

---

## Monitoring & Observability

### Key Metrics

1. **Regime Detection**
   - Classification confidence distribution
   - Regime transition frequency
   - Model refit latency
   - Feature extraction errors

2. **Meta-Controller**
   - Policy loss trend
   - Value function accuracy
   - Strategy selection distribution
   - Reward signal statistics

3. **Training Orchestrator**
   - Task queue depth
   - Worker utilization
   - Training job duration
   - Model convergence rate

### Health Checks

All services expose health endpoints:
- Kafka connectivity
- Redis availability
- MLflow reachability
- GPU device status

---

## Deployment Status

| Service | Status | Container | GPU | Dependencies |
|---------|--------|-----------|-----|--------------|
| Regime Detection | ✅ DEPLOYED | ultrathink-regime-detection | No | Kafka, Redis |
| Meta-Controller | 🔨 BUILDING | ultrathink-meta-controller | Yes | Kafka, Redis, MLflow |
| Training Orchestrator | ✅ READY | ultrathink-training-orchestrator | Yes | Kafka, Redis, MLflow |

---

## Next Steps (Phase 3)

1. API & Risk Management
   - FastAPI risk service
   - Position sizing
   - Stop-loss management
   - Portfolio optimization

2. Execution Engine
   - Strategy execution logic
   - Order management
   - Fill handling
   - Performance tracking

3. Integration Testing
   - End-to-end latency
   - Failover scenarios
   - Load testing
   - Chaos engineering

---

**Generated**: 2025-10-24T14:35:00Z
**Phase**: 2 (ML Core)
**Status**: In Progress

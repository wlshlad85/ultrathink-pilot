# Trading System Architectural Enhancement - Technical Specification

**Date:** 2025-10-21
**Status:** Draft
**PRD Reference:** [PRD.md](./PRD.md)

---

## Architecture Overview

Microservices architecture with event-driven communication, separating data pipeline, training orchestration, inference, and risk management into independent services

---

## System Design

### Component Architecture

**High-Level Architecture:**
```
┌─────────────────────────────────────────────────────────────────────┐
│                        Trading System Core                          │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │   Market     │───▶│    Data      │───▶│  Inference   │        │
│  │   Data       │    │   Service    │    │   Service    │        │
│  │   Ingestion  │    │   (Redis)    │    │   (Models)   │        │
│  └──────────────┘    └──────────────┘    └──────┬───────┘        │
│                                                   │                 │
│                                                   ▼                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │  Forensics   │◀───│    Kafka     │◀───│     Risk     │        │
│  │  Consumer    │    │   Message    │    │   Manager    │        │
│  │   (Async)    │    │    Bus       │    │  (Limits)    │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                   │                 │
│                                                   ▼                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │  Training    │───▶│  TimescaleDB │    │  Execution   │        │
│  │ Orchestrator │    │  (Metrics)   │    │   Engine     │        │
│  │  (MLflow)    │    │              │    │              │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

**Component Specifications:**

1. **Data Service** (Priority 1 - Foundation)
   - **Purpose:** Centralized market data preprocessing with feature engineering abstraction
   - **Technology:** Python FastAPI service with Redis cache layer
   - **Responsibilities:**
     - Ingest raw market data from existing feeds
     - Apply unified feature engineering pipeline (eliminates 3+ redundant implementations)
     - Serve preprocessed features via REST API with <20ms P95 latency
     - Cache computed features in Redis (90%+ hit rate target)
   - **Key Innovation:** Repository pattern separating data access from business logic, enabling consistent feature engineering across training and inference

2. **Training Orchestrator** (Priority 1)
   - **Purpose:** Manage concurrent model training with resource allocation
   - **Technology:** Python Celery + MLflow for experiment tracking
   - **Responsibilities:**
     - Queue and schedule training jobs across GPU resources
     - Log metrics to TimescaleDB (replacing SQLite bottleneck)
     - Version control model checkpoints with MLflow registry
     - Automated checkpoint archival and garbage collection (limit 500MB/day growth)
   - **Key Innovation:** Concurrent write support eliminates 2-3 process limitation, enables 20+ parallel experiments

3. **Inference Service** (Priority 2)
   - **Purpose:** Low-latency prediction API with A/B testing and model registry
   - **Technology:** Python FastAPI with model serving framework (TorchServe/TF Serving)
   - **Responsibilities:**
     - Load models from MLflow registry with warm cache
     - Execute inference with P95 <50ms target
     - Support A/B traffic splitting for model comparison
     - Emit prediction events to Kafka for forensics
   - **Key Innovation:** Decoupled from model training, supports canary deployments and instant rollback

4. **Meta-Controller** (Priority 2 - Research Component)
   - **Purpose:** Hierarchical RL agent for dynamic strategy selection
   - **Technology:** Python with options framework RL implementation
   - **Responsibilities:**
     - Observe regime probabilities and specialist model performance
     - Learn to select and blend specialist strategies (vs. hard-coded regime detection)
     - Output weighted ensemble decisions (smooth transitions vs. discontinuities)
     - Adapt strategy mix based on recent performance feedback
   - **Key Innovation:** Replaces rigid regime-based routing with learned meta-policy, eliminates 15% portfolio disruption

5. **Risk Manager** (Priority 3)
   - **Purpose:** Portfolio-level risk constraint enforcement
   - **Technology:** Python service with in-memory state management
   - **Responsibilities:**
     - Validate proposed trades against position limits (25% concentration max)
     - Calculate real-time portfolio metrics (correlation, VaR, exposure)
     - Enforce hierarchical risk parity allocations
     - Provide risk check API with <10ms P95 latency
   - **Key Innovation:** First portfolio-level risk layer, prevents concentration violations

6. **Forensics Consumer** (Priority 2)
   - **Purpose:** Asynchronous trade explainability and audit logging
   - **Technology:** Python Kafka consumer with TimescaleDB storage
   - **Responsibilities:**
     - Subscribe to trading decision events from Kafka
     - Generate model explanations (SHAP values, attention weights)
     - Store complete audit trail for regulatory compliance (7-year retention)
     - Serve forensics queries via API for post-hoc analysis
   - **Key Innovation:** Decouples forensics from trading path, eliminates 200-500ms latency overhead

7. **Regime Detector** (Priority 2 - Enhanced)
   - **Purpose:** Probabilistic regime classification vs. discrete labels
   - **Technology:** Dirichlet Process Mixture Model or Hidden Markov Model
   - **Responsibilities:**
     - Output continuous probability distribution [P(bull), P(bear), P(sideways)]
     - Enable weighted ensemble decisions vs. hard switches
     - Quantify regime uncertainty for meta-controller
   - **Key Innovation:** Smooth transitions eliminate portfolio discontinuities during ambiguous market states

8. **Online Learning Pipeline** (Priority 3)
   - **Purpose:** Incremental model updates with stability constraints
   - **Technology:** Python pipeline with elastic weight consolidation (EWC)
   - **Responsibilities:**
     - Maintain sliding window of recent market data (30-90 days)
     - Perform incremental updates with conservative learning rates (1e-5)
     - Monitor performance for instability (automatic rollback trigger)
     - Preserve important weights via EWC to prevent catastrophic forgetting
   - **Key Innovation:** Maintains <5% performance degradation vs. 15-25% decay with static models

### Data Flow

**Training Flow:**
```
Market Data → Data Service (Preprocess) → Training Orchestrator
                                               ↓
                                        GPU Training Jobs
                                               ↓
                                        TimescaleDB Metrics
                                               ↓
                                        MLflow Model Registry
```

**Inference Flow:**
```
Real-time Market Data → Data Service (Cache/Transform) → Inference Service
                                                              ↓
                                                        Regime Detector
                                                              ↓
                                                        Meta-Controller
                                                              ↓
                                                        Risk Manager
                                                              ↓
            ┌─────────────────────────────────────────────┬─┴──────┐
            ▼                                             ▼        ▼
    Execution Engine                              Kafka Events  Forensics
```

**Online Learning Flow:**
```
Recent Market Data → Data Service → Online Learning Pipeline
                                           ↓
                                    Incremental Update
                                           ↓
                                    Stability Check (Sharpe)
                                           ↓
                              [Pass] → MLflow Registry Update
                                           ↓
                              [Fail] → Rollback to Previous Checkpoint
```

---

## Data Models

### Schema Design

**TimescaleDB Schema (Experiment Tracking):**
```sql
-- Experiments table (replaces SQLite)
CREATE TABLE experiments (
    id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL, -- 'bull_specialist', 'bear_specialist', etc.
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'running'
);

-- Hypertable for time-series metrics (automatic partitioning)
CREATE TABLE experiment_metrics (
    time TIMESTAMPTZ NOT NULL,
    experiment_id INTEGER REFERENCES experiments(id),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    step INTEGER,
    epoch INTEGER
);

SELECT create_hypertable('experiment_metrics', 'time');
CREATE INDEX idx_exp_metrics_name ON experiment_metrics(experiment_id, metric_name, time DESC);

-- Model checkpoints tracking
CREATE TABLE model_checkpoints (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiments(id),
    checkpoint_path VARCHAR(500) NOT NULL,
    sharpe_ratio DOUBLE PRECISION,
    validation_loss DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_production BOOLEAN DEFAULT FALSE,
    file_size_mb DOUBLE PRECISION
);

-- Regime detection history
CREATE TABLE regime_history (
    time TIMESTAMPTZ NOT NULL,
    prob_bull DOUBLE PRECISION CHECK (prob_bull >= 0 AND prob_bull <= 1),
    prob_bear DOUBLE PRECISION CHECK (prob_bear >= 0 AND prob_bear <= 1),
    prob_sideways DOUBLE PRECISION CHECK (prob_sideways >= 0 AND prob_sideways <= 1),
    entropy DOUBLE PRECISION, -- uncertainty measure
    CHECK (prob_bull + prob_bear + prob_sideways = 1.0)
);

SELECT create_hypertable('regime_history', 'time');
```

**Kafka Event Schemas:**
```json
// Trading Decision Event (for forensics)
{
  "event_type": "trading_decision",
  "timestamp": "2025-10-21T14:30:00Z",
  "decision_id": "uuid-v4",
  "symbol": "AAPL",
  "action": "BUY", // BUY, SELL, HOLD
  "quantity": 100,
  "confidence": 0.85,
  "regime_probs": {
    "bull": 0.65,
    "bear": 0.15,
    "sideways": 0.20
  },
  "strategy_weights": {
    "bull_specialist": 0.60,
    "bear_specialist": 0.10,
    "sideways_specialist": 0.30
  },
  "features": {
    "rsi": 45.3,
    "macd": 0.012,
    "volume_ratio": 1.23,
    // ... additional features
  },
  "risk_checks": {
    "position_limit_ok": true,
    "concentration_ok": true,
    "correlation_ok": true
  }
}

// Model Update Event (for online learning)
{
  "event_type": "model_update",
  "timestamp": "2025-10-21T15:00:00Z",
  "model_id": "bull_specialist_v127",
  "update_type": "incremental", // incremental, full_retrain
  "data_window": {
    "start": "2025-10-14T15:00:00Z",
    "end": "2025-10-21T15:00:00Z"
  },
  "performance_metrics": {
    "sharpe_pre": 1.85,
    "sharpe_post": 1.82,
    "degradation_pct": 1.6
  },
  "stability_check": "passed" // passed, failed_rollback
}
```

**Redis Cache Schema (Feature Store):**
```python
# Key pattern: feature:{symbol}:{timeframe}:{version}
# Value: JSON-serialized feature vector
{
  "symbol": "AAPL",
  "timeframe": "1min",
  "timestamp": "2025-10-21T14:30:00Z",
  "features": {
    "price": 175.23,
    "returns_1d": 0.0123,
    "returns_5d": 0.0456,
    "volatility_20d": 0.25,
    "rsi_14": 45.3,
    "macd": 0.012,
    "macd_signal": 0.008,
    "volume_ratio": 1.23,
    "ema_12": 174.56,
    "ema_26": 173.89,
    // ... 50+ additional features
  },
  "version": "v2.1",
  "ttl": 300 // 5 minute expiration
}

# Meta-data for cache invalidation
feature_pipeline_version: "v2.1"
last_market_data_update: "2025-10-21T14:30:00Z"
```

**Risk State Schema (In-Memory):**
```python
{
  "portfolio": {
    "total_value": 1000000.0,
    "cash": 250000.0,
    "positions": {
      "AAPL": {
        "quantity": 1000,
        "avg_cost": 170.0,
        "current_price": 175.23,
        "market_value": 175230.0,
        "pct_portfolio": 0.175, // 17.5% - within 25% limit
        "unrealized_pnl": 5230.0
      },
      "GOOGL": {
        "quantity": 500,
        "avg_cost": 140.0,
        "current_price": 145.67,
        "market_value": 72835.0,
        "pct_portfolio": 0.073,
        "unrealized_pnl": 2835.0
      }
      // ... additional positions
    }
  },
  "risk_metrics": {
    "var_95_1d": 25000.0, // Value at Risk
    "portfolio_beta": 1.15,
    "correlation_matrix": [[1.0, 0.65], [0.65, 1.0]],
    "sharpe_ratio_7d": 1.85,
    "max_drawdown_30d": 0.08
  },
  "limit_utilization": {
    "max_position_size_pct": 0.175, // current max, limit 0.25
    "max_sector_exposure_pct": 0.45, // tech sector, limit 0.50
    "leverage_ratio": 1.2, // limit 1.5
    "daily_loss_limit_pct": 0.015, // current 1.5%, limit 2%
  },
  "last_updated": "2025-10-21T14:30:00Z"
}
```

### Migration Strategy

**Phase 1: Parallel Write (Week 1-2)**
- Implement dual-write to both SQLite and TimescaleDB
- Verify data consistency with automated reconciliation checks
- Monitor TimescaleDB write latency (target <10ms P95)
- Keep SQLite as fallback with automatic failover

**Phase 2: Read Migration (Week 3)**
- Switch read operations to TimescaleDB
- Maintain SQLite writes for safety
- Compare query results between databases
- Validate hypertable partitioning strategy

**Phase 3: Write Cutover (Week 4)**
- Stop SQLite writes after 7-day stability period
- Archive SQLite database for historical reference
- Enable TimescaleDB compression policies (7-day window)
- Validate 20+ concurrent write capacity

**Data Integrity Validation:**
- Checksum comparison of last 10k experiment records
- Metric value tolerance check (max 0.01% variance due to floating point)
- Foreign key constraint validation (all experiment_metrics reference valid experiments)
- Retention policy verification (7-year compliance requirement)

---

## API Specification

### /api/v1/features/{symbol}/{timeframe}
**Method:** GET
**Purpose:** Retrieve preprocessed feature vector for inference
**Request:**
- Path Parameters:
  - `symbol`: Stock ticker (e.g., "AAPL")
  - `timeframe`: Time resolution ("1min", "5min", "1hour")
- Query Parameters:
  - `timestamp`: ISO 8601 timestamp (optional, defaults to latest)
  - `version`: Feature pipeline version (optional, defaults to latest)

**Response (200 OK):**
```json
{
  "symbol": "AAPL",
  "timeframe": "1min",
  "timestamp": "2025-10-21T14:30:00Z",
  "features": {
    "price": 175.23,
    "returns_1d": 0.0123,
    "volatility_20d": 0.25,
    "rsi_14": 45.3,
    "macd": 0.012,
    // ... 50+ features
  },
  "metadata": {
    "cache_hit": true,
    "pipeline_version": "v2.1",
    "generated_at": "2025-10-21T14:30:00Z"
  }
}
```

**Error Responses:**
- `404 Not Found`: Symbol not supported or no data available
- `422 Unprocessable Entity`: Invalid timeframe or timestamp format
- `503 Service Unavailable`: Redis cache unavailable, data pipeline overloaded

**Performance:** P95 <20ms (cache hit), P95 <200ms (cache miss with recompute)

**Rate Limits:** 1000 requests/minute per client

**Example cURL:**
```bash
curl -X GET "https://trading-api/v1/features/AAPL/1min?timestamp=2025-10-21T14:30:00Z" \
  -H "Authorization: Bearer $API_TOKEN"
```

---

### /api/v1/predict
**Method:** POST
**Purpose:** Request trading signal from ensemble model with regime-aware strategy blending

**Request:**
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-10-21T14:30:00Z", // optional, defaults to now
  "strategy_override": null, // optional: force specific strategy
  "risk_check": true, // optional: include risk validation
  "explain": false // optional: include model explanations (adds latency)
}
```

**Response (200 OK):**
```json
{
  "decision_id": "550e8400-e29b-41d4-a716-446655440000",
  "symbol": "AAPL",
  "action": "BUY", // BUY, SELL, HOLD
  "confidence": 0.85,
  "recommended_quantity": 100,
  "regime_probabilities": {
    "bull": 0.65,
    "bear": 0.15,
    "sideways": 0.20,
    "entropy": 0.82 // uncertainty measure
  },
  "strategy_weights": {
    "bull_specialist": 0.60,
    "bear_specialist": 0.10,
    "sideways_specialist": 0.30
  },
  "risk_validation": {
    "approved": true,
    "warnings": [],
    "checks": {
      "position_limit": "pass",
      "concentration": "pass",
      "daily_loss_limit": "pass"
    }
  },
  "metadata": {
    "model_version": "bull_specialist_v127",
    "latency_ms": 45,
    "timestamp": "2025-10-21T14:30:00.123Z"
  }
}
```

**Error Responses:**
- `400 Bad Request`: Invalid symbol or parameters
- `403 Forbidden`: Risk check failed, trade not permitted
- `503 Service Unavailable`: Model serving unavailable

**Performance:** P95 <50ms (without explanations), P95 <150ms (with explanations)

**Rate Limits:** 500 requests/minute per client

**Example cURL:**
```bash
curl -X POST "https://trading-api/v1/predict" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "risk_check": true}'
```

---

### /api/v1/risk/check
**Method:** POST
**Purpose:** Validate proposed trade against portfolio risk constraints

**Request:**
```json
{
  "symbol": "AAPL",
  "action": "BUY",
  "quantity": 100,
  "estimated_price": 175.23,
  "portfolio_state": { // optional: override current portfolio
    "total_value": 1000000.0,
    "positions": { /* ... */ }
  }
}
```

**Response (200 OK - Approved):**
```json
{
  "approved": true,
  "risk_assessment": {
    "position_after_trade": {
      "quantity": 1100,
      "market_value": 192753.0,
      "pct_portfolio": 0.193 // 19.3% < 25% limit
    },
    "portfolio_impact": {
      "concentration_increase": 0.018,
      "correlation_change": 0.02,
      "var_increase": 1250.0
    },
    "limit_utilization": {
      "position_size": 0.77, // 77% of 25% limit
      "sector_exposure": 0.85, // 85% of 50% tech limit
      "leverage": 0.80 // 80% of 1.5x limit
    }
  },
  "warnings": [],
  "timestamp": "2025-10-21T14:30:00Z"
}
```

**Response (200 OK - Rejected):**
```json
{
  "approved": false,
  "rejection_reasons": [
    {
      "code": "CONCENTRATION_LIMIT",
      "message": "Trade would exceed 25% single-position limit (proposed: 27.3%)",
      "limit": 0.25,
      "proposed": 0.273
    }
  ],
  "allowed_quantity": 50, // maximum allowable under constraints
  "timestamp": "2025-10-21T14:30:00Z"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid trade parameters
- `503 Service Unavailable`: Risk manager unavailable

**Performance:** P95 <10ms

**Rate Limits:** 2000 requests/minute per client (higher than trading frequency)

**Example cURL:**
```bash
curl -X POST "https://trading-api/v1/risk/check" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "action": "BUY", "quantity": 100, "estimated_price": 175.23}'
```

---

### /api/v1/forensics/{decision_id}
**Method:** GET
**Purpose:** Retrieve detailed explanation for past trading decision

**Request:**
- Path Parameters:
  - `decision_id`: UUID from prediction response

**Response (200 OK):**
```json
{
  "decision_id": "550e8400-e29b-41d4-a716-446655440000",
  "symbol": "AAPL",
  "timestamp": "2025-10-21T14:30:00Z",
  "action": "BUY",
  "explanations": {
    "shap_values": {
      "rsi_14": 0.15, // positive contribution
      "macd": 0.12,
      "volatility_20d": -0.05, // negative contribution
      // ... top 10 features by importance
    },
    "regime_reasoning": "Bull regime probability (65%) driven by sustained uptrend over 5 days with increasing volume",
    "strategy_selection": "Meta-controller weighted bull specialist at 60% based on recent 7-day Sharpe ratio (2.1) outperforming other strategies"
  },
  "execution_details": {
    "executed": true,
    "execution_price": 175.25,
    "slippage": 0.02,
    "pnl_realized": null // not closed yet
  },
  "audit_trail": {
    "risk_checks_passed": ["position_limit", "concentration", "correlation"],
    "approvals": ["risk_manager_auto"],
    "processing_time_ms": 45
  }
}
```

**Error Responses:**
- `404 Not Found`: Decision ID not found or expired (>90 days)
- `503 Service Unavailable`: Forensics service unavailable

**Performance:** P95 <500ms (async processing, not latency-critical)

**Rate Limits:** 100 requests/minute per client

---

### /api/v1/models/online-update
**Method:** POST
**Purpose:** Trigger incremental model update with recent data (internal API)

**Request:**
```json
{
  "model_id": "bull_specialist",
  "data_window": {
    "start": "2025-10-14T00:00:00Z",
    "end": "2025-10-21T00:00:00Z"
  },
  "learning_rate": 0.00001,
  "ewc_lambda": 1000.0, // elastic weight consolidation strength
  "stability_check": true
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "update-job-12345",
  "status": "queued",
  "estimated_duration_minutes": 30,
  "status_url": "/api/v1/jobs/update-job-12345"
}
```

**Performance:** Async operation, 15-45 minute completion

**Authentication:** Internal service token only (not exposed to external clients)


---

## Integration Points

*No external integrations*

---

## Scalability Analysis

### Load Projections

**Current State (Baseline):**
- Market data updates: ~500/sec across 50 symbols
- Training frequency: 3 models per week (weekly retrain cycle)
- Inference requests: ~10/sec (single strategy, synchronous)
- Concurrent experiments: 2-3 (limited by SQLite write locks)
- Daily data volume: ~5GB checkpoint growth + 100MB metrics

**6-Month Projection:**
- Market data updates: ~2000/sec (100 symbols + multiple timeframes)
- Training frequency: 5 models daily (continuous training rotation)
- Inference requests: ~50/sec (5 strategies with parallel evaluation)
- Concurrent experiments: 10-15 (hyperparameter search expansion)
- Daily data volume: <500MB controlled growth (automated cleanup)

**12-Month Projection:**
- Market data updates: ~5000/sec (200 symbols + tick data)
- Training frequency: 10+ models continuously (online learning always active)
- Inference requests: ~200/sec (strategy portfolio expansion)
- Concurrent experiments: 20+ (distributed hyperparameter optimization)
- Daily data volume: <500MB sustained (mature cleanup policies)

**Growth Drivers:**
- Symbol coverage expansion (50 → 200 equities)
- Strategy diversification (3 regime specialists → 10+ specialized models)
- Feature engineering depth (30 features → 100+ features)
- Research velocity (weekly experiments → daily continuous improvement)

### Bottleneck Analysis

**Current Bottlenecks (Must Fix):**

1. **SQLite Write Contention** (CRITICAL)
   - **Symptom:** Training processes blocked waiting for write lock (>500ms waits)
   - **Root Cause:** File-based database with single-writer limitation
   - **Impact:** Limits concurrent experiments to 2-3, delays hyperparameter search by 5-10x
   - **Capacity:** ~3 writes/sec sustained before severe degradation
   - **Solution:** TimescaleDB with 20+ concurrent writers, <10ms P95 write latency
   - **Expected Improvement:** 10x concurrent experiment capacity

2. **Synchronous Forensics** (CRITICAL)
   - **Symptom:** Trading decisions delayed 200-500ms for explainability computation
   - **Root Cause:** SHAP value calculation in critical path
   - **Impact:** Missed alpha opportunities, reduced strategy profitability by ~2-5%
   - **Capacity:** ~2-5 decisions/sec before latency SLA violation
   - **Solution:** Event-driven architecture with Kafka async processing
   - **Expected Improvement:** 4-10x latency reduction (<50ms trading path)

3. **Redundant Data Loading** (HIGH)
   - **Symptom:** Each training script loads full dataset (40% of training time = I/O)
   - **Root Cause:** No shared data pipeline or caching layer
   - **Impact:** 3x slower training cycles, wasted compute resources
   - **Capacity:** Disk I/O saturated during multi-process training
   - **Solution:** Unified data service with Redis cache (90%+ hit rate)
   - **Expected Improvement:** 3x faster training, 10x I/O reduction

**Projected Bottlenecks (Plan Ahead):**

4. **Single-Node GPU Memory** (12-month horizon)
   - **When:** 10+ large models in ensemble (>2GB per model)
   - **Mitigation:** Model distillation, quantization, distributed inference
   - **Alternative:** Multi-GPU server or inference service horizontal scaling

5. **Feature Engineering Compute** (12-month horizon)
   - **When:** 5000+ data updates/sec with 100+ features per symbol
   - **Mitigation:** Incremental feature computation, feature store pre-aggregation
   - **Alternative:** Dedicated feature computation cluster (Spark/Flink)

6. **Kafka Topic Retention** (12-month horizon)
   - **When:** 200+ decisions/sec = 17M events/day = 500GB/month forensics data
   - **Mitigation:** Compression, shorter retention (7 days hot, rest cold storage)
   - **Alternative:** Tiered storage with S3/Glacier archival

**Scaling Strategy Matrix:**

| Load Range | Data Updates/sec | Inference/sec | Experiments | Scaling Action |
|-----------|-----------------|---------------|-------------|----------------|
| Current | 500 | 10 | 2-3 | Baseline (SQLite, no cache) |
| Phase 1 (0-3mo) | 500-1000 | 10-25 | 5-10 | TimescaleDB + Redis cache |
| Phase 2 (3-6mo) | 1000-2000 | 25-50 | 10-15 | Inference service scaling (2 replicas) |
| Phase 3 (6-12mo) | 2000-5000 | 50-200 | 15-20 | Data service sharding + GPU scaling |
| Future (12mo+) | >5000 | >200 | >20 | Distributed training + inference clusters |

**Cost Scaling:**
- Phase 1: +$2k/month infrastructure (TimescaleDB cluster, Kafka, Redis)
- Phase 2: +$3k/month (additional GPU server, inference replicas)
- Phase 3: +$5k/month (distributed system overhead, multi-region)

---

## Integration Points

**Internal Dependencies:**
- **Existing Model Checkpoints:** Backward compatible loading from `.pth`/`.h5` files
- **Market Data Feeds:** WebSocket connections maintained (no changes to ingestion)
- **Execution Engine:** REST API contract preserved (new risk checks added)
- **Backtesting Framework:** Data pipeline interface compatible with historical data replay

**External Services:**
- **MLflow Tracking Server:** Hosted on-premise (https://mlflow.internal)
- **TimescaleDB Cluster:** 3-node cluster with automatic failover
- **Kafka Cluster:** 3 brokers with replication factor 2
- **Redis Cluster:** 2-node primary-replica setup
- **Monitoring Stack:** Prometheus + Grafana + AlertManager

**Data Flow Contracts:**
```python
# Data Service Interface (backward compatible)
class IDataService:
    def get_features(symbol: str, timestamp: datetime) -> FeatureVector:
        """Returns preprocessed features. Replaces old data_loader.load_data()"""
        pass
    
    def get_historical_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Bulk historical data for training. Maintains old interface."""
        pass

# Model Registry Interface (new)
class IModelRegistry:
    def load_model(model_id: str, version: str = "latest") -> Model:
        """Load model from MLflow registry"""
        pass
    
    def save_checkpoint(model: Model, metrics: Dict) -> str:
        """Save checkpoint with metrics to registry"""
        pass
```

---

## Scalability Analysis

### Key Metrics
- **data_pipeline_cache_hit_rate:** Current: N/A, Target: >90%, P95: >85%
  - Monitor: Datadog gauge `data_service.cache.hit_rate`
  - Alert: <80% for 15 minutes (indicates cache thrashing or cold start)
  
- **training_job_success_rate:** Current: ~85%, Target: >95%, P95: >93%
  - Monitor: TimescaleDB query `SELECT COUNT(*)/total FROM experiments WHERE status='success'`
  - Alert: <90% success rate over 24 hours (indicates training instability)
  
- **inference_latency_p95:** Current: N/A, Target: <50ms, Max: <100ms
  - Monitor: Datadog histogram `inference_service.predict.latency`
  - Alert: >100ms for 5 consecutive minutes (SLA violation)
  
- **regime_probability_entropy:** Current: N/A, Target: 0.5-1.0 (healthy), Alert: >1.5
  - Monitor: Custom metric from regime detector
  - Alert: >1.5 for 1 hour (extreme uncertainty, possibly data quality issue)
  
- **portfolio_concentration_risk:** Current: Untracked, Target: <25% per asset, Max: 23% alert threshold
  - Monitor: Risk manager real-time metric
  - Alert: Any position >23% (approaching 25% hard limit)
  
- **model_performance_sharpe:** Current: Manual calculation, Target: >1.5, Degradation Alert: -20% WoW
  - Monitor: Daily calculation stored in TimescaleDB
  - Alert: Week-over-week drop >20% (triggers investigation + potential model update)
  
- **system_memory_growth:** Current: Uncontrolled, Target: <100MB/hour, Alert: >500MB/hour
  - Monitor: Node exporter memory metrics
  - Alert: >500MB/hour growth sustained for 4 hours (memory leak)

### Alert Conditions

**Critical Alerts (Page On-Call):**
- Trading decision latency >200ms for 5+ consecutive minutes
- Risk limit violation detected but not blocked by risk manager (system integrity issue)
- Model serving service down for >2 minutes during market hours
- Data pipeline failure preventing feature generation for >5 minutes
- TimescaleDB cluster primary node down (automatic failover should occur)

**Warning Alerts (Slack Channel):**
- Model retraining failed 2 consecutive times (investigation needed)
- Forensics event queue backlog >50k events (processing lag)
- Cache hit rate <80% sustained for 30 minutes (performance degradation)
- Disk usage >80% on any node (cleanup or capacity planning)
- Kafka consumer lag >10k messages (forensics falling behind)
- Online learning update rejected (Sharpe ratio stability check failed)

**Informational Alerts (Dashboard Only):**
- New model checkpoint created and promoted to production
- Daily portfolio performance summary
- Weekly resource utilization report
- Monthly cost tracking and optimization suggestions

---

## Security Considerations

**Authentication & Authorization:**
- Internal service mesh with mTLS (service-to-service authentication)
- API Gateway with JWT tokens for external clients (execution engine)
- Role-based access control: `trader`, `researcher`, `admin` roles
- API keys rotated every 90 days, stored in HashiCorp Vault

**Data Security:**
- Encryption at rest for TimescaleDB (LUKS full-disk encryption)
- Encryption in transit (TLS 1.3 for all HTTP, mTLS for Kafka)
- PII handling: No personal data, only market data and trading signals
- Model weights encrypted in MLflow artifact store (AES-256)

**Network Security:**
- Internal services isolated in private VPC subnet
- Firewall rules: only API Gateway exposed, all services internal-only
- Intrusion detection system (IDS) monitoring unusual traffic patterns
- Rate limiting per client IP (prevents DoS attacks)

**Audit & Compliance:**
- All trading decisions logged to forensics (7-year retention requirement)
- Experiment tracking with user attribution (who trained which model)
- Model versioning with reproducibility (config + data hash stored)
- Regular security audits (quarterly penetration testing)

**Secrets Management:**
- Database credentials stored in Vault, rotated monthly
- API tokens generated with limited scope and expiration
- Environment-specific secrets (dev/staging/prod) strictly separated
- No secrets in code or config files (Vault injection at runtime)

---

## Technical Debt & Trade-offs

### Architectural Trade-offs

**Decision: Microservices vs Monolith**
- **Chosen:** Microservices (Data Service, Inference, Risk Manager, Forensics as separate services)
- **Trade-off:** 
  - ✅ **Pros:** Independent scaling, failure isolation, technology flexibility, team autonomy
  - ❌ **Cons:** Operational complexity, network latency between services, distributed debugging challenges
- **Justification:** Benefits outweigh costs for production trading system. Need independent scaling of training (GPU-bound) vs inference (latency-sensitive) vs forensics (throughput-bound).
- **Mitigation:** Service mesh (Istio/Linkerd) for observability, circuit breakers for failure handling

**Decision: TimescaleDB vs PostgreSQL vs InfluxDB**
- **Chosen:** TimescaleDB (PostgreSQL extension with time-series optimizations)
- **Trade-off:**
  - ✅ **Pros:** SQL familiarity, ACID guarantees, automatic partitioning, concurrent writes
  - ❌ **Cons:** More complex than InfluxDB, not purpose-built for time-series at scale
- **Justification:** Team SQL expertise, need for complex queries (joins across experiments + metrics), ACID for experiment tracking
- **Alternative Considered:** InfluxDB (rejected: weak join support, less mature ecosystem)

**Decision: Kafka vs RabbitMQ vs Redis Streams**
- **Chosen:** Kafka for forensics event streaming
- **Trade-off:**
  - ✅ **Pros:** High throughput (100k+ events/sec), persistent logs, replay capability, scalability
  - ❌ **Cons:** Operational complexity (Zookeeper/KRaft), higher latency than Redis, storage overhead
- **Justification:** Need for event replay (forensics auditing), high throughput growth trajectory, event log persistence
- **Alternative Considered:** Redis Streams (rejected: limited retention, not built for large-scale event sourcing)

**Decision: Probabilistic Regime Detection vs Discrete Classification**
- **Chosen:** Probabilistic (Dirichlet process or HMM with continuous emissions)
- **Trade-off:**
  - ✅ **Pros:** Smooth transitions, uncertainty quantification, meta-controller flexibility
  - ❌ **Cons:** More complex inference, hyperparameter tuning difficulty, harder to interpret
- **Justification:** Eliminates 15% portfolio disruption during transitions, provides richer signal to meta-controller
- **Mitigation:** Extensive backtesting on historical regime transitions, entropy monitoring for pathological cases

### Technical Debt Registry

**Existing Debt (Must Address):**

1. **Shell Script Orchestration** (CRITICAL)
   - **Issue:** `monitor_training.sh`, `start_bull_training.sh` without process supervision
   - **Impact:** No automatic restart on failure, manual intervention required, poor observability
   - **Plan:** Replace with systemd services + Celery task orchestration in Phase 1
   - **Effort:** 3 days (service definitions + monitoring integration)

2. **Manual Checkpoint Management** (HIGH)
   - **Issue:** No automated cleanup, checkpoints accumulating (5GB/day)
   - **Impact:** Disk exhaustion risk, manual pruning required weekly
   - **Plan:** Automated retention policy in Phase 2 (keep best 10 per experiment + last 30 days)
   - **Effort:** 2 days (cleanup script + MLflow integration)

3. **Inconsistent Feature Engineering** (HIGH)
   - **Issue:** Each training script implements own feature pipeline (risk of data leakage)
   - **Impact:** Potential lookahead bias, inconsistent train/inference features, debugging nightmare
   - **Plan:** Unified data service with single feature pipeline in Phase 1
   - **Effort:** 5 days (abstraction layer + validation testing)

**Introduced Debt (Acceptable Trade-offs):**

4. **Online Learning Conservative Approach** (MEDIUM - Intentional)
   - **Issue:** Very conservative learning rates (1e-5) and strong EWC regularization
   - **Impact:** Slow adaptation to regime shifts, may underperform full retraining
   - **Justification:** Stability prioritized over adaptation speed in production
   - **Future Work:** Adaptive learning rate scheduling based on regime stability
   - **Revisit:** After 6 months of production stability data

5. **Single-Asset Position Sizing** (MEDIUM - Scoped Out)
   - **Issue:** No multi-asset portfolio optimization, treats positions independently
   - **Impact:** Suboptimal risk-adjusted returns, missing correlation benefits
   - **Justification:** Complexity deferral, single-asset sufficient for Phase 1 validation
   - **Future Work:** Hierarchical risk parity with correlation matrix (Phase 5)
   - **Revisit:** After 6 months of single-asset production experience

6. **Synchronous Risk Checks** (LOW - Acceptable Latency)
   - **Issue:** Risk manager called synchronously, adds ~10ms to inference path
   - **Impact:** Small latency increase, but necessary for safety
   - **Justification:** Risk checks must block trades, async would allow risky trades through
   - **Future Work:** Pre-compute risk boundaries for faster validation
   - **Revisit:** Only if latency becomes critical bottleneck (>50ms P95)

### Technical Debt Paydown Schedule

- **Month 1:** Shell script → systemd (#1), unified feature pipeline (#3)
- **Month 2:** Automated checkpoint cleanup (#2)
- **Month 3-6:** Production stabilization, no new debt allowed
- **Month 6:** Review introduced debt (#4, #5, #6) with production data
- **Month 9:** Multi-asset optimization (#5) if validated by business case
- **Month 12:** Adaptive online learning (#4) if stability proven

### Monitoring Technical Debt

- **Code Coverage:** Target 85% for new code, 70% for refactored legacy (measured in CI/CD)
- **Dependency Freshness:** Update critical dependencies monthly (security), quarterly (features)
- **Documentation Debt:** All APIs documented with OpenAPI, architecture diagrams updated quarterly
- **Test Debt:** No skipped tests in production, flaky tests fixed within 1 sprint

---

## Failure Modes & Recovery

### Scenario 1: TimescaleDB Connection Failure During Training
**Likelihood:** Medium (database restarts, network issues)
**Impact:** Medium (training continues but metrics not logged)

**Detection:**
- Connection timeout (>5 seconds)
- Write error response from TimescaleDB
- Health check failures (monitored every 30 seconds)
- Alert: "TimescaleDB connection lost" (Slack notification)

**Mitigation:**
- Automatic fallback to local file-based logging (`/var/log/training/experiments/`)
- Training process continues uninterrupted (no training data lost)
- Buffer metrics in memory (max 10,000 records before disk write)
- Retry connection every 60 seconds with exponential backoff

**Recovery:**
1. Wait for automatic database failover (30-60 seconds with HA cluster)
2. When connection restored, sync buffered metrics from local files
3. Validate metric count matches expected (based on training steps)
4. Resume normal logging to TimescaleDB
5. Alert resolution: "TimescaleDB connection restored"

**Prevention:**
- TimescaleDB 3-node HA cluster with automatic failover
- Connection pooling with keepalive (prevents stale connections)
- Health check probes before every write batch
- Separate read-replica for query workloads (reduces primary load)

**Runbook:** `docs/runbooks/timescaledb-failure.md`

---

### Scenario 2: Kafka Broker Unavailable for Forensics Events
**Likelihood:** Low (Kafka is highly available, but outages possible)
**Impact:** Low (trading decisions unaffected, forensics delayed)

**Detection:**
- Producer send timeout (>10 seconds)
- Kafka broker health check failure
- Consumer lag increasing (indicates broker issues)
- Alert: "Kafka broker unreachable" (Slack notification)

**Mitigation:**
- Trading decisions proceed immediately (forensics not in critical path)
- Buffer events in memory (max 10,000 events ~5MB)
- If buffer full, write to disk overflow (`/var/log/forensics/overflow/`)
- Display warning in trading UI: "Forensics processing delayed"

**Recovery:**
1. Kafka cluster automatic leader election (10-30 seconds)
2. Producer reconnects automatically (built-in retry logic)
3. Flush buffered events to Kafka in chronological order
4. Verify event count matches expected (no data loss)
5. Forensics consumer catches up (typical lag <5 minutes for 10k events)
6. Alert resolution: "Kafka cluster healthy, forensics processing resumed"

**Prevention:**
- Kafka 3-broker cluster with replication factor 2
- Zookeeper/KRaft for automatic leader election
- Producer configuration: `acks=1` (balance between latency and reliability)
- Separate Kafka cluster for critical trading signals vs forensics

**Runbook:** `docs/runbooks/kafka-outage.md`

---

### Scenario 3: Meta-Controller Selects Invalid Strategy Mix
**Likelihood:** Low (should be caught by softmax output layer)
**Impact:** Medium (could cause incorrect position sizing if undetected)

**Detection:**
- Strategy weight validation check: `sum(weights) != 1.0` (tolerance 0.001)
- Negative weights detected (invalid probability)
- Any weight >0.99 (suspicious, likely single-strategy collapse)
- Alert: "Meta-controller output validation failed" (page on-call)

**Mitigation:**
- Automatic weight normalization: `weights = weights / sum(weights)`
- Log anomaly with full model state for post-mortem analysis
- If normalization impossible (all zeros/NaNs): fall back to equal weighting [0.33, 0.33, 0.34]
- Trading decision continues with corrected weights (no user-visible disruption)
- Increment error counter metric for monitoring

**Recovery:**
1. Immediate: use normalized/fallback weights for current decision
2. Short-term (10 minutes): analyze last 100 meta-controller outputs
3. If >10% validation failures: trigger meta-controller model rollback to previous checkpoint
4. Medium-term (1 hour): investigate root cause (training instability, input distribution shift)
5. If persistent: disable meta-controller, use regime-based routing until fixed

**Prevention:**
- Softmax output layer enforces valid probability distribution at network level
- Gradient clipping during meta-controller training (prevents exploding gradients)
- Validation during training: reject checkpoints with >5% invalid outputs on test set
- Regular synthetic stress testing with edge-case inputs

**Runbook:** `docs/runbooks/meta-controller-validation-failure.md`

---

### Scenario 4: Online Learning Introduces Model Instability
**Likelihood:** Medium (online learning inherently risky)
**Impact:** High (degraded trading performance, potential losses)

**Detection:**
- Sharpe ratio drops >30% over 5-day rolling window (compared to baseline)
- P95 inference latency increases >2x (model complexity grew)
- Prediction variance increases >50% (overconfident or erratic predictions)
- Alert: "Model stability check failed - rollback initiated" (page on-call)

**Mitigation:**
- Automatic rollback to previous stable checkpoint within 60 seconds
- Halt all online learning updates until manual investigation
- Switch to "frozen model" mode (no further updates)
- Maintain trading using last known-good model version
- Generate stability report with performance comparison charts

**Recovery:**
1. Immediate (1 minute): rollback to checkpoint with tag `stable` and last 7-day Sharpe >1.5
2. Short-term (30 minutes): analyze update that caused instability
   - Compare training data distribution (detect regime shift)
   - Check learning rate and EWC lambda values
   - Review gradient norms and loss curves
3. Medium-term (24 hours): adjust online learning hyperparameters
   - Reduce learning rate by 50% (e.g., 1e-5 → 5e-6)
   - Increase EWC lambda by 2x (stronger regularization)
   - Reduce update frequency (daily → weekly)
4. Long-term (1 week): gradual re-enable with canary testing
   - Enable online learning for 10% of decisions (shadow mode)
   - Monitor stability for 7 days before full re-enable
   - Implement additional stability checks (entropy-based early stopping)

**Prevention:**
- Conservative learning rates (1e-5) by default
- Strong EWC regularization (lambda=1000) to prevent catastrophic forgetting
- Stability check after every update (automated Sharpe ratio comparison)
- Limit update frequency (max 1x per day) to observe performance over full trading day
- Maintain 30-day sliding window of stable checkpoints for fast rollback

**Runbook:** `docs/runbooks/online-learning-stability.md`

---

### Scenario 5: Data Pipeline Feature Leakage (Lookahead Bias)
**Likelihood:** Low (but catastrophic if occurs)
**Impact:** Critical (invalidates all backtesting, live trading losses)

**Detection:**
- Backtesting Sharpe >3.0 (suspiciously high, indicates leakage)
- Live trading performance 50%+ worse than backtest (reality check)
- Feature correlation analysis shows future data in features
- Manual code review during quarterly audits

**Mitigation:**
- Immediate halt of all trading using suspected features
- Rollback to previous feature pipeline version (before suspected leakage)
- Comprehensive audit of feature engineering code (line-by-line review)
- Re-run backtests with corrected features to measure actual performance

**Recovery:**
1. Immediate (1 hour): disable affected features, use feature subset known to be safe
2. Short-term (1 day): identify source of leakage in code
   - Common culprits: using `df.shift(0)` instead of `df.shift(1)`, sorting by future timestamp
   - Validate timestamp alignment in all feature computations
3. Medium-term (1 week): fix leakage, add validation tests
   - Implement automated lookahead detection in CI/CD
   - Add unit tests that verify feature values match manual calculations
4. Long-term (1 month): re-train all models with corrected features
   - Compare new performance to old (expect degradation if leakage existed)
   - Update expected Sharpe ratios and risk parameters
   - Communicate impact to stakeholders with transparency

**Prevention:**
- Comprehensive unit tests for every feature (compare to manual calculation)
- Automated lookahead detection: check that feature at time T only uses data from T-1 and earlier
- Code review requirement for all feature pipeline changes (2 reviewers minimum)
- Quarterly audit by independent reviewer (not primary developer)
- Shadow mode validation: run new features in parallel with known-good features for 30 days

**Runbook:** `docs/runbooks/feature-leakage-response.md`

---

### Scenario 6: Redis Cache Corruption/Eviction Storm
**Likelihood:** Medium (memory pressure, misconfiguration)
**Impact:** Medium (performance degradation, no data loss)

**Detection:**
- Cache hit rate drops <50% suddenly (normal >90%)
- Feature generation latency spikes >500ms (normal <20ms with cache)
- Redis memory usage >90% (eviction policies kicking in)
- Alert: "Redis cache hit rate degraded" (Slack notification)

**Mitigation:**
- Automatic fallback to direct feature computation (slower but functional)
- Inference service continues with degraded performance (~150ms vs 50ms)
- Trading decisions unaffected (latency still acceptable for strategies)
- Investigate cache eviction policy (LRU vs LFU) and memory limits

**Recovery:**
1. Immediate (5 minutes): increase Redis memory limit if available resources exist
2. Short-term (30 minutes): identify cache warming strategy
   - Pre-compute features for most-traded symbols
   - Prioritize cache entries by access frequency
3. Medium-term (1 hour): optimize feature size
   - Compress feature vectors (reduce from 200 features to 50 most important)
   - Implement progressive feature loading (partial cache hits)
4. Long-term (1 week): Redis cluster scaling
   - Add second Redis node for horizontal scaling
   - Shard by symbol (deterministic key hashing)

**Prevention:**
- Redis cluster with 2+ nodes (16GB per node minimum)
- Monitoring of cache hit rate and memory usage (alert at 80%)
- Eviction policy tuning: `allkeys-lru` for general caching
- TTL policies: expire features after 5 minutes (staleness acceptable)
- Regular cache warming during off-hours (pre-populate common symbols)

**Runbook:** `docs/runbooks/redis-cache-degradation.md`

---

## Cross-References

- **Product Requirements:** [PRD.md](./PRD.md)
- **Implementation Plan:** [implementation-plan.md](./implementation-plan.md)
- **Test Strategy:** [test-strategy.md](./test-strategy.md)

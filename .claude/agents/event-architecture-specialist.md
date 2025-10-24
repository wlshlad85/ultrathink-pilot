# Event Architecture Specialist

Expert agent for designing and implementing event-driven architecture using Kafka to decouple forensics processing from critical trading path, reducing latency from 200-500ms to <50ms for trading decisions.

## Role and Objective

Design and implement Kafka-based event-driven architecture that decouples forensics processing (SHAP values, explanations) from the critical trading decision path. This eliminates the 200-500ms synchronous overhead currently added by forensics analysis, achieving <50ms trading latency while maintaining comprehensive audit trails through asynchronous event consumption.

**Key Deliverables:**
- 3-broker Kafka cluster with replication factor 2
- Event schemas for trading_decision and model_update events
- Kafka producer integrated into inference service trading path
- Forensics Consumer service for asynchronous SHAP value computation
- `/api/v1/forensics/{decision_id}` endpoint for audit queries
- Circuit breakers preventing Kafka outages from impacting trading

## Requirements

### Kafka Cluster Setup
- **Brokers:** 3-node cluster for high availability
- **Replication Factor:** 2 (data survives single broker failure)
- **Retention:** 7-day hot retention, S3/Glacier tiered archival
- **Throughput:** 100k forensics events/sec with <5 sec lag
- **Topics:**
  - `trading_decisions`: All trading decision events
  - `model_updates`: Incremental model update events
  - `regime_changes`: Regime probability transitions
  - `risk_decisions`: Risk approval/rejection events

### Event Schemas

**Trading Decision Event:**
```json
{
  "event_type": "trading_decision",
  "timestamp": "2025-10-21T14:30:00Z",
  "decision_id": "550e8400-e29b-41d4-a716-446655440000",
  "symbol": "AAPL",
  "action": "BUY",
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
    "volume_ratio": 1.23
    // Top 20 features for SHAP
  },
  "risk_checks": {
    "position_limit_ok": true,
    "concentration_ok": true,
    "correlation_ok": true
  }
}
```

**Model Update Event:**
```json
{
  "event_type": "model_update",
  "timestamp": "2025-10-21T15:00:00Z",
  "model_id": "bull_specialist_v127",
  "update_type": "incremental",  // incremental, full_retrain
  "data_window": {
    "start": "2025-10-14T15:00:00Z",
    "end": "2025-10-21T15:00:00Z"
  },
  "performance_metrics": {
    "sharpe_pre": 1.85,
    "sharpe_post": 1.82,
    "degradation_pct": 1.6
  },
  "stability_check": "passed"  // passed, failed_rollback
}
```

### Forensics Consumer Service
```python
class ForensicsConsumer:
    def __init__(self, kafka_bootstrap_servers: str):
        self.consumer = KafkaConsumer(
            'trading_decisions',
            bootstrap_servers=kafka_bootstrap_servers,
            group_id='forensics_processors',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

    async def process_events(self):
        """
        Asynchronous forensics processing loop
        """
        for message in self.consumer:
            event = message.value

            # SHAP value computation (CPU intensive)
            shap_values = await compute_shap_values(
                model=load_model(event['model_version']),
                features=event['features']
            )

            # Store forensics in TimescaleDB
            await store_forensics(
                decision_id=event['decision_id'],
                shap_values=shap_values,
                regime_reasoning=generate_regime_explanation(event),
                strategy_reasoning=generate_strategy_explanation(event)
            )
```

### Performance Requirements
- **Event Throughput:** 100k events/sec sustained
- **Consumer Lag:** <5 seconds P95
- **Forensics Query Latency:** <500ms P95 (non-critical path)
- **Kafka Availability:** 99.9% uptime (graceful degradation on outage)

### Circuit Breaker Pattern
```python
class KafkaCircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout  # seconds
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                logger.warning("Circuit breaker OPEN, skipping Kafka call")
                return None  # Graceful degradation

        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            raise
```

## Dependencies

**Upstream Dependencies:**
- `infrastructure-engineer`: Kafka cluster provisioning with 3 brokers
- `database-migration-specialist`: TimescaleDB for forensics audit trail storage

**Downstream Dependencies:**
- `inference-api-engineer`: Kafka event emission from inference service
- `monitoring-observability-specialist`: Kafka lag monitoring, throughput dashboards

**Collaborative Dependencies:**
- `online-learning-engineer`: Model update event consumption
- `risk-management-engineer`: Risk decision event logging
- `qa-testing-engineer`: Cascade failure testing (Kafka outage scenarios)

## Context and Constraints

### Current State (From PRD)
- **Synchronous Forensics:** SHAP values computed in trading decision path
- **Latency Overhead:** 200-500ms added to every trading decision
- **Alpha Loss:** Delayed decisions reduce profitability by ~2-5%
- **Capacity:** ~2-5 decisions/sec before latency SLA violation

### Target Architecture
```
Trading Decision (Inference Service)
         ↓
    Kafka Producer (Async, Non-Blocking)
         ↓
    Kafka Cluster (trading_decisions topic)
         ↓
    Forensics Consumer (Background Processing)
         ↓
    TimescaleDB (Audit Trail Storage)
         ↓
    Forensics API (/api/v1/forensics/{id})
```

### Integration Points
- **Inference Service:** Kafka producer emits events post-decision
- **TimescaleDB:** Forensics storage with 7-year retention
- **Monitoring:** Prometheus metrics for lag, throughput, errors
- **API Gateway:** Query endpoint for forensics retrieval

### Performance Targets
- **Trading Latency Reduction:** 200-500ms → <50ms (4-10x improvement)
- **Forensics Throughput:** 100k events/sec (20x current capacity)
- **Consumer Lag:** <5 sec (forensics available near real-time)
- **No Impact on Trading:** Circuit breaker prevents Kafka failures affecting decisions

## Tools Available

- **Read, Write, Edit:** Kafka configs, Python producers/consumers, event schemas
- **Bash:** Kafka cluster management (kafka-topics, kafka-consumer-groups)
- **Grep, Glob:** Find existing synchronous forensics code for refactoring

## Success Criteria

### Phase 1: Kafka Infrastructure (Weeks 1-2)
- ✅ 3-broker Kafka cluster operational with replication factor 2
- ✅ Topics created: trading_decisions, model_updates, regime_changes, risk_decisions
- ✅ Producer integration tested: Events successfully published
- ✅ Consumer group functional: Forensics processing from topic

### Phase 2: Forensics Decoupling (Weeks 3-4)
- ✅ Inference service emits Kafka events (async, non-blocking)
- ✅ Synchronous SHAP computation removed from trading path
- ✅ Forensics Consumer processes events with <5 sec lag
- ✅ Trading latency reduced to <50ms P95

### Phase 3: Production Hardening (Weeks 5-6)
- ✅ Circuit breaker prevents Kafka outages from blocking trades
- ✅ Load testing: 100k events/sec sustained throughput
- ✅ Tiered storage: 7-day hot retention, S3 archival configured
- ✅ /api/v1/forensics/{decision_id} endpoint operational

### Acceptance Criteria (From Test Strategy)
- Event-driven forensics with trading decisions <50ms P95
- Forensics processed asynchronously with <5 sec lag
- Kafka outage does not impact trading decision latency (circuit breaker)
- 100k forensics events/sec throughput validated

## Implementation Notes

### File Structure
```
ultrathink-pilot/
├── event_architecture/
│   ├── __init__.py
│   ├── kafka_producer.py          # Trading decision event emission
│   ├── kafka_consumer.py          # Forensics background processing
│   ├── event_schemas.py           # Pydantic models for events
│   ├── circuit_breaker.py         # Kafka failure handling
│   └── config.py                  # Kafka cluster config
├── forensics_api/
│   ├── __init__.py
│   ├── api.py                     # /api/v1/forensics endpoint
│   ├── shap_calculator.py         # SHAP value computation
│   └── explanation_generator.py  # Regime/strategy reasoning
├── kafka_config/
│   ├── server.properties          # Broker configuration
│   ├── topics.sh                  # Topic creation script
│   └── consumer_groups.yml        # Consumer group configs
└── tests/
    ├── test_producer.py           # Event emission tests
    ├── test_consumer.py           # Forensics processing tests
    └── test_circuit_breaker.py    # Failure handling tests
```

### Kafka Producer Integration
```python
from aiokafka import AIOKafkaProducer
import json

class TradingEventProducer:
    def __init__(self, bootstrap_servers: str):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='gzip',
            acks='all',  # Wait for all replicas
            retry_backoff_ms=100
        )

    async def emit_trading_decision(self, decision: EnsembleDecision):
        """
        Non-blocking event emission
        """
        event = {
            "event_type": "trading_decision",
            "timestamp": datetime.now().isoformat(),
            "decision_id": decision.id,
            "symbol": decision.symbol,
            "action": decision.action,
            "features": decision.features,
            # ... other fields
        }

        try:
            await self.producer.send_and_wait(
                topic='trading_decisions',
                value=event,
                key=decision.symbol.encode('utf-8')  # Partition by symbol
            )
        except Exception as e:
            logger.error(f"Kafka emit failed: {e}")
            # Circuit breaker handles repeated failures
```

### Forensics API Endpoint
```python
@app.get("/api/v1/forensics/{decision_id}")
async def get_forensics(decision_id: str):
    """
    Retrieve forensics for a specific trading decision
    """
    forensics = await db.query(
        f"""
        SELECT
            decision_id,
            shap_values,
            regime_reasoning,
            strategy_selection,
            execution_details
        FROM forensics_audit
        WHERE decision_id = '{decision_id}'
        """
    )

    if not forensics:
        raise HTTPException(404, "Decision ID not found")

    return ForensicsResponse(
        decision_id=decision_id,
        explanations=forensics['shap_values'],
        regime_reasoning=forensics['regime_reasoning'],
        strategy_selection=forensics['strategy_selection'],
        execution_details=forensics['execution_details']
    )
```

### Monitoring & Alerts
- **Consumer Lag:** Alert if >5 sec for 10 consecutive minutes
- **Event Backlog:** Alert if topic has >50k unconsumed messages
- **Producer Failures:** Alert if >10 consecutive send failures
- **Circuit Breaker Open:** Critical alert when trading path bypasses Kafka
- **Disk Usage:** Alert if Kafka log segments >80% disk capacity

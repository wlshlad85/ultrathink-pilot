# UltraThink Pilot Expert Agents

> **GOVERNANCE UPDATE**: These 12 operational agents now report to Deputy Agent, who reports to Master Orchestrator.

This directory contains 12 specialized expert agents for the trading system architectural enhancement project, now operating under a hierarchical 14-agent command structure.

## Hierarchy

```
Master Orchestrator (Strategic Command)
    ↓
Deputy Agent (Tactical Coordination)
    ↓
12 Specialist Agents (Operational Execution)
```

**Communication Flow**:
- Agents → Deputy: Status reports (100ms heartbeats)
- Deputy → Master: Filtered escalations (critical only)
- Master → Any Agent: Emergency override (bypass Deputy)

**Authority**: Agents execute within domain, Deputy coordinates tasks (±20%), Master has absolute override.

## Team 1: Foundation (Phase 1)
- **event-architecture-specialist** - Kafka cluster deployment and event-driven architecture
- **database-migration-specialist** - SQLite → TimescaleDB migration, MLflow optimization
- **data-pipeline-architect** - Unified data service with Redis caching

## Team 2: Core ML (Phase 2)
- **regime-detection-specialist** - Probabilistic regime detection (DPGMM)
- **meta-controller-researcher** - Hierarchical RL meta-controller for strategy blending
- **ml-training-specialist** - Training orchestrator with Celery + MLflow

## Team 3: API & Risk (Phase 2)
- **inference-api-engineer** - FastAPI inference service with TorchServe
- **risk-management-engineer** - Portfolio-level risk controls and VaR calculation

## Team 4: Support (Continuous)
- **infrastructure-engineer** - Docker orchestration and GPU resource scheduling
- **qa-testing-engineer** - Test infrastructure (85% coverage target)
- **monitoring-observability-specialist** - Grafana dashboards and Prometheus alerts
- **online-learning-engineer** - EWC implementation for incremental learning

## Usage

Agents are automatically available in the project. They now report through Deputy Agent and can receive Master overrides.

**Logs**:
- `/home/rich/ultrathink-pilot/agent-coordination/logs/[agent-name].log`
- `/home/rich/ultrathink-pilot/agent-coordination/status.json`

## Status Reporting

View agent progress:
```bash
cd /home/rich/ultrathink-pilot/agent-coordination
python3 report_status.py [foundation|ml|api]
```

## Documentation References
- **Orchestration Spec**: `master-orchestrator-system.md` (NEW - complete governance)
- **Governance Guide**: `CLAUDE.md` (updated with hierarchy)
- PRD: `trading-system-architectural-enhancement-docs/trading-system-architectural-enhancement/PRD.md`
- Technical Spec: `technical-spec.md` (49KB)
- Implementation Plan: `implementation-plan.md`

## Orchestration Integration

Each agent automatically registers with Deputy on initialization:
```python
deputy_agent.register(
    agent_id=self.agent_id,
    capabilities=self.get_capabilities(),
    resource_requirements={"cpu": 4, "memory_gb": 8}
)
```

Agents can escalate critical issues directly to Master if needed.

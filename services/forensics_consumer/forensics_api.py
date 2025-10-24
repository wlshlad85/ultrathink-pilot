"""
Forensics API - Query interface for trading decision audit trail.
FastAPI application for retrieving forensics data.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import os

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


# Pydantic models
class ForensicsRecord(BaseModel):
    """Single forensics audit record."""
    decision_id: str
    timestamp: datetime
    symbol: str
    action: str
    quantity: int
    confidence: float
    regime_probs: Dict[str, float]
    strategy_weights: Dict[str, float]
    features: Optional[Dict[str, float]] = None
    risk_checks: Optional[Dict[str, bool]] = None
    model_version: Optional[str] = None
    latency_ms: Optional[float] = None
    shap_values: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None
    created_at: datetime


class ForensicsResponse(BaseModel):
    """Response for forensics queries."""
    total: int
    records: List[ForensicsRecord]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database_connected: bool


# Database connection
def get_db_connection():
    """Get TimescaleDB connection."""
    db_config = {
        'host': os.environ.get('TIMESCALEDB_HOST', 'timescaledb'),
        'port': int(os.environ.get('TIMESCALEDB_PORT', '5432')),
        'database': os.environ.get('TIMESCALEDB_DATABASE', 'ultrathink_experiments'),
        'user': os.environ.get('TIMESCALEDB_USER', 'ultrathink'),
        'password': os.environ.get('TIMESCALEDB_PASSWORD', 'changeme_in_production')
    }

    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")


app = FastAPI(
    title="Trading Forensics API",
    description="Query interface for trading decision audit trail",
    version="1.0.0"
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": "Trading Forensics API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        conn = get_db_connection()
        conn.close()
        database_connected = True
    except:
        database_connected = False

    return HealthResponse(
        status="healthy" if database_connected else "degraded",
        database_connected=database_connected
    )


@app.get("/api/v1/forensics/{decision_id}", response_model=ForensicsRecord, tags=["Forensics"])
async def get_decision_forensics(decision_id: str):
    """
    Get forensics data for a specific decision.

    Args:
        decision_id: Decision UUID

    Returns:
        Forensics record with SHAP explanations
    """
    conn = get_db_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT * FROM trading_decisions_audit
            WHERE decision_id = %s
            """
            cursor.execute(query, (decision_id,))
            record = cursor.fetchone()

            if record is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Decision {decision_id} not found"
                )

            return ForensicsRecord(**record)

    finally:
        conn.close()


@app.get("/api/v1/forensics", response_model=ForensicsResponse, tags=["Forensics"])
async def query_forensics(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    action: Optional[str] = Query(None, description="Filter by action (BUY/SELL/HOLD)"),
    start_time: Optional[datetime] = Query(None, description="Start timestamp"),
    end_time: Optional[datetime] = Query(None, description="End timestamp"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    Query forensics audit trail with filters.

    Args:
        symbol: Filter by symbol
        action: Filter by action
        start_time: Start timestamp
        end_time: End timestamp
        limit: Max records
        offset: Pagination offset

    Returns:
        List of forensics records
    """
    conn = get_db_connection()

    try:
        # Build query with filters
        conditions = []
        params = []

        if symbol:
            conditions.append("symbol = %s")
            params.append(symbol)

        if action:
            conditions.append("action = %s")
            params.append(action.upper())

        if start_time:
            conditions.append("timestamp >= %s")
            params.append(start_time)

        if end_time:
            conditions.append("timestamp <= %s")
            params.append(end_time)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        # Count total
        count_query = f"""
        SELECT COUNT(*) as total
        FROM trading_decisions_audit
        WHERE {where_clause}
        """

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(count_query, params)
            total = cursor.fetchone()['total']

            # Fetch records
            query = f"""
            SELECT * FROM trading_decisions_audit
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
            """

            cursor.execute(query, params + [limit, offset])
            records = cursor.fetchall()

            return ForensicsResponse(
                total=total,
                records=[ForensicsRecord(**r) for r in records]
            )

    finally:
        conn.close()


@app.get("/api/v1/forensics/symbols/{symbol}/stats", tags=["Analytics"])
async def get_symbol_stats(
    symbol: str,
    days: int = Query(7, ge=1, le=365, description="Number of days to analyze")
):
    """
    Get trading statistics for a symbol.

    Args:
        symbol: Symbol to analyze
        days: Number of days to include

    Returns:
        Statistics summary
    """
    conn = get_db_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            start_time = datetime.now() - timedelta(days=days)

            query = """
            SELECT
                COUNT(*) as total_decisions,
                COUNT(*) FILTER (WHERE action = 'BUY') as buy_count,
                COUNT(*) FILTER (WHERE action = 'SELL') as sell_count,
                COUNT(*) FILTER (WHERE action = 'HOLD') as hold_count,
                AVG(confidence) as avg_confidence,
                AVG(latency_ms) as avg_latency_ms,
                MIN(timestamp) as first_decision,
                MAX(timestamp) as last_decision
            FROM trading_decisions_audit
            WHERE symbol = %s
            AND timestamp >= %s
            """

            cursor.execute(query, (symbol, start_time))
            stats = cursor.fetchone()

            return stats

    finally:
        conn.close()


@app.get("/api/v1/forensics/analytics/regime-accuracy", tags=["Analytics"])
async def get_regime_accuracy(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """
    Analyze regime detection accuracy.

    Args:
        days: Number of days to include

    Returns:
        Regime analysis
    """
    conn = get_db_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            start_time = datetime.now() - timedelta(days=days)

            query = """
            SELECT
                (regime_probs->>'bull')::float as bull_prob,
                (regime_probs->>'bear')::float as bear_prob,
                (regime_probs->>'sideways')::float as sideways_prob,
                action,
                confidence
            FROM trading_decisions_audit
            WHERE timestamp >= %s
            ORDER BY timestamp DESC
            LIMIT 1000
            """

            cursor.execute(query, (start_time,))
            records = cursor.fetchall()

            # Calculate regime distribution
            regime_counts = {"bull": 0, "bear": 0, "sideways": 0}
            for record in records:
                probs = {
                    "bull": record['bull_prob'],
                    "bear": record['bear_prob'],
                    "sideways": record['sideways_prob']
                }
                dominant_regime = max(probs.items(), key=lambda x: x[1])[0]
                regime_counts[dominant_regime] += 1

            return {
                "total_decisions": len(records),
                "regime_distribution": regime_counts,
                "avg_confidence": sum(r['confidence'] for r in records) / len(records) if records else 0
            }

    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    uvicorn.run(
        "forensics_api:app",
        host="0.0.0.0",
        port=8090,
        reload=False
    )

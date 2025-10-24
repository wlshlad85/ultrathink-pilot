"""
Regime Detection API Service
FastAPI endpoint for probabilistic regime detection with TimescaleDB integration

Endpoints:
- GET /regime/probabilities - Get current regime probability distribution
- GET /regime/history/{symbol} - Retrieve historical regime data
- POST /regime/fit - Fit/update model with new data
- GET /health - Health check

Author: regime-detection-specialist
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging
from contextlib import contextmanager

from probabilistic_regime_detector import (
    ProbabilisticRegimeDetector,
    RegimeProbabilities
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Probabilistic Regime Detection API",
    description="Market regime classification with continuous probability distributions",
    version="1.0.0"
)

# Initialize detector (singleton pattern)
detector = ProbabilisticRegimeDetector()
detector_initialized = False

# Database connection configuration
DB_CONFIG = {
    'host': os.getenv('TIMESCALEDB_HOST', 'timescaledb'),
    'port': int(os.getenv('TIMESCALEDB_PORT', 5432)),
    'database': os.getenv('TIMESCALEDB_DATABASE', 'ultrathink_experiments'),
    'user': os.getenv('TIMESCALEDB_USER', 'ultrathink'),
    'password': os.getenv('TIMESCALEDB_PASSWORD', 'changeme_in_production')
}


# Pydantic models for API
class MarketData(BaseModel):
    """Market data input for regime detection"""
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL')")
    returns_5d: float = Field(..., description="5-day cumulative returns")
    volatility_20d: float = Field(..., ge=0, description="20-day rolling volatility")
    trend_strength: float = Field(..., ge=-1, le=1, description="Trend strength [-1, 1]")
    volume_ratio: float = Field(..., gt=0, description="Volume ratio (current/avg)")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)


class RegimeResponse(BaseModel):
    """Regime probability distribution response"""
    symbol: str
    prob_bull: float = Field(..., ge=0, le=1)
    prob_bear: float = Field(..., ge=0, le=1)
    prob_sideways: float = Field(..., ge=0, le=1)
    entropy: float = Field(..., ge=0)
    dominant_regime: str
    confidence: float = Field(..., ge=0, le=1)
    timestamp: datetime
    stored_to_db: bool = False

    @validator('prob_bull', 'prob_bear', 'prob_sideways')
    def validate_probability_sum(cls, v, values):
        """Validate that probabilities sum to 1.0"""
        if 'prob_sideways' in values and 'prob_bear' in values:
            prob_sum = values['prob_bull'] + values['prob_bear'] + v
            if abs(prob_sum - 1.0) > 0.001:
                raise ValueError(f"Probabilities must sum to 1.0 (got {prob_sum:.6f})")
        return v


class HistoricalRegimeData(BaseModel):
    """Historical regime data point"""
    time: datetime
    symbol: str
    prob_bull: float
    prob_bear: float
    prob_sideways: float
    entropy: float
    detected_regime: str


class FitRequest(BaseModel):
    """Request to fit/update model"""
    market_data_history: List[Dict]
    save_model: bool = Field(default=True, description="Save model after fitting")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    detector_initialized: bool
    database_connected: bool
    timestamp: datetime


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def store_regime_to_db(symbol: str, probs: RegimeProbabilities) -> bool:
    """
    Store regime probabilities to TimescaleDB regime_history table

    Args:
        symbol: Stock symbol
        probs: Regime probability distribution

    Returns:
        True if stored successfully
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                insert_query = """
                INSERT INTO regime_history
                (time, symbol, prob_bull, prob_bear, prob_sideways, entropy, detected_regime, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """

                metadata = {
                    'confidence': probs.confidence,
                    'model_version': '1.0.0',
                    'detector_type': 'DPGMM'
                }

                cur.execute(insert_query, (
                    probs.timestamp,
                    symbol,
                    probs.prob_bull,
                    probs.prob_bear,
                    probs.prob_sideways,
                    probs.entropy,
                    probs.dominant_regime,
                    psycopg2.extras.Json(metadata)
                ))

        logger.info(f"Stored regime data for {symbol} at {probs.timestamp}")
        return True

    except Exception as e:
        logger.error(f"Failed to store regime data: {e}")
        return False


def initialize_detector():
    """Initialize detector with cached model if available"""
    global detector, detector_initialized

    model_path = '/app/models/regime_detector.pkl'

    try:
        if os.path.exists(model_path):
            detector.load_model(model_path)
            logger.info("Loaded cached model from disk")
        else:
            logger.warning("No cached model found. Detector will use bootstrap mode.")
            # Attempt to load training data from database
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get recent trading decisions as proxy for market data
                    cur.execute("""
                        SELECT
                            features->>'returns_5d' as returns_5d,
                            features->>'volatility_20d' as volatility_20d,
                            features->>'trend_strength' as trend_strength,
                            features->>'volume_ratio' as volume_ratio
                        FROM trading_decisions
                        WHERE time > NOW() - INTERVAL '90 days'
                        AND features ? 'returns_5d'
                        ORDER BY time DESC
                        LIMIT 1000
                    """)

                    rows = cur.fetchall()
                    if rows:
                        market_data = [
                            {
                                'returns_5d': float(row['returns_5d']),
                                'volatility_20d': float(row['volatility_20d']),
                                'trend_strength': float(row['trend_strength']),
                                'volume_ratio': float(row['volume_ratio'])
                            }
                            for row in rows if all(row[k] for k in row.keys())
                        ]

                        if len(market_data) >= 100:
                            detector.fit(market_data)
                            detector.save_model(model_path)
                            logger.info(f"Fitted model with {len(market_data)} historical samples")

        detector_initialized = True

    except Exception as e:
        logger.error(f"Detector initialization failed: {e}")
        detector_initialized = False


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize detector on startup"""
    logger.info("Starting Regime Detection API...")
    initialize_detector()
    logger.info(f"Detector initialized: {detector_initialized}")


# API Endpoints
@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint"""
    return {
        "service": "Probabilistic Regime Detection API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": [
            "/regime/probabilities",
            "/regime/history/{symbol}",
            "/regime/fit",
            "/health"
        ]
    }


@app.post("/regime/probabilities", response_model=RegimeResponse)
async def predict_regime_probabilities(market_data: MarketData):
    """
    Predict regime probability distribution for given market data

    Returns continuous probability distribution over [bull, bear, sideways].
    This enables smooth regime transitions for meta-controller strategy blending.

    Key Innovation: No hard regime switches - meta-controller receives full
    probability distribution and can weight strategies accordingly.
    """
    try:
        # Predict probabilities
        probs = detector.predict_probabilities(market_data.dict())

        # Store to TimescaleDB
        stored = store_regime_to_db(market_data.symbol, probs)

        response = RegimeResponse(
            symbol=market_data.symbol,
            prob_bull=probs.prob_bull,
            prob_bear=probs.prob_bear,
            prob_sideways=probs.prob_sideways,
            entropy=probs.entropy,
            dominant_regime=probs.dominant_regime,
            confidence=probs.confidence,
            timestamp=probs.timestamp,
            stored_to_db=stored
        )

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/regime/probabilities/{symbol}", response_model=RegimeResponse)
async def get_latest_regime(symbol: str):
    """
    Get latest regime probabilities for a symbol from database

    Args:
        symbol: Stock symbol

    Returns:
        Latest regime probability distribution
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT time, symbol, prob_bull, prob_bear, prob_sideways,
                           entropy, detected_regime, metadata
                    FROM regime_history
                    WHERE symbol = %s
                    ORDER BY time DESC
                    LIMIT 1
                """, (symbol,))

                row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No regime data found for symbol {symbol}"
                    )

                return RegimeResponse(
                    symbol=row['symbol'],
                    prob_bull=row['prob_bull'],
                    prob_bear=row['prob_bear'],
                    prob_sideways=row['prob_sideways'],
                    entropy=row['entropy'],
                    dominant_regime=row['detected_regime'],
                    confidence=row['metadata'].get('confidence', 0.0),
                    timestamp=row['time'],
                    stored_to_db=True
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve regime data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/regime/history/{symbol}", response_model=List[HistoricalRegimeData])
async def get_regime_history(
    symbol: str,
    hours: int = Query(default=24, ge=1, le=720, description="Hours of history to retrieve")
):
    """
    Retrieve historical regime probability data for a symbol

    Args:
        symbol: Stock symbol
        hours: Number of hours of history (default 24, max 720 = 30 days)

    Returns:
        List of historical regime probability distributions
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT time, symbol, prob_bull, prob_bear, prob_sideways,
                           entropy, detected_regime
                    FROM regime_history
                    WHERE symbol = %s
                      AND time > NOW() - INTERVAL '%s hours'
                    ORDER BY time DESC
                    LIMIT 10000
                """, (symbol, hours))

                rows = cur.fetchall()

                return [
                    HistoricalRegimeData(
                        time=row['time'],
                        symbol=row['symbol'],
                        prob_bull=row['prob_bull'],
                        prob_bear=row['prob_bear'],
                        prob_sideways=row['prob_sideways'],
                        entropy=row['entropy'],
                        detected_regime=row['detected_regime']
                    )
                    for row in rows
                ]

    except Exception as e:
        logger.error(f"Failed to retrieve regime history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/regime/fit")
async def fit_detector(request: FitRequest):
    """
    Fit or update regime detector with new market data

    This endpoint allows for model retraining with new historical data.
    Use sparingly - model supports online learning automatically.

    Args:
        request: FitRequest with market_data_history

    Returns:
        Success message with model statistics
    """
    try:
        if len(request.market_data_history) < 100:
            raise HTTPException(
                status_code=400,
                detail="Need at least 100 samples to fit model"
            )

        detector.fit(request.market_data_history)

        if request.save_model:
            model_path = '/app/models/regime_detector.pkl'
            os.makedirs('/app/models', exist_ok=True)
            detector.save_model(model_path)

        global detector_initialized
        detector_initialized = True

        return {
            "status": "success",
            "message": f"Model fitted with {len(request.market_data_history)} samples",
            "active_components": detector._count_active_components(),
            "regime_mapping": detector.regime_mapping,
            "model_saved": request.save_model
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model fitting failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring

    Checks:
    - API server status
    - Detector initialization status
    - Database connectivity

    Returns:
        Health status
    """
    db_connected = False

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                db_connected = True
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")

    status = "healthy" if (detector_initialized and db_connected) else "degraded"

    return HealthResponse(
        status=status,
        detector_initialized=detector_initialized,
        database_connected=db_connected,
        timestamp=datetime.utcnow()
    )


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

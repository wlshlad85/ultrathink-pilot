"""
Meta-Controller API Service
FastAPI endpoint for strategy weight blending based on regime probabilities

Endpoints:
- POST /api/v1/meta-controller/decide - Get strategy weights
- GET /api/v1/meta-controller/history/{symbol} - Get decision history
- GET /health - Health check
- POST /api/v1/meta-controller/update - Trigger policy update (admin)

Author: meta-controller-researcher (Agent 10/12)
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import os
from contextlib import asynccontextmanager
import requests

from meta_controller_v2 import (
    MetaControllerRL,
    MetaControllerDB,
    RegimeInput,
    StrategyWeights
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REGIME_DETECTION_URL = os.getenv('REGIME_DETECTION_URL', 'http://regime-detection:8001')
MODEL_PATH = os.getenv('META_CONTROLLER_MODEL_PATH', '/app/models/meta_controller.pt')

# Global instances
meta_controller: Optional[MetaControllerRL] = None
db_interface: Optional[MetaControllerDB] = None


# Pydantic models for API
class MarketFeatures(BaseModel):
    """Market indicators for decision making"""
    recent_pnl: float = Field(0.0, description="Recent P&L")
    volatility_20d: float = Field(..., ge=0, description="20-day volatility")
    trend_strength: float = Field(..., ge=-1, le=1, description="Trend strength [-1, 1]")
    volume_ratio: float = Field(..., gt=0, description="Volume ratio")


class DecisionRequest(BaseModel):
    """Request for strategy weight decision"""
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC-USD')")
    prob_bull: float = Field(..., ge=0, le=1, description="Bull regime probability")
    prob_bear: float = Field(..., ge=0, le=1, description="Bear regime probability")
    prob_sideways: float = Field(..., ge=0, le=1, description="Sideways regime probability")
    entropy: float = Field(..., ge=0, description="Regime entropy (uncertainty)")
    confidence: float = Field(..., ge=0, le=1, description="Regime confidence")
    market_features: MarketFeatures
    use_epsilon_greedy: bool = Field(True, description="Apply exploration")
    store_to_db: bool = Field(True, description="Store decision to TimescaleDB")

    @validator('prob_bull', 'prob_bear', 'prob_sideways')
    def validate_probabilities(cls, v, values):
        """Validate probabilities sum to 1.0"""
        if 'prob_sideways' in values and 'prob_bear' in values:
            prob_sum = values.get('prob_bull', 0) + values.get('prob_bear', 0) + v
            if abs(prob_sum - 1.0) > 0.001:
                raise ValueError(f"Probabilities must sum to 1.0 (got {prob_sum:.6f})")
        return v


class DecisionResponse(BaseModel):
    """Response with strategy weights"""
    symbol: str
    weights: Dict[str, float]
    method: str = Field(..., description="'hierarchical_rl', 'fallback', or 'bootstrap'")
    confidence: float
    timestamp: datetime
    regime_input: Dict[str, float]
    stored_to_db: bool = False


class HistoryResponse(BaseModel):
    """Historical decision record"""
    time: datetime
    symbol: str
    regime_probabilities: Dict[str, float]
    strategy_weights: Dict[str, float]
    method: str
    confidence: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    database_connected: bool
    epsilon: float
    episodes_trained: int
    timestamp: datetime


class PolicyUpdateRequest(BaseModel):
    """Request to update policy (admin only)"""
    update_epochs: int = Field(4, ge=1, le=20, description="Number of update epochs")
    clip_epsilon: float = Field(0.2, ge=0.1, le=0.5, description="PPO clipping parameter")


class PolicyUpdateResponse(BaseModel):
    """Policy update statistics"""
    success: bool
    stats: Optional[Dict[str, float]]
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle"""
    global meta_controller, db_interface

    logger.info("Starting Meta-Controller API...")

    # Initialize meta-controller
    meta_controller = MetaControllerRL(
        learning_rate=1e-4,
        gamma=0.99,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    # Load cached model if exists
    if os.path.exists(MODEL_PATH):
        meta_controller.load_model(MODEL_PATH)
        logger.info(f"Loaded cached model from {MODEL_PATH}")
    else:
        logger.warning("No cached model found, starting fresh")

    # Initialize database interface
    try:
        db_interface = MetaControllerDB(
            host=os.getenv('TIMESCALEDB_HOST', 'timescaledb'),
            port=int(os.getenv('TIMESCALEDB_PORT', 5432)),
            database=os.getenv('TIMESCALEDB_DATABASE', 'ultrathink_experiments'),
            user=os.getenv('TIMESCALEDB_USER', 'ultrathink'),
            password=os.getenv('TIMESCALEDB_PASSWORD', 'changeme_in_production')
        )
        logger.info("Database interface initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        db_interface = None

    yield

    # Cleanup
    logger.info("Shutting down Meta-Controller API...")
    if meta_controller:
        meta_controller.save_model(MODEL_PATH)


# Initialize FastAPI app
app = FastAPI(
    title="Meta-Controller API",
    description="Hierarchical RL strategy weight blending for smooth regime transitions",
    version="2.0.0",
    lifespan=lifespan
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint"""
    return {
        "service": "Meta-Controller API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": [
            "/api/v1/meta-controller/decide",
            "/api/v1/meta-controller/history/{symbol}",
            "/api/v1/meta-controller/update",
            "/health"
        ]
    }


@app.post("/api/v1/meta-controller/decide", response_model=DecisionResponse)
async def decide_strategy_weights(request: DecisionRequest):
    """
    Decide strategy weights based on regime probabilities

    This is the main endpoint for the meta-controller. It takes regime
    probabilities and market features as input and returns continuous
    strategy weights for blending specialist models.

    Key Innovation: Returns continuous weights (not discrete selection)
    to enable smooth transitions during regime changes.
    """
    try:
        if meta_controller is None:
            raise HTTPException(status_code=503, detail="Meta-controller not initialized")

        # Create RegimeInput
        regime_input = RegimeInput(
            prob_bull=request.prob_bull,
            prob_bear=request.prob_bear,
            prob_sideways=request.prob_sideways,
            entropy=request.entropy,
            confidence=request.confidence,
            timestamp=datetime.utcnow()
        )

        # Validate regime input
        regime_input.validate()

        # Extract market features
        market_features = {
            'recent_pnl': request.market_features.recent_pnl,
            'volatility_20d': request.market_features.volatility_20d,
            'trend_strength': request.market_features.trend_strength,
            'volume_ratio': request.market_features.volume_ratio
        }

        # Predict strategy weights
        strategy_weights = meta_controller.predict_weights(
            regime_input=regime_input,
            market_features=market_features,
            use_epsilon_greedy=request.use_epsilon_greedy
        )

        # Store to database
        stored = False
        if request.store_to_db and db_interface:
            stored = db_interface.store_decision(
                symbol=request.symbol,
                regime_input=regime_input,
                strategy_weights=strategy_weights,
                market_features=market_features
            )

        # Build response
        response = DecisionResponse(
            symbol=request.symbol,
            weights={
                'bull_specialist': strategy_weights.bull_specialist,
                'bear_specialist': strategy_weights.bear_specialist,
                'sideways_specialist': strategy_weights.sideways_specialist,
                'momentum': strategy_weights.momentum,
                'mean_reversion': strategy_weights.mean_reversion
            },
            method=strategy_weights.method,
            confidence=strategy_weights.confidence,
            timestamp=strategy_weights.timestamp,
            regime_input={
                'prob_bull': regime_input.prob_bull,
                'prob_bear': regime_input.prob_bear,
                'prob_sideways': regime_input.prob_sideways,
                'entropy': regime_input.entropy,
                'confidence': regime_input.confidence
            },
            stored_to_db=stored
        )

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Decision failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Decision failed: {str(e)}")


@app.get("/api/v1/meta-controller/decide/{symbol}", response_model=DecisionResponse)
async def decide_from_regime_service(
    symbol: str,
    use_epsilon_greedy: bool = Query(True, description="Apply exploration"),
    store_to_db: bool = Query(True, description="Store to database")
):
    """
    Get strategy weights by fetching regime probabilities from regime detection service

    This is a convenience endpoint that automatically fetches the latest
    regime probabilities from the regime detection service.

    Args:
        symbol: Trading symbol
        use_epsilon_greedy: Apply epsilon-greedy exploration
        store_to_db: Store decision to database

    Returns:
        DecisionResponse with strategy weights
    """
    try:
        # Fetch regime probabilities from regime detection service
        regime_url = f"{REGIME_DETECTION_URL}/regime/probabilities/{symbol}"
        logger.info(f"Fetching regime data from {regime_url}")

        response = requests.get(regime_url, timeout=5)
        response.raise_for_status()
        regime_data = response.json()

        # Extract regime probabilities
        request_data = DecisionRequest(
            symbol=symbol,
            prob_bull=regime_data['prob_bull'],
            prob_bear=regime_data['prob_bear'],
            prob_sideways=regime_data['prob_sideways'],
            entropy=regime_data['entropy'],
            confidence=regime_data['confidence'],
            market_features=MarketFeatures(
                recent_pnl=0.0,  # TODO: Fetch from metrics service
                volatility_20d=0.02,  # Default
                trend_strength=0.0,  # Default
                volume_ratio=1.0  # Default
            ),
            use_epsilon_greedy=use_epsilon_greedy,
            store_to_db=store_to_db
        )

        # Delegate to main decision endpoint
        return await decide_strategy_weights(request_data)

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch regime data: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Regime detection service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Decision failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/meta-controller/history/{symbol}", response_model=List[HistoryResponse])
async def get_decision_history(
    symbol: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum records")
):
    """
    Retrieve historical meta-controller decisions for a symbol

    Args:
        symbol: Trading symbol
        limit: Maximum number of records to return

    Returns:
        List of historical decisions
    """
    try:
        if db_interface is None:
            raise HTTPException(status_code=503, detail="Database not initialized")

        decisions = db_interface.get_recent_decisions(symbol=symbol, limit=limit)

        return [
            HistoryResponse(
                time=d['time'],
                symbol=d['symbol'],
                regime_probabilities={
                    'prob_bull': d['prob_bull'],
                    'prob_bear': d['prob_bear'],
                    'prob_sideways': d['prob_sideways'],
                    'entropy': d['regime_entropy']
                },
                strategy_weights={
                    'bull_specialist': d['weight_bull_specialist'],
                    'bear_specialist': d['weight_bear_specialist'],
                    'sideways_specialist': d['weight_sideways_specialist'],
                    'momentum': d['weight_momentum'],
                    'mean_reversion': d['weight_mean_reversion']
                },
                method=d['method'],
                confidence=d['confidence']
            )
            for d in decisions
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"History retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/meta-controller/update", response_model=PolicyUpdateResponse)
async def update_policy(request: PolicyUpdateRequest):
    """
    Trigger policy update (PPO training step)

    This endpoint is for admin/training purposes. In production, policy
    updates happen automatically via online learning.

    Args:
        request: Update configuration

    Returns:
        Update statistics
    """
    try:
        if meta_controller is None:
            raise HTTPException(status_code=503, detail="Meta-controller not initialized")

        stats = meta_controller.update_policy(
            update_epochs=request.update_epochs,
            clip_epsilon=request.clip_epsilon
        )

        if not stats:
            return PolicyUpdateResponse(
                success=False,
                stats=None,
                message="Insufficient experience for update (need >32 samples)"
            )

        # Save updated model
        meta_controller.save_model(MODEL_PATH)

        return PolicyUpdateResponse(
            success=True,
            stats=stats,
            message=f"Policy updated successfully (episode {stats.get('episode', 0)})"
        )

    except Exception as e:
        logger.error(f"Policy update failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Checks:
    - Model initialization status
    - Database connectivity
    - Current epsilon (exploration rate)
    - Training progress

    Returns:
        Health status
    """
    model_loaded = meta_controller is not None
    db_connected = False

    try:
        if db_interface:
            # Simple connectivity check
            with db_interface._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    db_connected = True
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")

    status = "healthy" if (model_loaded and db_connected) else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        database_connected=db_connected,
        epsilon=meta_controller.epsilon if meta_controller else 0.0,
        episodes_trained=meta_controller.episode_count if meta_controller else 0,
        timestamp=datetime.utcnow()
    )


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

"""
Risk Manager Service - FastAPI Application
Provides REST API for portfolio risk management
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

from portfolio_risk_manager import (
    PortfolioRiskManager,
    RiskCheckResult,
    RejectionReason
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
risk_checks_total = Counter(
    'risk_checks_total',
    'Total number of risk checks',
    ['result']  # approved, rejected
)
risk_check_latency = Histogram(
    'risk_check_latency_seconds',
    'Risk check latency in seconds',
    buckets=[0.001, 0.005, 0.010, 0.025, 0.050, 0.100]
)
portfolio_value_gauge = Gauge(
    'portfolio_value_dollars',
    'Current portfolio value in dollars'
)
position_count_gauge = Gauge(
    'position_count',
    'Number of open positions'
)
leverage_ratio_gauge = Gauge(
    'leverage_ratio',
    'Current portfolio leverage ratio'
)
var_gauge = Gauge(
    'var_95_1d_dollars',
    'Value at Risk (95% confidence, 1-day horizon) in dollars'
)

# Global risk manager instance
risk_manager: Optional[PortfolioRiskManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global risk_manager
    logger.info("Starting Risk Manager Service...")

    # Initialize risk manager with production settings
    risk_manager = PortfolioRiskManager(
        total_capital=1_000_000.0,
        max_position_pct=0.25,
        max_sector_pct=0.50,
        max_leverage=1.5,
        max_daily_loss_pct=0.02
    )

    # Start background tasks
    asyncio.create_task(daily_reset_task())
    asyncio.create_task(metrics_update_task())

    logger.info("Risk Manager Service started successfully")
    yield

    logger.info("Shutting down Risk Manager Service...")


# FastAPI app
app = FastAPI(
    title="Risk Manager Service",
    description="Portfolio-level risk constraint enforcement",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response models
class TradeCheckRequest(BaseModel):
    """Request to validate a proposed trade"""
    symbol: str = Field(..., description="Trading symbol (e.g., 'AAPL')")
    action: str = Field(..., description="Trade action: BUY or SELL")
    quantity: int = Field(..., gt=0, description="Number of shares")
    estimated_price: float = Field(..., gt=0, description="Expected execution price")
    sector: str = Field(default="unknown", description="Asset sector")

    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "action": "BUY",
                "quantity": 100,
                "estimated_price": 175.23,
                "sector": "technology"
            }
        }


class RejectionReasonResponse(BaseModel):
    """Risk check rejection reason"""
    code: str
    message: str
    limit: float
    proposed: float


class RiskAssessmentResponse(BaseModel):
    """Risk assessment details"""
    position_after_trade: Dict
    portfolio_impact: Dict
    limit_utilization: Dict


class TradeCheckResponse(BaseModel):
    """Response from trade validation"""
    approved: bool
    rejection_reasons: list[RejectionReasonResponse] = []
    allowed_quantity: Optional[int] = None
    risk_assessment: Optional[RiskAssessmentResponse] = None
    warnings: list[str] = []
    timestamp: datetime


class ExecutionUpdate(BaseModel):
    """Update from execution engine"""
    symbol: str
    action: str
    quantity: int
    execution_price: float
    sector: str = "unknown"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PortfolioStateResponse(BaseModel):
    """Current portfolio state"""
    portfolio: Dict
    risk_metrics: Dict
    limit_utilization: Dict
    last_updated: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    timestamp: datetime
    portfolio_value: float
    position_count: int


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="risk-manager",
        version="1.0.0",
        timestamp=datetime.utcnow(),
        portfolio_value=risk_manager.portfolio_value,
        position_count=len(risk_manager.positions)
    )


@app.post("/api/v1/risk/check", response_model=TradeCheckResponse)
async def check_trade(request: TradeCheckRequest):
    """
    Validate proposed trade against portfolio risk constraints

    Target latency: <10ms P95

    Returns:
    - approved: Boolean indicating if trade is allowed
    - rejection_reasons: List of constraint violations (if rejected)
    - allowed_quantity: Maximum allowed quantity (if rejected due to size)
    - risk_assessment: Detailed impact analysis
    """
    start_time = datetime.utcnow()

    try:
        # Validate action
        if request.action not in ["BUY", "SELL"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: {request.action}. Must be BUY or SELL"
            )

        # Perform risk check
        result = await risk_manager.check_trade(
            symbol=request.symbol,
            action=request.action,
            quantity=request.quantity,
            estimated_price=request.estimated_price,
            sector=request.sector
        )

        # Record metrics
        risk_checks_total.labels(
            result="approved" if result.approved else "rejected"
        ).inc()

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        risk_check_latency.observe(elapsed)

        # Convert rejection reasons
        rejection_reasons = [
            RejectionReasonResponse(
                code=reason.code.value,
                message=reason.message,
                limit=reason.limit,
                proposed=reason.proposed
            )
            for reason in result.rejection_reasons
        ]

        # Build response
        response = TradeCheckResponse(
            approved=result.approved,
            rejection_reasons=rejection_reasons,
            allowed_quantity=result.allowed_quantity,
            risk_assessment=result.risk_assessment,
            warnings=result.warnings,
            timestamp=result.timestamp
        )

        logger.info(
            f"Risk check: {request.symbol} {request.action} {request.quantity}@${request.estimated_price:.2f} - "
            f"{'APPROVED' if result.approved else 'REJECTED'} ({elapsed*1000:.2f}ms)"
        )

        return response

    except Exception as e:
        logger.error(f"Risk check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk check failed: {str(e)}"
        )


@app.post("/api/v1/risk/execution")
async def update_execution(update: ExecutionUpdate):
    """
    Update portfolio state from execution

    Called by execution engine after trade completion
    """
    try:
        risk_manager.update_position(
            symbol=update.symbol,
            quantity=update.quantity if update.action == "BUY" else -update.quantity,
            price=update.execution_price,
            sector=update.sector
        )

        logger.info(
            f"Execution update: {update.symbol} {update.action} {update.quantity}@${update.execution_price:.2f}"
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "updated",
                "portfolio_value": risk_manager.portfolio_value,
                "position_count": len(risk_manager.positions)
            }
        )

    except Exception as e:
        logger.error(f"Execution update failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution update failed: {str(e)}"
        )


@app.get("/api/v1/risk/portfolio", response_model=PortfolioStateResponse)
async def get_portfolio():
    """
    Get current portfolio state

    Returns:
    - Complete portfolio holdings
    - Risk metrics (VaR, leverage, daily P&L)
    - Limit utilization percentages
    """
    try:
        state = risk_manager.get_portfolio_state()
        return PortfolioStateResponse(**state)

    except Exception as e:
        logger.error(f"Portfolio state retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Portfolio state retrieval failed: {str(e)}"
        )


@app.post("/api/v1/risk/price-update")
async def update_price(symbol: str, price: float):
    """Update current price for a position"""
    try:
        risk_manager.update_price(symbol, price)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "updated",
                "symbol": symbol,
                "price": price
            }
        )

    except Exception as e:
        logger.error(f"Price update failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Price update failed: {str(e)}"
        )


@app.post("/api/v1/risk/reset-daily")
async def reset_daily():
    """Reset daily P&L tracking (called at market open)"""
    try:
        risk_manager.reset_daily_metrics()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "reset",
                "daily_start_value": risk_manager.daily_start_value
            }
        )

    except Exception as e:
        logger.error(f"Daily reset failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Daily reset failed: {str(e)}"
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


# Background tasks
async def daily_reset_task():
    """Reset daily metrics at midnight UTC"""
    while True:
        try:
            now = datetime.utcnow()
            # Calculate seconds until next midnight
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if tomorrow <= now:
                tomorrow = tomorrow.replace(day=tomorrow.day + 1)

            sleep_seconds = (tomorrow - now).total_seconds()
            await asyncio.sleep(sleep_seconds)

            # Reset daily metrics
            risk_manager.reset_daily_metrics()
            logger.info("Daily metrics reset completed")

        except Exception as e:
            logger.error(f"Daily reset task error: {e}", exc_info=True)
            await asyncio.sleep(3600)  # Retry in 1 hour


async def metrics_update_task():
    """Update Prometheus metrics every 10 seconds"""
    while True:
        try:
            await asyncio.sleep(10)

            # Update gauges
            portfolio_value_gauge.set(risk_manager.portfolio_value)
            position_count_gauge.set(len(risk_manager.positions))
            leverage_ratio_gauge.set(risk_manager.leverage_ratio)
            var_gauge.set(risk_manager.calculate_var())

        except Exception as e:
            logger.error(f"Metrics update task error: {e}", exc_info=True)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )

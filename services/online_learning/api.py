#!/usr/bin/env python3
"""
Online Learning API

FastAPI service for incremental model updates with EWC.

Endpoints:
- POST /api/v1/models/online-update - Trigger incremental update
- GET /api/v1/models/stability - Get stability status
- GET /api/v1/models/performance - Get performance metrics
- POST /api/v1/models/rollback - Rollback to previous checkpoint
- GET /api/v1/health - Health check

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8005
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
import logging
from pathlib import Path
import torch

from ewc_trainer import EWCTrainer, EWCConfig
from stability_checker import StabilityChecker, StabilityStatus
from data_manager import SlidingWindowDataManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Online Learning Service",
    description="Incremental model updates with Elastic Weight Consolidation",
    version="1.0.0"
)

# Global state
trainer: Optional[EWCTrainer] = None
stability_checker: Optional[StabilityChecker] = None
data_manager: Optional[SlidingWindowDataManager] = None
update_in_progress: bool = False


# Request/Response Models
class OnlineUpdateRequest(BaseModel):
    """Request to perform incremental update."""
    window_days: int = Field(60, ge=30, le=90, description="Data window size in days")
    learning_rate: Optional[float] = Field(None, ge=1e-7, le=1e-3)
    ewc_lambda: Optional[float] = Field(None, ge=100, le=10000)
    skip_stability_check: bool = Field(False, description="Skip stability check (dangerous)")


class OnlineUpdateResponse(BaseModel):
    """Response from incremental update."""
    success: bool
    update_count: int
    metrics: Dict
    stability_status: str
    degradation_percent: float
    checkpoint_path: str
    timestamp: str


class StabilityStatusResponse(BaseModel):
    """Response with stability status."""
    status: str
    last_check: Optional[str]
    degradation_percent: Optional[float]
    performance_trend: Dict
    alerts_count: int


class PerformanceMetricsResponse(BaseModel):
    """Response with performance metrics."""
    sharpe_ratio: float
    total_return: float
    volatility: float
    max_drawdown: float
    win_rate: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    trainer_initialized: bool
    update_in_progress: bool
    last_update: Optional[str]
    uptime_seconds: float


# Startup/Shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    global trainer, stability_checker, data_manager

    logger.info("Initializing Online Learning Service...")

    # Initialize components
    try:
        # Load model architecture (placeholder - should load actual model)
        from rl.ppo_agent import ActorCritic
        state_dim = 43
        action_dim = 3
        model = ActorCritic(state_dim, action_dim)

        # Initialize trainer
        config = EWCConfig()
        trainer = EWCTrainer(model, config)

        # Initialize stability checker
        stability_checker = StabilityChecker(
            alert_file="/home/rich/ultrathink-pilot/rl/models/online_learning/stability_alerts.jsonl"
        )

        # Initialize data manager
        data_manager = SlidingWindowDataManager(
            data_dir="/home/rich/ultrathink-pilot/data"
        )

        logger.info("Online Learning Service initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Online Learning Service...")


# API Endpoints
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import time

    last_update = None
    if trainer and trainer.ewc_state.last_update_time:
        last_update = trainer.ewc_state.last_update_time.isoformat()

    return HealthResponse(
        status="healthy",
        service="online_learning",
        trainer_initialized=trainer is not None,
        update_in_progress=update_in_progress,
        last_update=last_update,
        uptime_seconds=time.time()  # Simplified
    )


@app.post("/api/v1/models/online-update", response_model=OnlineUpdateResponse)
async def trigger_online_update(
    request: OnlineUpdateRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger incremental model update with EWC.

    Process:
    1. Fetch sliding window data (30-90 days)
    2. Evaluate current model performance
    3. Perform incremental update with EWC
    4. Check stability (Sharpe ratio comparison)
    5. Rollback if degradation >30%
    6. Save checkpoint
    """
    global update_in_progress

    if update_in_progress:
        raise HTTPException(status_code=409, detail="Update already in progress")

    if not trainer or not stability_checker or not data_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")

    update_in_progress = True

    try:
        logger.info(f"Starting online update with window_days={request.window_days}")

        # Update configuration if provided
        if request.learning_rate:
            trainer.config.learning_rate = request.learning_rate
            trainer.optimizer.param_groups[0]['lr'] = request.learning_rate

        if request.ewc_lambda:
            trainer.config.ewc_lambda = request.ewc_lambda

        # Step 1: Get sliding window data
        logger.info("Fetching sliding window data...")
        train_loader, val_loader = data_manager.get_data_loaders(
            window_days=request.window_days,
            batch_size=trainer.config.batch_size
        )

        # Step 2: Evaluate current performance (before update)
        logger.info("Evaluating performance before update...")
        # Create test environment (placeholder)
        from rl.trading_env_v3 import TradingEnv
        test_data = data_manager.get_test_data()
        test_env = TradingEnv(test_data)

        metrics_before = stability_checker.evaluate_performance(
            trainer.model,
            test_env,
            num_episodes=100
        )

        # Step 3: Compute Fisher Information if first update
        if trainer.ewc_state.update_count == 0:
            logger.info("Computing Fisher Information Matrix (first update)...")
            trainer.compute_fisher_information(train_loader)

        # Step 4: Perform incremental update
        logger.info("Performing incremental update with EWC regularization...")
        update_metrics = trainer.incremental_update(train_loader, val_loader)

        # Step 5: Evaluate performance after update
        logger.info("Evaluating performance after update...")
        metrics_after = stability_checker.evaluate_performance(
            trainer.model,
            test_env,
            num_episodes=100
        )

        # Step 6: Check stability
        stability_result = stability_checker.check_stability(
            metrics_before,
            metrics_after
        )

        # Step 7: Rollback if needed
        if stability_result.should_rollback and not request.skip_stability_check:
            logger.error("Stability check failed! Rolling back to previous checkpoint...")

            # Find last checkpoint
            checkpoints = sorted(
                Path(trainer.config.checkpoint_dir).glob("ewc_checkpoint_*.pth"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if checkpoints:
                trainer.load_checkpoint(str(checkpoints[0]))
                logger.info(f"Rolled back to: {checkpoints[0]}")
            else:
                logger.error("No checkpoints available for rollback!")

            update_in_progress = False

            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Stability check failed - model rolled back",
                    "degradation_percent": stability_result.degradation_percent,
                    "message": stability_result.alert_message
                }
            )

        # Step 8: Save checkpoint on success
        logger.info("Saving checkpoint...")
        checkpoint_path = trainer.save_checkpoint(
            metadata={
                'window_days': request.window_days,
                'sharpe_before': metrics_before.sharpe_ratio,
                'sharpe_after': metrics_after.sharpe_ratio,
                'degradation': stability_result.degradation_percent
            }
        )

        # Update Fisher Information for next update
        logger.info("Updating Fisher Information Matrix...")
        trainer.compute_fisher_information(train_loader)

        update_in_progress = False

        return OnlineUpdateResponse(
            success=True,
            update_count=trainer.ewc_state.update_count,
            metrics=update_metrics,
            stability_status=stability_result.status.value,
            degradation_percent=stability_result.degradation_percent,
            checkpoint_path=checkpoint_path,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        update_in_progress = False
        logger.error(f"Online update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/stability", response_model=StabilityStatusResponse)
async def get_stability_status():
    """Get current stability status."""
    if not stability_checker:
        raise HTTPException(status_code=500, detail="Stability checker not initialized")

    last_check = None
    degradation = None

    if stability_checker.stability_checks:
        last = stability_checker.stability_checks[-1]
        last_check = datetime.now().isoformat()  # Simplified
        degradation = last.degradation_percent

    trend = stability_checker.get_performance_trend()

    return StabilityStatusResponse(
        status=last.status.value if stability_checker.stability_checks else "unknown",
        last_check=last_check,
        degradation_percent=degradation,
        performance_trend=trend,
        alerts_count=len(stability_checker.alerts)
    )


@app.get("/api/v1/models/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics():
    """Get latest performance metrics."""
    if not stability_checker or not stability_checker.performance_history:
        raise HTTPException(status_code=404, detail="No performance data available")

    latest = stability_checker.performance_history[-1]

    return PerformanceMetricsResponse(
        sharpe_ratio=latest.sharpe_ratio,
        total_return=latest.total_return,
        volatility=latest.volatility,
        max_drawdown=latest.max_drawdown,
        win_rate=latest.win_rate,
        timestamp=latest.timestamp.isoformat()
    )


@app.post("/api/v1/models/rollback")
async def rollback_model(checkpoint_index: int = 0):
    """
    Rollback to previous checkpoint.

    Args:
        checkpoint_index: Index of checkpoint (0=most recent, 1=second most recent, etc.)
    """
    if not trainer:
        raise HTTPException(status_code=500, detail="Trainer not initialized")

    checkpoints = sorted(
        Path(trainer.config.checkpoint_dir).glob("ewc_checkpoint_*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if checkpoint_index >= len(checkpoints):
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint index {checkpoint_index} not found. "
                   f"Only {len(checkpoints)} checkpoints available."
        )

    checkpoint = checkpoints[checkpoint_index]
    trainer.load_checkpoint(str(checkpoint))

    logger.info(f"Rolled back to checkpoint: {checkpoint}")

    return {
        "success": True,
        "checkpoint": str(checkpoint),
        "update_count": trainer.ewc_state.update_count,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/models/checkpoints")
async def list_checkpoints():
    """List available checkpoints."""
    if not trainer:
        raise HTTPException(status_code=500, detail="Trainer not initialized")

    checkpoints = sorted(
        Path(trainer.config.checkpoint_dir).glob("ewc_checkpoint_*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return {
        "checkpoints": [
            {
                "path": str(cp),
                "timestamp": datetime.fromtimestamp(cp.stat().st_mtime).isoformat(),
                "size_mb": cp.stat().st_size / (1024 * 1024)
            }
            for cp in checkpoints
        ],
        "count": len(checkpoints)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        log_level="info"
    )

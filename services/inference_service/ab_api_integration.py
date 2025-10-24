"""
A/B Testing API integration for Inference Service.
Add these routes to your FastAPI app to enable A/B testing functionality.
"""
from fastapi import APIRouter, HTTPException, status
from typing import List
import logging

from models import (
    CreateABTestRequest, ABTestConfigResponse, UpdateTrafficSplitRequest,
    ABTestStatsResponse, ABTestMode, PredictRequest, PredictResponse
)
from ab_testing_manager import ABTestingManager, ABTestMode as ManagerABTestMode
from ab_storage import get_storage_backend

logger = logging.getLogger(__name__)

# Create router for A/B testing endpoints
ab_router = APIRouter(prefix="/api/v1/ab-test", tags=["A/B Testing"])


# Global A/B testing manager (will be initialized by main app)
ab_manager: ABTestingManager = None


def initialize_ab_manager(model_cache):
    """
    Initialize A/B testing manager with model cache.
    Call this from the main app's lifespan function.
    """
    global ab_manager
    ab_manager = ABTestingManager(model_cache=model_cache)
    logger.info("A/B Testing Manager initialized")
    return ab_manager


async def initialize_ab_storage():
    """
    Initialize storage backend for A/B testing.
    Call this from the main app's lifespan function.
    """
    try:
        storage = await get_storage_backend()
        if ab_manager:
            ab_manager.storage_backend = storage
        logger.info("A/B Testing storage backend initialized")
        return storage
    except Exception as e:
        logger.warning(f"Failed to initialize A/B storage backend: {e}")
        logger.warning("A/B testing will work but results won't be persisted")
        return None


@ab_router.post("/create", response_model=ABTestConfigResponse, status_code=status.HTTP_201_CREATED)
async def create_ab_test(request: CreateABTestRequest):
    """
    Create a new A/B test.

    Example:
    ```
    {
        "test_id": "bull_v2_canary",
        "control_model": "bull_specialist",
        "treatment_model": "bull_specialist_v2",
        "traffic_split": 0.05,
        "mode": "traffic_split",
        "description": "Canary deployment of bull specialist v2"
    }
    ```
    """
    if ab_manager is None:
        raise HTTPException(
            status_code=503,
            detail="A/B testing manager not initialized"
        )

    try:
        # Map API mode to manager mode
        mode_map = {
            ABTestMode.TRAFFIC_SPLIT: ManagerABTestMode.TRAFFIC_SPLIT,
            ABTestMode.SHADOW: ManagerABTestMode.SHADOW,
            ABTestMode.DISABLED: ManagerABTestMode.DISABLED
        }

        config = ab_manager.create_test(
            test_id=request.test_id,
            control_model=request.control_model,
            treatment_model=request.treatment_model,
            traffic_split=request.traffic_split,
            mode=mode_map[request.mode],
            description=request.description
        )

        return ABTestConfigResponse(
            test_id=config.test_id,
            control_model=config.control_model,
            treatment_model=config.treatment_model,
            traffic_split=config.traffic_split,
            mode=request.mode,
            enabled=config.enabled,
            created_at=config.created_at,
            description=config.description
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create A/B test: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@ab_router.get("/list", response_model=List[ABTestConfigResponse])
async def list_ab_tests():
    """List all active A/B tests."""
    if ab_manager is None:
        raise HTTPException(
            status_code=503,
            detail="A/B testing manager not initialized"
        )

    tests = ab_manager.list_tests()
    return [
        ABTestConfigResponse(
            test_id=test['test_id'],
            control_model=test['control_model'],
            treatment_model=test['treatment_model'],
            traffic_split=test['traffic_split'],
            mode=test['mode'],
            enabled=test['enabled'],
            created_at=test['created_at'],
            description=test['description']
        )
        for test in tests
    ]


@ab_router.get("/{test_id}/stats", response_model=ABTestStatsResponse)
async def get_ab_test_stats(test_id: str, hours_back: int = 24):
    """
    Get statistics for an A/B test.

    Args:
        test_id: Test identifier
        hours_back: Number of hours to analyze (default: 24)
    """
    if ab_manager is None:
        raise HTTPException(
            status_code=503,
            detail="A/B testing manager not initialized"
        )

    # Try to get from storage backend first
    if ab_manager.storage_backend:
        try:
            stats = await ab_manager.storage_backend.get_test_stats(test_id, hours_back)
            return ABTestStatsResponse(**stats)
        except Exception as e:
            logger.warning(f"Failed to get stats from storage: {e}")

    # Fallback to buffered stats
    stats = ab_manager.get_test_stats(test_id)
    return ABTestStatsResponse(
        test_id=stats.get('test_id', test_id),
        time_window_hours=hours_back,
        total_samples=stats.get('sample_size', 0),
        control_count=stats.get('control_count', 0),
        treatment_count=stats.get('treatment_count', 0),
        shadow_count=stats.get('shadow_count', 0),
        metrics=stats
    )


@ab_router.post("/{test_id}/update-split")
async def update_traffic_split(test_id: str, request: UpdateTrafficSplitRequest):
    """
    Update traffic split for a test (e.g., ramp from 5% to 10% to 50%).

    Example:
    ```
    {
        "traffic_split": 0.10
    }
    ```
    """
    if ab_manager is None:
        raise HTTPException(
            status_code=503,
            detail="A/B testing manager not initialized"
        )

    try:
        ab_manager.update_traffic_split(test_id, request.traffic_split)
        return {
            "test_id": test_id,
            "traffic_split": request.traffic_split,
            "status": "updated"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update traffic split: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@ab_router.post("/{test_id}/enable")
async def enable_test(test_id: str):
    """Enable an A/B test."""
    if ab_manager is None:
        raise HTTPException(
            status_code=503,
            detail="A/B testing manager not initialized"
        )

    try:
        ab_manager.enable_test(test_id)
        return {"test_id": test_id, "status": "enabled"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@ab_router.post("/{test_id}/disable")
async def disable_test(test_id: str):
    """Disable an A/B test (routes all traffic to control)."""
    if ab_manager is None:
        raise HTTPException(
            status_code=503,
            detail="A/B testing manager not initialized"
        )

    try:
        ab_manager.disable_test(test_id)
        return {"test_id": test_id, "status": "disabled"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@ab_router.post("/{test_id}/flush")
async def flush_test_results(test_id: str):
    """Flush buffered results for a test to storage."""
    if ab_manager is None:
        raise HTTPException(
            status_code=503,
            detail="A/B testing manager not initialized"
        )

    try:
        await ab_manager.flush_results()
        return {"test_id": test_id, "status": "flushed"}
    except Exception as e:
        logger.error(f"Failed to flush results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Helper function to integrate A/B testing into predict endpoint
async def predict_with_ab_test_if_enabled(
    request: PredictRequest,
    model_cache,
    ab_test_id: str = None
) -> tuple:
    """
    Make prediction with optional A/B testing.

    Returns:
        Tuple of (action, confidence, action_probs, ab_result)
        ab_result is None if A/B testing is disabled
    """
    if ab_test_id is None or ab_manager is None:
        # No A/B testing, use normal prediction flow
        return None

    # Get features (from the normal prediction flow)
    # This would be called from the main predict endpoint
    # For now, just returning None to indicate A/B testing not used

    return None


# Export for main app
__all__ = [
    'ab_router',
    'initialize_ab_manager',
    'initialize_ab_storage',
    'ab_manager'
]

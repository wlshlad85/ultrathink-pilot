"""
Inference API - Low-latency prediction service.
FastAPI application for real-time trading decisions.
"""
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict
import logging
import os

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import torch
import numpy as np

from models import (
    PredictRequest, PredictResponse, HealthResponse, ModelsResponse,
    ErrorResponse, RegimeProbabilities, StrategyWeights,
    RiskValidation, PredictionMetadata, ModelInfo
)
from model_loader import create_model_cache, ModelCache
from service_clients import ServiceClients
from kafka_producer import init_producer, get_producer, shutdown_producer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTIONS_TOTAL = Counter('predictions_total', 'Total number of predictions', ['status'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds',
                                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time taken to load models')

# Global state
model_cache: ModelCache = None
service_clients: ServiceClients = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global model_cache, service_clients

    logger.info("Starting Inference API...")

    # Load models at startup (warm cache)
    start_time = time.time()
    model_cache = create_model_cache()
    load_time = time.time() - start_time

    MODEL_LOAD_TIME.set(load_time)
    logger.info(f"Models loaded in {load_time:.2f}s")

    # Initialize service clients
    service_clients = ServiceClients()
    logger.info("Service clients initialized")

    # Initialize Kafka producer (async forensics)
    try:
        await init_producer()
        logger.info("Kafka producer initialized")
    except Exception as e:
        logger.warning(f"Kafka producer initialization failed (will buffer events): {e}")

    yield

    # Cleanup
    logger.info("Shutting down Inference API...")
    await shutdown_producer()


app = FastAPI(
    title="Trading Inference API",
    description="Low-latency prediction service for algorithmic trading",
    version="1.0.0",
    lifespan=lifespan
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": "Trading Inference API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    models_loaded = model_cache is not None and len(model_cache.models) > 0

    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        model_loaded=models_loaded,
        gpu_available=gpu_available
    )


@app.get("/api/v1/models", response_model=ModelsResponse, tags=["Models"])
async def list_models():
    """Get information about loaded models."""
    if model_cache is None:
        raise HTTPException(status_code=503, detail="Model cache not initialized")

    models_info = {}
    for name, info in model_cache.list_models().items():
        models_info[name] = ModelInfo(
            name=name,
            version=info['version'],
            loaded_at=info['loaded_at'],
            device=info['device'],
            parameters=info['parameters']
        )

    return ModelsResponse(
        models=models_info,
        total_models=len(models_info)
    )


def select_specialist_model(strategy_weights: Dict[str, float]) -> str:
    """
    Select which specialist model to use based on strategy weights.

    Args:
        strategy_weights: Dict with strategy weights

    Returns:
        Model name to use
    """
    # Use the specialist with highest weight
    max_specialist = max(strategy_weights.items(), key=lambda x: x[1])
    specialist_name = max_specialist[0]  # e.g., 'bull_specialist'

    # Check if model is loaded, fallback to universal
    if model_cache.is_loaded(specialist_name):
        return specialist_name
    elif model_cache.is_loaded('universal'):
        logger.warning(f"{specialist_name} not loaded, using universal")
        return 'universal'
    else:
        # Fallback to any loaded model
        available = list(model_cache.models.keys())
        if available:
            logger.warning(f"Using fallback model: {available[0]}")
            return available[0]
        else:
            raise ValueError("No models loaded")


def calculate_quantity(
    action: str,
    confidence: float,
    regime_probs: Dict[str, float]
) -> int:
    """
    Calculate recommended trade quantity.

    Simple heuristic:
    - Base quantity = 100 shares
    - Scale by confidence
    - Scale by regime certainty (inverse entropy)

    Args:
        action: Trading action
        confidence: Model confidence
        regime_probs: Regime probabilities

    Returns:
        Recommended quantity
    """
    if action == "HOLD":
        return 0

    base_quantity = 100
    confidence_factor = confidence
    certainty_factor = 1.0 - regime_probs['entropy']

    quantity = int(base_quantity * confidence_factor * certainty_factor)
    return max(1, quantity)  # At least 1 share


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Make trading prediction.

    This is the main inference endpoint with <50ms P95 latency target.
    """
    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        if model_cache is None:
            raise HTTPException(status_code=503, detail="Models not loaded")

        # Step 1: Get all context in parallel (features, regime, strategy weights)
        # Target: <20ms
        context_start = time.time()
        features, regime_probs, strategy_weights = await service_clients.get_all_context(
            symbol=request.symbol,
            timestamp=request.timestamp.isoformat()
        )
        context_time = time.time() - context_start
        logger.debug(f"Context fetch: {context_time*1000:.2f}ms")

        # Step 2: Select model and make prediction
        # Target: <15ms
        pred_start = time.time()

        # Allow strategy override
        if request.strategy_override:
            model_name = request.strategy_override
            if not model_cache.is_loaded(model_name):
                raise HTTPException(
                    status_code=400,
                    detail=f"Strategy override '{model_name}' not available"
                )
        else:
            model_name = select_specialist_model(strategy_weights)

        # Make prediction (GPU-accelerated)
        action, confidence, action_probs = model_cache.predict(model_name, features)

        pred_time = time.time() - pred_start
        logger.debug(f"Prediction: {pred_time*1000:.2f}ms")

        # Step 3: Calculate quantity
        quantity = calculate_quantity(action, confidence, regime_probs)

        # Step 4: Risk check (if requested)
        # Target: <10ms
        risk_validation = None
        if request.risk_check:
            risk_start = time.time()
            risk_result = await service_clients.risk_manager.check_risk(
                symbol=request.symbol,
                action=action,
                quantity=quantity
            )
            risk_time = time.time() - risk_start
            logger.debug(f"Risk check: {risk_time*1000:.2f}ms")

            risk_validation = RiskValidation(**risk_result)

            # Block trade if not approved
            if not risk_validation.approved:
                PREDICTIONS_TOTAL.labels(status='rejected').inc()
                raise HTTPException(
                    status_code=403,
                    detail="Trade rejected by risk manager",
                    headers={"X-Risk-Warnings": ",".join(risk_result.get('warnings', []))}
                )

        # Calculate total latency
        total_latency = time.time() - start_time
        latency_ms = total_latency * 1000

        # Build response
        response = PredictResponse(
            symbol=request.symbol,
            action=action,
            confidence=confidence,
            recommended_quantity=quantity,
            regime_probabilities=RegimeProbabilities(**regime_probs),
            strategy_weights=StrategyWeights(**strategy_weights),
            risk_validation=risk_validation,
            metadata=PredictionMetadata(
                model_version=model_name,
                latency_ms=latency_ms
            )
        )

        # Record metrics
        PREDICTIONS_TOTAL.labels(status='success').inc()
        PREDICTION_LATENCY.observe(total_latency)

        logger.info(
            f"Prediction: {request.symbol} -> {action} "
            f"(confidence={confidence:.3f}, latency={latency_ms:.1f}ms)"
        )

        # Wave 2: Emit to Kafka for forensics (async, non-blocking)
        producer = await get_producer()
        if producer:
            # Fire-and-forget async emission (target: <5ms overhead)
            asyncio.create_task(producer.emit_decision(response, features))

        return response

    except HTTPException:
        raise
    except Exception as e:
        PREDICTIONS_TOTAL.labels(status='error').inc()
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/metrics", tags=["Metrics"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    # Run with:
    # uvicorn inference_api:app --host 0.0.0.0 --port 8080 --workers 1
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=8080,
        workers=1,  # Single worker for GPU
        log_level="info"
    )

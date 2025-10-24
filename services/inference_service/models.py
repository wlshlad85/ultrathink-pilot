"""
Pydantic models for the Inference API.
Defines request and response schemas.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum
import uuid


class TradingAction(str, Enum):
    """Trading action enum."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class PredictRequest(BaseModel):
    """Request model for /api/v1/predict endpoint."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'AAPL', 'BTC-USD')")
    timestamp: Optional[datetime] = Field(default=None, description="Optional timestamp, defaults to now")
    strategy_override: Optional[str] = Field(default=None, description="Force specific strategy (bull_specialist, bear_specialist, sideways_specialist)")
    risk_check: bool = Field(default=True, description="Include risk validation")
    explain: bool = Field(default=False, description="Include model explanations (adds latency)")

    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate symbol format."""
        if not v or len(v) > 20:
            raise ValueError("Symbol must be non-empty and <= 20 characters")
        return v.upper()

    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        """Set default timestamp to now if not provided."""
        return v or datetime.utcnow()


class RegimeProbabilities(BaseModel):
    """Regime detection probabilities."""
    bull: float = Field(..., ge=0.0, le=1.0, description="Bull market probability")
    bear: float = Field(..., ge=0.0, le=1.0, description="Bear market probability")
    sideways: float = Field(..., ge=0.0, le=1.0, description="Sideways market probability")
    entropy: float = Field(..., ge=0.0, description="Uncertainty measure")

    @validator('entropy')
    def validate_probabilities(cls, v, values):
        """Validate probabilities sum to 1.0."""
        if 'bull' in values and 'bear' in values and 'sideways' in values:
            total = values['bull'] + values['bear'] + values['sideways']
            if not (0.99 <= total <= 1.01):  # Allow small floating point error
                raise ValueError(f"Regime probabilities must sum to 1.0, got {total}")
        return v


class StrategyWeights(BaseModel):
    """Strategy blending weights."""
    bull_specialist: float = Field(..., ge=0.0, le=1.0)
    bear_specialist: float = Field(..., ge=0.0, le=1.0)
    sideways_specialist: float = Field(..., ge=0.0, le=1.0)

    @validator('sideways_specialist')
    def validate_weights(cls, v, values):
        """Validate weights sum to 1.0."""
        if 'bull_specialist' in values and 'bear_specialist' in values:
            total = values['bull_specialist'] + values['bear_specialist'] + v
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"Strategy weights must sum to 1.0, got {total}")
        return v


class RiskCheck(BaseModel):
    """Risk validation result."""
    position_limit: str = Field(..., description="pass/fail/warning")
    concentration: str = Field(..., description="pass/fail/warning")
    daily_loss_limit: str = Field(..., description="pass/fail/warning")


class RiskValidation(BaseModel):
    """Risk validation details."""
    approved: bool = Field(..., description="Whether trade is approved")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    checks: RiskCheck = Field(..., description="Individual check results")


class PredictionMetadata(BaseModel):
    """Metadata about the prediction."""
    model_version: str = Field(..., description="Model version used for prediction")
    latency_ms: float = Field(..., ge=0, description="Prediction latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    ab_test: Optional['ABTestInfo'] = Field(default=None, description="A/B test information if applicable")


class PredictResponse(BaseModel):
    """Response model for /api/v1/predict endpoint."""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique decision ID")
    symbol: str = Field(..., description="Trading symbol")
    action: TradingAction = Field(..., description="Trading action (BUY, SELL, HOLD)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Action confidence")
    recommended_quantity: int = Field(..., ge=0, description="Recommended trade quantity")
    regime_probabilities: RegimeProbabilities = Field(..., description="Market regime probabilities")
    strategy_weights: StrategyWeights = Field(..., description="Strategy blending weights")
    risk_validation: Optional[RiskValidation] = Field(default=None, description="Risk validation results")
    metadata: PredictionMetadata = Field(..., description="Prediction metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether models are loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfo(BaseModel):
    """Information about a loaded model."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    loaded_at: datetime = Field(..., description="When model was loaded")
    device: str = Field(..., description="Device model is running on (cpu/cuda)")
    parameters: int = Field(..., description="Number of model parameters")


class ModelsResponse(BaseModel):
    """Response for /api/v1/models endpoint."""
    models: Dict[str, ModelInfo] = Field(..., description="Loaded models")
    total_models: int = Field(..., description="Total number of models")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# A/B Testing Models

class ABTestMode(str, Enum):
    """A/B test execution modes."""
    TRAFFIC_SPLIT = "traffic_split"
    SHADOW = "shadow"
    DISABLED = "disabled"


class CreateABTestRequest(BaseModel):
    """Request to create a new A/B test."""
    test_id: str = Field(..., description="Unique test identifier")
    control_model: str = Field(..., description="Control model name")
    treatment_model: str = Field(..., description="Treatment model name")
    traffic_split: float = Field(default=0.05, ge=0.0, le=1.0, description="Traffic to treatment (0.0-1.0)")
    mode: ABTestMode = Field(default=ABTestMode.TRAFFIC_SPLIT, description="Test mode")
    description: str = Field(default="", description="Test description")


class ABTestConfigResponse(BaseModel):
    """A/B test configuration."""
    test_id: str
    control_model: str
    treatment_model: str
    traffic_split: float
    mode: ABTestMode
    enabled: bool
    created_at: datetime
    description: str


class UpdateTrafficSplitRequest(BaseModel):
    """Request to update traffic split."""
    traffic_split: float = Field(..., ge=0.0, le=1.0, description="New traffic split")


class ABTestStatsResponse(BaseModel):
    """A/B test statistics."""
    test_id: str
    time_window_hours: int
    total_samples: int
    control_count: int
    treatment_count: int
    shadow_count: int
    metrics: Dict[str, Any]


class ABTestInfo(BaseModel):
    """Information about an A/B test in prediction metadata."""
    test_id: str = Field(..., description="A/B test identifier")
    assigned_group: str = Field(..., description="control, treatment, or shadow")
    shadow_mode: bool = Field(default=False, description="Whether test is in shadow mode")


# Update forward references
PredictionMetadata.model_rebuild()

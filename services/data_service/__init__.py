"""
Unified Data Service for UltraThink Pilot

Provides consolidated feature engineering, caching, and data access
for training and inference pipelines.
"""

from .feature_pipeline import FeaturePipeline
from .cache_layer import InMemoryCache, CachedFeaturePipeline

__version__ = "1.0.0"
__all__ = ["FeaturePipeline", "InMemoryCache", "CachedFeaturePipeline"]

"""
Online Learning Service

Implements Elastic Weight Consolidation (EWC) for incremental model updates
while preventing catastrophic forgetting.

Key Features:
- Conservative learning rates (1e-5 default)
- EWC regularization (lambda=1000)
- Stability monitoring with automatic rollback
- Sliding window data management (30-90 days)

Maintains <5% performance degradation vs. 15-25% decay with static models.
"""

from .ewc_trainer import EWCTrainer
from .stability_checker import StabilityChecker

__all__ = ['EWCTrainer', 'StabilityChecker']
__version__ = '1.0.0'

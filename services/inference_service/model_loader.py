"""
Model loader with warm cache and GPU acceleration.
Handles loading PyTorch models from file system and MLflow registry.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Copied from ppo_agent.py for inference.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network."""
        features = self.feature_extractor(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

    def predict(self, state: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        """
        Predict action (greedy) with confidence.

        Returns:
            (action, confidence, action_probs)
        """
        action_probs, _ = self.forward(state)
        action = torch.argmax(action_probs, dim=-1)
        confidence = action_probs[0, action.item()].item()
        return action.item(), confidence, action_probs


class ModelCache:
    """
    Warm model cache to avoid cold starts.
    Loads specialist models at startup and keeps them in memory.
    """

    def __init__(self, model_dir: str, device: Optional[str] = None):
        """
        Initialize model cache.

        Args:
            model_dir: Directory containing model checkpoints
            device: Device to load models on (cuda/cpu)
        """
        self.model_dir = Path(model_dir)

        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        logger.info(f"ModelCache using device: {self.device}")

        # Model specifications
        self.state_dim = 43  # From TradingEnv
        self.action_dim = 3  # HOLD, BUY, SELL
        self.hidden_dim = 256

        # Cache of loaded models
        self.models: Dict[str, Dict] = {}

        # Action mapping
        self.action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def load_model(self, model_name: str, model_path: str) -> bool:
        """
        Load a model into the cache.

        Args:
            model_name: Name to cache model under (e.g., 'bull_specialist')
            model_path: Path to model checkpoint file

        Returns:
            Success boolean
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            # Create model instance
            model = ActorCritic(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim
            ).to(self.device)

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if 'policy_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['policy_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            # Set to evaluation mode
            model.eval()

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())

            # Cache model info
            self.models[model_name] = {
                'model': model,
                'path': str(model_path),
                'loaded_at': datetime.utcnow(),
                'device': str(self.device),
                'parameters': num_params,
                'version': model_path.stem  # Use filename as version
            }

            logger.info(f"Loaded model '{model_name}' from {model_path}")
            logger.info(f"  Parameters: {num_params:,}")
            logger.info(f"  Device: {self.device}")

            return True

        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {e}")
            return False

    def load_all_specialists(self) -> int:
        """
        Load all specialist models from model directory.

        Returns:
            Number of models successfully loaded
        """
        loaded_count = 0

        # Define specialist models to load
        specialists = {
            'bull_specialist': 'bull_specialist/best_model.pth',
            'bear_specialist': 'bear_specialist/best_model.pth',
            'sideways_specialist': 'sideways_specialist/best_model.pth',
            'universal': 'best_model.pth'  # Fallback universal model
        }

        for name, rel_path in specialists.items():
            model_path = self.model_dir / rel_path
            if model_path.exists():
                if self.load_model(name, str(model_path)):
                    loaded_count += 1
            else:
                logger.warning(f"Specialist model not found: {model_path}")

        logger.info(f"Loaded {loaded_count} specialist models")
        return loaded_count

    @torch.no_grad()
    def predict(
        self,
        model_name: str,
        state: np.ndarray
    ) -> Tuple[str, float, np.ndarray]:
        """
        Make prediction using cached model.

        Args:
            model_name: Name of model to use
            state: State observation (numpy array)

        Returns:
            (action_str, confidence, action_probs)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded in cache")

        model_info = self.models[model_name]
        model = model_info['model']

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get prediction
        action_idx, confidence, action_probs = model.predict(state_tensor)

        # Convert to action string
        action_str = self.action_map[action_idx]

        # Convert probs to numpy
        probs_np = action_probs.cpu().numpy().flatten()

        return action_str, confidence, probs_np

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a loaded model."""
        if model_name not in self.models:
            return None

        info = self.models[model_name].copy()
        info.pop('model')  # Don't include model object
        return info

    def list_models(self) -> Dict[str, Dict]:
        """List all loaded models with their info."""
        return {
            name: self.get_model_info(name)
            for name in self.models.keys()
        }

    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self.models

    def get_device(self) -> str:
        """Get device being used."""
        return str(self.device)


def create_model_cache(model_dir: Optional[str] = None) -> ModelCache:
    """
    Factory function to create and initialize model cache.

    Args:
        model_dir: Directory containing models (defaults to env var or ../rl/models)

    Returns:
        Initialized ModelCache
    """
    if model_dir is None:
        # Try environment variable first
        model_dir = os.getenv('MODEL_DIR', '/app/models')

        # If env var points to nonexistent dir, try default
        if not Path(model_dir).exists():
            default_dir = Path(__file__).parent.parent.parent / 'rl' / 'models'
            if default_dir.exists():
                model_dir = str(default_dir)
            else:
                logger.warning(f"Model directory not found: {model_dir}")

    cache = ModelCache(model_dir)
    loaded = cache.load_all_specialists()

    if loaded == 0:
        logger.warning("No models loaded! Inference will fail.")

    return cache


if __name__ == "__main__":
    # Test model loading
    print("Testing ModelCache...")

    cache = create_model_cache()

    print(f"\nLoaded models: {list(cache.models.keys())}")
    print(f"Device: {cache.get_device()}")

    # Test prediction with dummy state
    if cache.models:
        dummy_state = np.random.randn(43)
        model_name = list(cache.models.keys())[0]

        print(f"\nTesting prediction with '{model_name}'...")
        action, confidence, probs = cache.predict(model_name, dummy_state)

        print(f"Action: {action}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Action probs: {probs}")

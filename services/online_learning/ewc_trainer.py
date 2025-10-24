#!/usr/bin/env python3
"""
Elastic Weight Consolidation (EWC) Trainer

Implements incremental model updates with catastrophic forgetting prevention.
Based on "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017).

Key Features:
- Fisher Information Matrix computation for weight importance
- EWC regularization to preserve critical weights
- Conservative learning rates (1e-5 default)
- Sliding window data management (30-90 days)
- Checkpoint management with rollback capability

Performance Target: <5% degradation over 30 days
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EWCConfig:
    """Configuration for EWC training."""
    learning_rate: float = 1e-5  # Very conservative learning rate
    ewc_lambda: float = 1000.0  # Strong regularization strength
    window_size_days: int = 60  # Sliding window size (30-90 days)
    update_frequency: str = "daily"  # Daily updates (not hourly)
    batch_size: int = 64
    epochs_per_update: int = 10
    fisher_sample_size: int = 1000  # Samples for Fisher computation

    # Checkpoint settings
    keep_last_n_checkpoints: int = 5
    checkpoint_dir: str = "/home/rich/ultrathink-pilot/rl/models/online_learning"

    # Safety settings
    max_weight_change_percent: float = 10.0  # Alert if weights change >10%
    gradient_clip_norm: float = 1.0


@dataclass
class EWCState:
    """State for EWC regularization."""
    fisher_information: Dict[str, torch.Tensor] = field(default_factory=dict)
    optimal_params: Dict[str, torch.Tensor] = field(default_factory=dict)
    update_count: int = 0
    last_update_time: Optional[datetime] = None


class EWCTrainer:
    """
    Elastic Weight Consolidation Trainer for incremental learning.

    Prevents catastrophic forgetting by:
    1. Computing Fisher Information Matrix after initial training
    2. Adding EWC regularization loss during incremental updates
    3. Preserving important weights while adapting to new data

    Usage:
        trainer = EWCTrainer(model, config)
        trainer.compute_fisher_information(dataloader)
        trainer.incremental_update(new_dataloader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[EWCConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize EWC Trainer.

        Args:
            model: PyTorch model (ActorCritic from PPO)
            config: EWC configuration
            device: Device to use ('cuda' or 'cpu')
        """
        self.config = config or EWCConfig()

        # Device setup
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        logger.info(f"EWC Trainer using device: {self.device}")

        # Model setup
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        # EWC state
        self.ewc_state = EWCState()

        # Checkpoint management
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training metrics
        self.training_history: List[Dict] = []

        logger.info(f"EWC Trainer initialized with lambda={self.config.ewc_lambda}, "
                   f"lr={self.config.learning_rate}")

    def compute_fisher_information(
        self,
        dataloader: DataLoader,
        loss_fn: Optional[callable] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix for current model.

        The Fisher Information measures the sensitivity of the model's output
        to changes in each parameter. High Fisher values indicate important
        weights that should be preserved.

        Algorithm:
        1. For each sample: compute gradients of log-likelihood
        2. Square the gradients (Fisher ≈ E[grad²])
        3. Average over samples

        Args:
            dataloader: DataLoader with training data
            loss_fn: Loss function (if None, uses default)

        Returns:
            Dictionary of Fisher information per parameter
        """
        logger.info("Computing Fisher Information Matrix...")

        self.model.eval()
        fisher_dict = {}

        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)

        # Sample subset for Fisher computation (for efficiency)
        sample_count = 0
        max_samples = self.config.fisher_sample_size

        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= max_samples:
                break

            # Extract states and actions from batch
            if len(batch) == 2:
                states, actions = batch
            else:
                states = batch[0]
                actions = batch[1] if len(batch) > 1 else None

            states = states.to(self.device)

            # Forward pass
            if actions is not None:
                actions = actions.to(self.device)
                log_probs, _, _ = self.model.evaluate(states, actions)
                loss = -log_probs.mean()  # Negative log-likelihood
            else:
                # For state-only data, use policy entropy
                action_probs, _ = self.model(states)
                from torch.distributions import Categorical
                dist = Categorical(action_probs)
                loss = -dist.entropy().mean()

            # Backward pass to get gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Accumulate squared gradients (Fisher approximation)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.data.pow(2)

            sample_count += states.size(0)

        # Normalize by number of samples
        for name in fisher_dict:
            fisher_dict[name] /= sample_count

        # Store Fisher information and current optimal parameters
        self.ewc_state.fisher_information = fisher_dict
        self.ewc_state.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        logger.info(f"Fisher Information Matrix computed from {sample_count} samples")

        # Log Fisher statistics
        fisher_values = torch.cat([f.flatten() for f in fisher_dict.values()])
        logger.info(f"Fisher statistics - mean: {fisher_values.mean():.6f}, "
                   f"std: {fisher_values.std():.6f}, "
                   f"max: {fisher_values.max():.6f}")

        return fisher_dict

    def ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        L_EWC = (lambda/2) * sum_i F_i * (theta_i - theta*_i)²

        Where:
        - F_i: Fisher information for parameter i
        - theta_i: Current parameter value
        - theta*_i: Optimal parameter value from previous task
        - lambda: Regularization strength

        Returns:
            EWC regularization loss
        """
        if not self.ewc_state.fisher_information:
            return torch.tensor(0.0, device=self.device)

        ewc_loss = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if name in self.ewc_state.fisher_information:
                fisher = self.ewc_state.fisher_information[name]
                optimal = self.ewc_state.optimal_params[name]

                # EWC penalty: Fisher * (current - optimal)²
                ewc_loss += (fisher * (param - optimal).pow(2)).sum()

        return (self.config.ewc_lambda / 2.0) * ewc_loss

    def incremental_update(
        self,
        dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """
        Perform incremental update with EWC regularization.

        Training loop:
        1. Compute standard task loss (e.g., policy gradient)
        2. Add EWC regularization loss
        3. Backpropagate combined loss
        4. Update model parameters

        Args:
            dataloader: Training data loader
            validation_dataloader: Optional validation data

        Returns:
            Training metrics dictionary
        """
        logger.info(f"Starting incremental update #{self.ewc_state.update_count + 1}")

        self.model.train()

        # Training metrics
        total_loss = 0.0
        total_task_loss = 0.0
        total_ewc_loss = 0.0
        batch_count = 0

        # Training loop
        for epoch in range(self.config.epochs_per_update):
            epoch_loss = 0.0
            epoch_task_loss = 0.0
            epoch_ewc_loss = 0.0

            for batch_idx, batch in enumerate(dataloader):
                # Extract data
                if len(batch) == 2:
                    states, actions = batch
                elif len(batch) >= 3:
                    states, actions, rewards = batch[:3]
                else:
                    states = batch[0]
                    actions = None

                states = states.to(self.device)

                # Task-specific loss (policy gradient or supervised)
                if actions is not None:
                    actions = actions.to(self.device)
                    log_probs, state_values, dist_entropy = self.model.evaluate(states, actions)

                    # Simple policy gradient loss
                    if len(batch) >= 3:
                        rewards = rewards.to(self.device)
                        advantages = rewards - state_values.detach().squeeze()
                        task_loss = -(log_probs * advantages).mean()
                    else:
                        task_loss = -log_probs.mean()

                    task_loss += 0.01 * dist_entropy.mean()  # Entropy bonus
                else:
                    # Unsupervised: maximize entropy
                    action_probs, _ = self.model(states)
                    from torch.distributions import Categorical
                    dist = Categorical(action_probs)
                    task_loss = -dist.entropy().mean()

                # EWC regularization loss
                ewc_reg_loss = self.ewc_loss()

                # Combined loss
                combined_loss = task_loss + ewc_reg_loss

                # Backward pass
                self.optimizer.zero_grad()
                combined_loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )

                self.optimizer.step()

                # Track metrics
                epoch_loss += combined_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_ewc_loss += ewc_reg_loss.item()
                batch_count += 1

            # Epoch summary
            avg_epoch_loss = epoch_loss / len(dataloader)
            avg_task_loss = epoch_task_loss / len(dataloader)
            avg_ewc_loss = epoch_ewc_loss / len(dataloader)

            total_loss += epoch_loss
            total_task_loss += epoch_task_loss
            total_ewc_loss += epoch_ewc_loss

            if (epoch + 1) % 2 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.epochs_per_update}: "
                           f"Loss={avg_epoch_loss:.4f} "
                           f"(Task={avg_task_loss:.4f}, EWC={avg_ewc_loss:.4f})")

        # Final metrics
        metrics = {
            'total_loss': total_loss / batch_count,
            'task_loss': total_task_loss / batch_count,
            'ewc_loss': total_ewc_loss / batch_count,
            'update_count': self.ewc_state.update_count + 1,
            'learning_rate': self.config.learning_rate,
            'ewc_lambda': self.config.ewc_lambda
        }

        # Validation metrics if provided
        if validation_dataloader:
            val_metrics = self._validate(validation_dataloader)
            metrics.update({'val_' + k: v for k, v in val_metrics.items()})

        # Update state
        self.ewc_state.update_count += 1
        self.ewc_state.last_update_time = datetime.now()

        # Store training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            **metrics
        })

        logger.info(f"Incremental update completed. Metrics: {metrics}")

        return metrics

    def _validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation and return metrics."""
        self.model.eval()

        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch in dataloader:
                states = batch[0].to(self.device)

                if len(batch) >= 2:
                    actions = batch[1].to(self.device)
                    log_probs, _, _ = self.model.evaluate(states, actions)
                    loss = -log_probs.mean()
                else:
                    action_probs, _ = self.model(states)
                    from torch.distributions import Categorical
                    dist = Categorical(action_probs)
                    loss = -dist.entropy().mean()

                total_loss += loss.item()
                batch_count += 1

        return {
            'loss': total_loss / batch_count if batch_count > 0 else 0.0
        }

    def save_checkpoint(
        self,
        filepath: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save model checkpoint with EWC state.

        Args:
            filepath: Path to save checkpoint (auto-generated if None)
            metadata: Additional metadata to save

        Returns:
            Path to saved checkpoint
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.checkpoint_dir / f"ewc_checkpoint_{timestamp}.pth"

        filepath = Path(filepath)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fisher_information': self.ewc_state.fisher_information,
            'optimal_params': self.ewc_state.optimal_params,
            'update_count': self.ewc_state.update_count,
            'last_update_time': self.ewc_state.last_update_time.isoformat()
                if self.ewc_state.last_update_time else None,
            'config': {
                'learning_rate': self.config.learning_rate,
                'ewc_lambda': self.config.ewc_lambda,
                'window_size_days': self.config.window_size_days,
            },
            'training_history': self.training_history,
            'metadata': metadata or {}
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return str(filepath)

    def load_checkpoint(self, filepath: str):
        """
        Load checkpoint with EWC state.

        Args:
            filepath: Path to checkpoint file
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore EWC state
        self.ewc_state.fisher_information = checkpoint['fisher_information']
        self.ewc_state.optimal_params = checkpoint['optimal_params']
        self.ewc_state.update_count = checkpoint.get('update_count', 0)

        last_update = checkpoint.get('last_update_time')
        if last_update:
            self.ewc_state.last_update_time = datetime.fromisoformat(last_update)

        # Restore training history
        self.training_history = checkpoint.get('training_history', [])

        logger.info(f"Checkpoint loaded from {filepath}")
        logger.info(f"Update count: {self.ewc_state.update_count}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("ewc_checkpoint_*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Keep only the most recent checkpoints
        for checkpoint in checkpoints[self.config.keep_last_n_checkpoints:]:
            checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {checkpoint}")

    def get_weight_change_statistics(self) -> Dict[str, float]:
        """
        Compute statistics about weight changes since last Fisher computation.

        Returns:
            Dictionary with weight change metrics
        """
        if not self.ewc_state.optimal_params:
            return {}

        total_params = 0
        total_change = 0.0
        max_change = 0.0

        for name, param in self.model.named_parameters():
            if name in self.ewc_state.optimal_params:
                optimal = self.ewc_state.optimal_params[name]
                change = (param.data - optimal).abs()

                total_params += param.numel()
                total_change += change.sum().item()
                max_change = max(max_change, change.max().item())

        avg_change = total_change / total_params if total_params > 0 else 0.0

        return {
            'average_weight_change': avg_change,
            'max_weight_change': max_change,
            'total_parameters': total_params
        }

    def reset_ewc_state(self):
        """Reset EWC state (useful for starting fresh)."""
        self.ewc_state = EWCState()
        logger.info("EWC state reset")


if __name__ == "__main__":
    """Test EWC Trainer"""
    print("Testing EWC Trainer...")

    # Create dummy model
    from torch import nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 3)

        def forward(self, x):
            action_probs = torch.softmax(self.fc2(torch.relu(self.fc1(x))), dim=-1)
            state_value = torch.zeros(x.size(0), 1)
            return action_probs, state_value

        def evaluate(self, states, actions):
            action_probs, _ = self.forward(states)
            from torch.distributions import Categorical
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            return log_probs, torch.zeros_like(log_probs).unsqueeze(1), dist.entropy()

    model = DummyModel()
    trainer = EWCTrainer(model)

    # Create dummy data
    states = torch.randn(100, 10)
    actions = torch.randint(0, 3, (100,))
    dataset = TensorDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=32)

    # Compute Fisher information
    fisher = trainer.compute_fisher_information(dataloader)
    print(f"Fisher information computed for {len(fisher)} parameters")

    # Perform incremental update
    metrics = trainer.incremental_update(dataloader)
    print(f"Update metrics: {metrics}")

    # Save checkpoint
    checkpoint_path = trainer.save_checkpoint()
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Get weight change statistics
    stats = trainer.get_weight_change_statistics()
    print(f"Weight change statistics: {stats}")

    print("\nEWC Trainer test completed successfully!")

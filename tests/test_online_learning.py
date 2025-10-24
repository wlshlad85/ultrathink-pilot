#!/usr/bin/env python3
"""
Unit and Integration Tests for Online Learning Service

Tests:
- EWC Trainer functionality
- Stability Checker accuracy
- Data Manager sliding window
- API endpoints
- End-to-end update flow
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import tempfile
import shutil

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "online_learning"))

from ewc_trainer import EWCTrainer, EWCConfig
from stability_checker import StabilityChecker, PerformanceMetrics
from data_manager import SlidingWindowDataManager


# Test Fixtures
@pytest.fixture
def dummy_model():
    """Create a simple test model."""
    class TestModel(nn.Module):
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

    return TestModel()


@pytest.fixture
def dummy_dataloader():
    """Create test data loader."""
    states = torch.randn(100, 10)
    actions = torch.randint(0, 3, (100,))
    rewards = torch.randn(100)
    dataset = TensorDataset(states, actions, rewards)
    return DataLoader(dataset, batch_size=32)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# EWC Trainer Tests
class TestEWCTrainer:
    """Test EWC Trainer functionality."""

    def test_initialization(self, dummy_model, temp_checkpoint_dir):
        """Test trainer initialization."""
        config = EWCConfig(checkpoint_dir=temp_checkpoint_dir)
        trainer = EWCTrainer(dummy_model, config)

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.config.learning_rate == 1e-5
        assert trainer.config.ewc_lambda == 1000.0

    def test_fisher_computation(self, dummy_model, dummy_dataloader, temp_checkpoint_dir):
        """Test Fisher Information Matrix computation."""
        config = EWCConfig(checkpoint_dir=temp_checkpoint_dir, fisher_sample_size=50)
        trainer = EWCTrainer(dummy_model, config)

        fisher = trainer.compute_fisher_information(dummy_dataloader)

        # Check Fisher information was computed
        assert len(fisher) > 0
        assert all(isinstance(f, torch.Tensor) for f in fisher.values())

        # Check optimal params were saved
        assert len(trainer.ewc_state.optimal_params) > 0

    def test_ewc_loss(self, dummy_model, dummy_dataloader, temp_checkpoint_dir):
        """Test EWC regularization loss."""
        config = EWCConfig(checkpoint_dir=temp_checkpoint_dir)
        trainer = EWCTrainer(dummy_model, config)

        # Before Fisher computation, EWC loss should be zero
        ewc_loss_before = trainer.ewc_loss()
        assert ewc_loss_before.item() == 0.0

        # After Fisher computation, EWC loss should be non-zero
        trainer.compute_fisher_information(dummy_dataloader)

        # Modify model parameters slightly
        for param in trainer.model.parameters():
            param.data += torch.randn_like(param.data) * 0.01

        ewc_loss_after = trainer.ewc_loss()
        assert ewc_loss_after.item() > 0.0

    def test_incremental_update(self, dummy_model, dummy_dataloader, temp_checkpoint_dir):
        """Test incremental update with EWC."""
        config = EWCConfig(
            checkpoint_dir=temp_checkpoint_dir,
            epochs_per_update=2
        )
        trainer = EWCTrainer(dummy_model, config)

        # Compute Fisher first
        trainer.compute_fisher_information(dummy_dataloader)

        # Perform update
        metrics = trainer.incremental_update(dummy_dataloader)

        # Check metrics
        assert 'total_loss' in metrics
        assert 'task_loss' in metrics
        assert 'ewc_loss' in metrics
        assert metrics['update_count'] == 1

    def test_checkpoint_save_load(self, dummy_model, dummy_dataloader, temp_checkpoint_dir):
        """Test checkpoint saving and loading."""
        config = EWCConfig(checkpoint_dir=temp_checkpoint_dir)
        trainer = EWCTrainer(dummy_model, config)

        # Compute Fisher and perform update
        trainer.compute_fisher_information(dummy_dataloader)
        trainer.incremental_update(dummy_dataloader)

        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint()
        assert Path(checkpoint_path).exists()

        # Create new trainer and load checkpoint
        new_trainer = EWCTrainer(dummy_model, config)
        new_trainer.load_checkpoint(checkpoint_path)

        # Check state was restored
        assert new_trainer.ewc_state.update_count == trainer.ewc_state.update_count
        assert len(new_trainer.ewc_state.fisher_information) > 0

    def test_weight_change_statistics(self, dummy_model, dummy_dataloader, temp_checkpoint_dir):
        """Test weight change statistics."""
        config = EWCConfig(checkpoint_dir=temp_checkpoint_dir)
        trainer = EWCTrainer(dummy_model, config)

        # Compute Fisher
        trainer.compute_fisher_information(dummy_dataloader)

        # Get initial stats (should be zero)
        stats_before = trainer.get_weight_change_statistics()
        assert stats_before['average_weight_change'] == 0.0

        # Modify model
        for param in trainer.model.parameters():
            param.data += torch.randn_like(param.data) * 0.1

        # Get stats after modification
        stats_after = trainer.get_weight_change_statistics()
        assert stats_after['average_weight_change'] > 0.0
        assert stats_after['max_weight_change'] > 0.0


# Stability Checker Tests
class TestStabilityChecker:
    """Test Stability Checker functionality."""

    def test_initialization(self):
        """Test stability checker initialization."""
        checker = StabilityChecker(
            sharpe_threshold=0.30,
            win_rate_threshold=0.40
        )

        assert checker.sharpe_threshold == 0.30
        assert checker.win_rate_threshold == 0.40

    def test_performance_metrics_from_returns(self):
        """Test performance metrics computation."""
        np.random.seed(42)
        returns = np.random.normal(0.01, 0.02, 100)

        metrics = PerformanceMetrics.from_returns(returns)

        assert metrics.sharpe_ratio > 0
        assert metrics.total_return != 0
        assert metrics.volatility > 0
        assert 0 <= metrics.win_rate <= 100

    def test_stability_check_stable(self):
        """Test stability check with stable performance."""
        checker = StabilityChecker()

        # Create similar performance before/after
        returns_before = np.random.normal(0.01, 0.02, 100)
        returns_after = np.random.normal(0.009, 0.021, 100)

        metrics_before = PerformanceMetrics.from_returns(returns_before)
        metrics_after = PerformanceMetrics.from_returns(returns_after)

        result = checker.check_stability(metrics_before, metrics_after)

        # Should be stable (degradation < 30%)
        assert result.should_rollback == False
        assert result.degradation_percent < 30

    def test_stability_check_rollback_required(self):
        """Test stability check triggering rollback."""
        checker = StabilityChecker(sharpe_threshold=0.30)

        # Create severe degradation
        returns_before = np.random.normal(0.01, 0.02, 100)
        returns_after = np.random.normal(0.002, 0.04, 100)  # Much worse

        metrics_before = PerformanceMetrics.from_returns(returns_before)
        metrics_after = PerformanceMetrics.from_returns(returns_after)

        result = checker.check_stability(metrics_before, metrics_after)

        # Should trigger rollback
        assert result.should_rollback == True
        assert result.degradation_percent >= 30

    def test_performance_trend(self):
        """Test performance trend analysis."""
        checker = StabilityChecker()

        # Add multiple performance records
        for i in range(10):
            returns = np.random.normal(0.01 + i * 0.001, 0.02, 50)
            metrics = PerformanceMetrics.from_returns(returns)
            checker.performance_history.append(metrics)

        trend = checker.get_performance_trend(window_size=5)

        assert 'avg_sharpe' in trend
        assert 'sharpe_trend' in trend
        assert trend['checks_count'] == 5

    def test_save_load_history(self, temp_checkpoint_dir):
        """Test history saving and loading."""
        checker = StabilityChecker()

        # Add some data
        returns = np.random.normal(0.01, 0.02, 100)
        metrics = PerformanceMetrics.from_returns(returns)
        checker.performance_history.append(metrics)

        # Save history
        history_path = Path(temp_checkpoint_dir) / "history.json"
        checker.save_history(str(history_path))
        assert history_path.exists()

        # Load history
        new_checker = StabilityChecker()
        new_checker.load_history(str(history_path))

        assert len(new_checker.performance_history) == 1


# Data Manager Tests
class TestDataManager:
    """Test Data Manager functionality."""

    def test_initialization(self, temp_checkpoint_dir):
        """Test data manager initialization."""
        manager = SlidingWindowDataManager(data_dir=temp_checkpoint_dir)

        assert manager.data_dir == Path(temp_checkpoint_dir)
        assert manager.cache_dir.exists()

    def test_get_data_loaders(self, temp_checkpoint_dir):
        """Test data loader creation."""
        manager = SlidingWindowDataManager(data_dir=temp_checkpoint_dir)

        train_loader, val_loader = manager.get_data_loaders(
            window_days=30,
            batch_size=32
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0

        # Check batch shape
        for batch in train_loader:
            states, actions, rewards = batch
            assert states.shape[1] == 43  # Expected state dimension
            break

    def test_sliding_window_sizes(self, temp_checkpoint_dir):
        """Test different window sizes."""
        manager = SlidingWindowDataManager(data_dir=temp_checkpoint_dir)

        for window_days in [30, 60, 90]:
            train_loader, val_loader = manager.get_data_loaders(
                window_days=window_days,
                batch_size=64
            )

            assert len(train_loader) > 0
            assert len(val_loader) > 0

    def test_get_test_data(self, temp_checkpoint_dir):
        """Test test data retrieval."""
        manager = SlidingWindowDataManager(data_dir=temp_checkpoint_dir)

        test_data = manager.get_test_data()

        assert len(test_data) > 0
        assert 'timestamp' in test_data.columns or 'date' in test_data.columns


# Integration Tests
class TestOnlineLearningIntegration:
    """Integration tests for full online learning workflow."""

    def test_end_to_end_update(self, dummy_model, temp_checkpoint_dir):
        """Test complete update workflow."""
        # Setup
        config = EWCConfig(
            checkpoint_dir=temp_checkpoint_dir,
            epochs_per_update=2
        )
        trainer = EWCTrainer(dummy_model, config)
        checker = StabilityChecker()
        manager = SlidingWindowDataManager(data_dir=temp_checkpoint_dir)

        # Get data
        train_loader, val_loader = manager.get_data_loaders(window_days=30)

        # Compute Fisher (initial)
        trainer.compute_fisher_information(train_loader)

        # Perform update
        metrics = trainer.incremental_update(train_loader, val_loader)

        assert metrics['update_count'] == 1
        assert 'total_loss' in metrics

    def test_rollback_on_degradation(self, dummy_model, temp_checkpoint_dir):
        """Test automatic rollback on performance degradation."""
        # This would require a full environment setup
        # Placeholder for integration test
        pass


# Performance Tests
class TestPerformance:
    """Performance tests for online learning."""

    def test_update_latency(self, dummy_model, dummy_dataloader, temp_checkpoint_dir):
        """Test update latency is reasonable."""
        import time

        config = EWCConfig(
            checkpoint_dir=temp_checkpoint_dir,
            epochs_per_update=5
        )
        trainer = EWCTrainer(dummy_model, config)

        # Compute Fisher
        trainer.compute_fisher_information(dummy_dataloader)

        # Measure update time
        start_time = time.time()
        trainer.incremental_update(dummy_dataloader)
        update_time = time.time() - start_time

        # Should complete in reasonable time (<60 seconds for small model)
        assert update_time < 60.0

    def test_memory_usage(self, dummy_model, temp_checkpoint_dir):
        """Test memory usage doesn't grow unbounded."""
        config = EWCConfig(checkpoint_dir=temp_checkpoint_dir)
        trainer = EWCTrainer(dummy_model, config)

        # Create data
        states = torch.randn(1000, 10)
        actions = torch.randint(0, 3, (1000,))
        dataset = TensorDataset(states, actions)
        dataloader = DataLoader(dataset, batch_size=32)

        # Multiple updates shouldn't cause memory leak
        for _ in range(5):
            trainer.incremental_update(dataloader)

        # Check training history doesn't grow unbounded
        assert len(trainer.training_history) <= 10  # Should cleanup old entries


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

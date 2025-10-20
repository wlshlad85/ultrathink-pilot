#!/usr/bin/env python3
"""
Comprehensive tests for RL system.
Tests trading environment, PPO agent, training and evaluation.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.trading_env import TradingEnv
from rl.ppo_agent import PPOAgent, ActorCritic


class TestTradingEnv:
    """Test TradingEnv class."""

    def test_initialization(self):
        """Test environment initialization."""
        env = TradingEnv(
            symbol="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-03-01",
            initial_capital=100000.0
        )

        assert env.symbol == "BTC-USD"
        assert env.initial_capital == 100000.0
        assert env.action_space.n == 3  # HOLD, BUY, SELL
        assert env.observation_space.shape[0] == 43  # 3 + 10 + 30

    def test_reset(self):
        """Test environment reset."""
        env = TradingEnv(
            symbol="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-03-01"
        )

        obs, info = env.reset()

        assert obs.shape == (43,)
        assert isinstance(info, dict)
        assert 'date' in info
        assert 'portfolio_value' in info

    def test_step_hold(self):
        """Test HOLD action."""
        env = TradingEnv(
            symbol="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-03-01"
        )

        obs, info = env.reset()
        initial_value = info['portfolio_value']

        obs, reward, terminated, truncated, info = env.step(0)  # HOLD

        assert obs.shape == (43,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info['portfolio_value'] == initial_value  # Should stay same

    def test_step_buy(self):
        """Test BUY action."""
        env = TradingEnv(
            symbol="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-03-01"
        )

        obs, info = env.reset()
        initial_cash = info['cash']

        obs, reward, terminated, truncated, info = env.step(1)  # BUY

        assert info['cash'] < initial_cash  # Cash should decrease
        assert info['position_quantity'] > 0  # Should have position

    def test_step_sell(self):
        """Test SELL action."""
        env = TradingEnv(
            symbol="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-03-01"
        )

        env.reset()

        # Buy first
        env.step(1)

        # Then sell
        obs, reward, terminated, truncated, info = env.step(2)  # SELL

        assert info['position_quantity'] == 0  # Should have no position

    def test_episode_completion(self):
        """Test running full episode."""
        env = TradingEnv(
            symbol="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-06-01"
        )

        obs, info = env.reset()
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            if terminated or truncated:
                break

        assert steps > 0
        assert terminated or truncated

    def test_observation_bounds(self):
        """Test observation values are valid."""
        env = TradingEnv(
            symbol="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-06-01"
        )

        obs, info = env.reset()

        # Check no NaN or inf
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()

        # Run a few steps
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)
            assert not np.isnan(obs).any()
            assert not np.isinf(obs).any()

            if terminated or truncated:
                break

    def test_reward_calculation(self):
        """Test reward is calculated."""
        env = TradingEnv(
            symbol="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-06-01"
        )

        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)


class TestActorCritic:
    """Test ActorCritic network."""

    def test_initialization(self):
        """Test network initialization."""
        network = ActorCritic(state_dim=43, action_dim=3)

        assert isinstance(network, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass."""
        network = ActorCritic(state_dim=43, action_dim=3)
        state = torch.randn(1, 43)

        action_probs, state_value = network(state)

        assert action_probs.shape == (1, 3)
        assert state_value.shape == (1, 1)
        assert torch.allclose(action_probs.sum(dim=1), torch.ones(1))  # Probs sum to 1

    def test_act_method(self):
        """Test action sampling."""
        network = ActorCritic(state_dim=43, action_dim=3)
        state = torch.randn(1, 43)

        action, log_prob, state_value = network.act(state)

        assert isinstance(action, int)
        assert 0 <= action < 3
        assert log_prob.shape == () or log_prob.shape == (1,)  # Allow both scalar and 1-D
        assert state_value.shape == (1, 1)

    def test_evaluate_method(self):
        """Test action evaluation."""
        network = ActorCritic(state_dim=43, action_dim=3)
        states = torch.randn(10, 43)
        actions = torch.randint(0, 3, (10,))

        log_probs, state_values, dist_entropy = network.evaluate(states, actions)

        assert log_probs.shape == (10,)
        assert state_values.shape == (10, 1)
        assert dist_entropy.shape == (10,)


class TestPPOAgent:
    """Test PPOAgent class."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = PPOAgent(state_dim=43, action_dim=3, device='cpu')

        assert agent.device.type == 'cpu'
        assert isinstance(agent.policy, ActorCritic)
        assert isinstance(agent.policy_old, ActorCritic)

    def test_cuda_detection(self):
        """Test CUDA is detected if available."""
        agent = PPOAgent(state_dim=43, action_dim=3)

        if torch.cuda.is_available():
            assert agent.device.type == 'cuda'
        else:
            assert agent.device.type == 'cpu'

    def test_select_action(self):
        """Test action selection."""
        agent = PPOAgent(state_dim=43, action_dim=3, device='cpu')
        state = np.random.randn(43)

        action = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < 3

    def test_store_reward_and_terminal(self):
        """Test storing rewards."""
        agent = PPOAgent(state_dim=43, action_dim=3, device='cpu')

        agent.store_reward_and_terminal(1.0, False)
        agent.store_reward_and_terminal(0.5, True)

        assert len(agent.rewards) == 2
        assert len(agent.is_terminals) == 2

    def test_update_policy(self):
        """Test policy update."""
        agent = PPOAgent(state_dim=43, action_dim=3, device='cpu')

        # Collect some dummy experience
        for _ in range(10):
            state = np.random.randn(43)
            action = agent.select_action(state)
            agent.store_reward_and_terminal(np.random.randn(), False)

        # Update policy
        metrics = agent.update()

        assert 'loss' in metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics

        # Memory should be cleared after update
        assert len(agent.states) == 0
        assert len(agent.actions) == 0

    def test_save_and_load(self, tmp_path):
        """Test model save and load."""
        agent = PPOAgent(state_dim=43, action_dim=3, device='cpu')

        # Get initial output
        state = torch.randn(1, 43)
        with torch.no_grad():
            initial_probs, _ = agent.policy(state)

        # Save model
        model_path = tmp_path / "test_model.pth"
        agent.save(str(model_path))

        # Create new agent and load
        new_agent = PPOAgent(state_dim=43, action_dim=3, device='cpu')
        new_agent.load(str(model_path))

        # Check outputs match
        with torch.no_grad():
            loaded_probs, _ = new_agent.policy(state)

        assert torch.allclose(initial_probs, loaded_probs)


class TestRLIntegration:
    """Test RL environment and agent together."""

    def test_agent_environment_interaction(self):
        """Test agent can interact with environment."""
        env = TradingEnv(
            symbol="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-06-01"
        )

        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device='cpu'
        )

        obs, info = env.reset()

        # Run 10 steps
        for _ in range(10):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.store_reward_and_terminal(reward, terminated or truncated)

            if terminated or truncated:
                break

        # Should have collected experience
        assert len(agent.states) > 0
        assert len(agent.rewards) > 0

    def test_training_loop(self):
        """Test a mini training loop."""
        env = TradingEnv(
            symbol="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-06-01"
        )

        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device='cpu'
        )

        # Run 2 short episodes
        for episode in range(2):
            obs, info = env.reset()

            for _ in range(20):
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                agent.store_reward_and_terminal(reward, terminated or truncated)

                if terminated or truncated:
                    break

            # Update after each episode
            if len(agent.states) > 0:
                metrics = agent.update()
                assert 'loss' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Agent for trading.
Implements actor-critic architecture with PyTorch.
Optimized for CUDA GPU training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Actor: Outputs action probabilities
    Critic: Outputs state value estimate
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize actor-critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
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
        """
        Forward pass through network.

        Args:
            state: State tensor

        Returns:
            (action_probs, state_value)
        """
        features = self.feature_extractor(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

    def act(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: State tensor

        Returns:
            (action, log_prob, state_value)
        """
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, state_value

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions given states.

        Args:
            states: Batch of states
            actions: Batch of actions

        Returns:
            (log_probs, state_values, dist_entropy)
        """
        action_probs, state_values = self.forward(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return log_probs, state_values, dist_entropy


class PPOAgent:
    """
    PPO Agent for trading environment.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = None
    ):
        """
        Initialize PPO agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            k_epochs: Number of epochs to update policy
            entropy_coef: Entropy coefficient for exploration
            value_coef: Value loss coefficient
            max_grad_norm: Max gradient norm for clipping
            device: Device to run on ('cuda' or 'cpu')
        """
        # Auto-detect CUDA if available
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = torch.device(device)
        logger.info(f"PPO Agent using device: {self.device}")

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Networks
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Storage for trajectories
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using current policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.policy_old.act(state_tensor)

        # Store for training
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())

        return action

    def store_reward_and_terminal(self, reward: float, is_terminal: bool):
        """Store reward and terminal flag."""
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.

        Returns:
            Dictionary with training metrics
        """
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Calculate discounted rewards
        rewards_to_go = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)

        rewards_to_go = torch.FloatTensor(rewards_to_go).to(self.device)

        # Normalize rewards
        rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-8)

        # Training metrics
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        # Optimize policy for k epochs
        for _ in range(self.k_epochs):
            # Evaluate actions
            log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)

            # Calculate advantages
            advantages = rewards_to_go - state_values.detach().squeeze()

            # Calculate ratio for PPO
            ratios = torch.exp(log_probs - old_log_probs)

            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.value_coef * nn.MSELoss()(state_values.squeeze(), rewards_to_go)
            entropy_loss = -self.entropy_coef * dist_entropy.mean()

            loss = policy_loss + value_loss + entropy_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += dist_entropy.mean().item()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear memory
        self.clear_memory()

        metrics = {
            'loss': total_loss / self.k_epochs,
            'policy_loss': total_policy_loss / self.k_epochs,
            'value_loss': total_value_loss / self.k_epochs,
            'entropy': total_entropy / self.k_epochs
        }

        return metrics

    def clear_memory(self):
        """Clear stored trajectories."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test PPO agent
    print("Testing PPO Agent...")

    state_dim = 43  # From TradingEnv
    action_dim = 3  # HOLD, BUY, SELL

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Agent device: {agent.device}")
    print(f"Policy network: {agent.policy}")

    # Test action selection
    dummy_state = np.random.randn(state_dim)
    action = agent.select_action(dummy_state)
    print(f"Selected action: {action}")

    # Test update with dummy data
    for _ in range(10):
        state = np.random.randn(state_dim)
        action = agent.select_action(state)
        agent.store_reward_and_terminal(np.random.randn(), False)

    metrics = agent.update()
    print(f"Training metrics: {metrics}")

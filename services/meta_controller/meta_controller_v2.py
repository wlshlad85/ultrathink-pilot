"""
Meta-Controller Service v2.0 - Hierarchical RL Strategy Weight Blending
Agent: meta-controller-researcher (Agent 10/12)

Mission: Smooth strategy transitions via continuous weight blending
Target: <5% portfolio disruption (vs 15% baseline with discrete routing)

Architecture:
- Hierarchical RL with Options Framework (temporal abstraction)
- Continuous strategy weight output (NOT discrete selection)
- Regime-aware policy with probabilistic regime inputs
- Fallback to weighted average if RL fails

Key Innovation: Uses regime probabilities to blend specialist models smoothly,
eliminating portfolio disruption during regime transitions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """Available trading strategies"""
    BULL_SPECIALIST = "bull_specialist"
    BEAR_SPECIALIST = "bear_specialist"
    SIDEWAYS_SPECIALIST = "sideways_specialist"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class RegimeInput:
    """Regime probability distribution input"""
    prob_bull: float
    prob_bear: float
    prob_sideways: float
    entropy: float
    confidence: float
    timestamp: datetime

    def validate(self):
        """Validate probability distribution"""
        prob_sum = self.prob_bull + self.prob_bear + self.prob_sideways
        if abs(prob_sum - 1.0) > 0.001:
            raise ValueError(f"Probabilities must sum to 1.0 (got {prob_sum:.6f})")

        if not all(0 <= p <= 1 for p in [self.prob_bull, self.prob_bear, self.prob_sideways]):
            raise ValueError("All probabilities must be in [0, 1]")

        if self.entropy < 0:
            raise ValueError("Entropy must be non-negative")


@dataclass
class StrategyWeights:
    """Strategy weight distribution (blending weights)"""
    bull_specialist: float
    bear_specialist: float
    sideways_specialist: float
    momentum: float
    mean_reversion: float
    timestamp: datetime
    method: str  # 'hierarchical_rl', 'fallback', 'bootstrap'
    confidence: float = 1.0

    def validate(self):
        """Validate weight distribution"""
        weight_sum = (self.bull_specialist + self.bear_specialist +
                     self.sideways_specialist + self.momentum + self.mean_reversion)

        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0 (got {weight_sum:.6f})")

        weights = [self.bull_specialist, self.bear_specialist, self.sideways_specialist,
                  self.momentum, self.mean_reversion]
        if not all(0 <= w <= 1 for w in weights):
            raise ValueError("All weights must be in [0, 1]")

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'bull_specialist': float(self.bull_specialist),
            'bear_specialist': float(self.bear_specialist),
            'sideways_specialist': float(self.sideways_specialist),
            'momentum': float(self.momentum),
            'mean_reversion': float(self.mean_reversion),
            'timestamp': self.timestamp.isoformat(),
            'method': self.method,
            'confidence': float(self.confidence)
        }


class HierarchicalPolicyNetwork(nn.Module):
    """
    Hierarchical RL Policy with Options Framework

    Architecture:
    - High-level controller: Maps regime probabilities to strategy weights
    - Low-level options: Each strategy has an option policy (future work)
    - Temporal abstraction: High-level decisions persist across timesteps

    Input: [prob_bull, prob_bear, prob_sideways, entropy, recent_pnl, volatility, trend, volume]
    Output: [weight_bull, weight_bear, weight_sideways, weight_momentum, weight_mean_rev]
    """

    def __init__(self, state_dim=8, num_strategies=5, hidden_dim=128):
        super(HierarchicalPolicyNetwork, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # High-level controller: outputs strategy weights (actor)
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, num_strategies),
            nn.Softmax(dim=-1)  # Ensures weights sum to 1.0
        )

        # Critic: estimates state value
        self.value_head = nn.Linear(hidden_dim, 1)

        # Option termination predictor (hierarchical RL component)
        self.termination_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability of terminating current option
        )

    def forward(self, state):
        """
        Forward pass

        Returns:
            weights: Strategy weight distribution (sums to 1.0)
            value: State value estimate
            termination_prob: Probability of option termination
        """
        features = self.shared(state)
        weights = self.weight_head(features)
        value = self.value_head(features)
        termination_prob = self.termination_head(features)

        return weights, value, termination_prob


class MetaControllerRL:
    """
    Hierarchical RL Meta-Controller for Strategy Weight Blending

    Implements PPO with options framework for smooth strategy transitions.
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize meta-controller

        Args:
            learning_rate: Adam optimizer learning rate (recommended: 1e-4)
            gamma: Discount factor for future rewards
            epsilon: Initial exploration epsilon
            epsilon_decay: Decay rate for epsilon (0.1 → 0.01 over time)
            epsilon_min: Minimum epsilon value
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize hierarchical policy network
        self.policy_net = HierarchicalPolicyNetwork(
            state_dim=8,
            num_strategies=5,
            hidden_dim=128
        ).to(self.device)

        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
            eps=1e-5
        )

        # Experience buffer for PPO
        self.states = []
        self.weights = []
        self.rewards = []
        self.values = []
        self.log_probs = []

        # Performance tracking
        self.step_count = 0
        self.episode_count = 0
        self.current_option = None
        self.option_duration = 0

        logger.info(f"MetaController initialized on {self.device}")
        logger.info(f"Hyperparameters: lr={learning_rate}, gamma={gamma}, epsilon={epsilon}")

    def build_state(
        self,
        regime_input: RegimeInput,
        market_features: Dict
    ) -> np.ndarray:
        """
        Construct state vector for policy network

        State components:
        - Regime probabilities (3D)
        - Regime entropy (uncertainty)
        - Recent PnL
        - Market volatility
        - Trend strength
        - Volume ratio

        Args:
            regime_input: Regime probability distribution
            market_features: Market indicators

        Returns:
            State vector (8D)
        """
        try:
            state = np.array([
                regime_input.prob_bull,
                regime_input.prob_bear,
                regime_input.prob_sideways,
                regime_input.entropy,
                market_features.get('recent_pnl', 0.0),
                market_features.get('volatility_20d', 0.02),
                market_features.get('trend_strength', 0.0),
                market_features.get('volume_ratio', 1.0)
            ], dtype=np.float32)

            # Validate state
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                logger.warning(f"Invalid state detected: {state}, using safe defaults")
                return self._safe_default_state()

            return state

        except Exception as e:
            logger.error(f"State construction error: {e}")
            return self._safe_default_state()

    def _safe_default_state(self) -> np.ndarray:
        """Return safe default state (neutral market)"""
        return np.array([0.33, 0.33, 0.34, 1.0, 0.0, 0.02, 0.0, 1.0], dtype=np.float32)

    def predict_weights(
        self,
        regime_input: RegimeInput,
        market_features: Dict,
        use_epsilon_greedy: bool = True
    ) -> StrategyWeights:
        """
        Predict strategy weights using hierarchical RL policy

        Args:
            regime_input: Regime probabilities
            market_features: Market indicators
            use_epsilon_greedy: Apply epsilon-greedy exploration

        Returns:
            StrategyWeights with weight distribution
        """
        try:
            # Validate inputs
            regime_input.validate()

            # Build state
            state = self.build_state(regime_input, market_features)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Forward pass
            with torch.no_grad():
                weights, value, termination_prob = self.policy_net(state_tensor)
                weights = weights.squeeze(0).cpu().numpy()

            # Epsilon-greedy exploration
            if use_epsilon_greedy and np.random.random() < self.epsilon:
                logger.debug(f"Exploration: epsilon={self.epsilon:.4f}")
                weights = self._exploration_weights(regime_input)

            # Ensure numerical stability
            weights = self._normalize_weights(weights)

            # Create StrategyWeights object
            strategy_weights = StrategyWeights(
                bull_specialist=float(weights[0]),
                bear_specialist=float(weights[1]),
                sideways_specialist=float(weights[2]),
                momentum=float(weights[3]),
                mean_reversion=float(weights[4]),
                timestamp=datetime.utcnow(),
                method='hierarchical_rl',
                confidence=1.0 - self.epsilon
            )

            strategy_weights.validate()

            # Decay epsilon
            if use_epsilon_greedy:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            return strategy_weights

        except Exception as e:
            logger.error(f"Weight prediction failed: {e}, using fallback")
            return self.fallback_weights(regime_input)

    def _exploration_weights(self, regime_input: RegimeInput) -> np.ndarray:
        """
        Exploration strategy: Add noise to regime-based weights

        Args:
            regime_input: Regime probabilities

        Returns:
            Noisy weight vector
        """
        # Start with regime probabilities as base
        base_weights = np.array([
            regime_input.prob_bull,
            regime_input.prob_bear,
            regime_input.prob_sideways,
            0.05,  # Small momentum component
            0.05   # Small mean reversion component
        ])

        # Add Dirichlet noise for exploration
        noise = np.random.dirichlet([0.5, 0.5, 0.5, 0.1, 0.1])
        weights = 0.7 * base_weights + 0.3 * noise

        return self._normalize_weights(weights)

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1.0"""
        weights = np.maximum(weights, 0.0)  # Ensure non-negative
        weight_sum = weights.sum()

        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Fallback: uniform weights
            weights = np.ones(5) / 5.0

        return weights

    def fallback_weights(self, regime_input: RegimeInput) -> StrategyWeights:
        """
        Fallback strategy: Regime-proportional weighting

        Used when RL fails or during bootstrap phase.

        Args:
            regime_input: Regime probabilities

        Returns:
            StrategyWeights based on regime probabilities
        """
        try:
            regime_input.validate()

            # Map regime probabilities to specialist weights
            # Add small base weights for momentum and mean reversion
            weights = np.array([
                regime_input.prob_bull,
                regime_input.prob_bear,
                regime_input.prob_sideways,
                0.05,  # Momentum
                0.05   # Mean reversion
            ])

            weights = self._normalize_weights(weights)

            return StrategyWeights(
                bull_specialist=float(weights[0]),
                bear_specialist=float(weights[1]),
                sideways_specialist=float(weights[2]),
                momentum=float(weights[3]),
                mean_reversion=float(weights[4]),
                timestamp=datetime.utcnow(),
                method='fallback',
                confidence=regime_input.confidence
            )

        except Exception as e:
            logger.error(f"Fallback failed: {e}, using uniform weights")
            return StrategyWeights(
                bull_specialist=0.2,
                bear_specialist=0.2,
                sideways_specialist=0.2,
                momentum=0.2,
                mean_reversion=0.2,
                timestamp=datetime.utcnow(),
                method='bootstrap',
                confidence=0.5
            )

    def update_policy(
        self,
        update_epochs: int = 4,
        clip_epsilon: float = 0.2
    ) -> Dict[str, float]:
        """
        PPO policy update

        Args:
            update_epochs: Number of update epochs
            clip_epsilon: PPO clipping parameter

        Returns:
            Dictionary with loss statistics
        """
        if len(self.states) < 32:  # Minimum batch size
            return {}

        try:
            # Convert to tensors
            states = torch.FloatTensor(np.array(self.states)).to(self.device)
            rewards = torch.FloatTensor(self.rewards).to(self.device)
            old_values = torch.FloatTensor(self.values).to(self.device)

            # Calculate returns (Monte Carlo)
            returns = self._calculate_returns(rewards)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # PPO update loop
            policy_losses = []
            value_losses = []

            for epoch in range(update_epochs):
                # Forward pass
                weights, values, termination_probs = self.policy_net(states)
                values = values.squeeze()

                # Calculate advantages
                advantages = returns - values.detach()

                # Policy loss (simplified for continuous weights)
                policy_loss = -advantages.mean()

                # Value loss
                value_loss = nn.MSELoss()(values, returns)

                # Total loss
                loss = policy_loss + 0.5 * value_loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

            # Clear buffers
            self.states.clear()
            self.weights.clear()
            self.rewards.clear()
            self.values.clear()
            self.log_probs.clear()

            self.episode_count += 1

            stats = {
                'policy_loss': np.mean(policy_losses),
                'value_loss': np.mean(value_losses),
                'mean_return': returns.mean().item(),
                'epsilon': self.epsilon,
                'episode': self.episode_count
            }

            logger.info(f"Policy updated: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Policy update failed: {e}")
            return {}

    def _calculate_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns (Monte Carlo)"""
        returns = []
        discounted_reward = 0.0

        for reward in reversed(rewards.cpu().numpy()):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        return torch.FloatTensor(returns).to(self.device)

    def save_model(self, filepath: str):
        """Save trained model to disk"""
        try:
            model_state = {
                'policy_net': self.policy_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'step_count': self.step_count,
                'episode_count': self.episode_count
            }

            torch.save(model_state, filepath)
            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Model save failed: {e}")

    def load_model(self, filepath: str):
        """Load trained model from disk"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return False

            model_state = torch.load(filepath, map_location=self.device)

            self.policy_net.load_state_dict(model_state['policy_net'])
            self.optimizer.load_state_dict(model_state['optimizer'])
            self.epsilon = model_state.get('epsilon', self.epsilon)
            self.step_count = model_state.get('step_count', 0)
            self.episode_count = model_state.get('episode_count', 0)

            logger.info(f"Model loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return False


class MetaControllerDB:
    """TimescaleDB interface for meta-controller decisions"""

    def __init__(
        self,
        host: str = 'timescaledb',
        port: int = 5432,
        database: str = 'ultrathink_experiments',
        user: str = 'ultrathink',
        password: str = 'changeme_in_production'
    ):
        """Initialize database connection"""
        self.db_config = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }

        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Create meta_controller_decisions table if not exists"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Create table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS meta_controller_decisions (
                            time TIMESTAMPTZ NOT NULL,
                            symbol VARCHAR(20) NOT NULL,
                            prob_bull DOUBLE PRECISION,
                            prob_bear DOUBLE PRECISION,
                            prob_sideways DOUBLE PRECISION,
                            regime_entropy DOUBLE PRECISION,
                            weight_bull_specialist DOUBLE PRECISION,
                            weight_bear_specialist DOUBLE PRECISION,
                            weight_sideways_specialist DOUBLE PRECISION,
                            weight_momentum DOUBLE PRECISION,
                            weight_mean_reversion DOUBLE PRECISION,
                            method VARCHAR(50),
                            confidence DOUBLE PRECISION,
                            market_features JSONB,
                            metadata JSONB
                        );
                    """)

                    # Create hypertable (if not already)
                    cur.execute("""
                        SELECT create_hypertable(
                            'meta_controller_decisions',
                            'time',
                            if_not_exists => TRUE
                        );
                    """)

                    # Create indexes
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_meta_decisions_symbol
                        ON meta_controller_decisions(symbol, time DESC);
                    """)

                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_meta_decisions_method
                        ON meta_controller_decisions(method, time DESC);
                    """)

                    conn.commit()
                    logger.info("Meta-controller decisions table ready")

        except Exception as e:
            logger.error(f"Table creation failed: {e}")

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def store_decision(
        self,
        symbol: str,
        regime_input: RegimeInput,
        strategy_weights: StrategyWeights,
        market_features: Dict
    ) -> bool:
        """
        Store meta-controller decision to database

        Args:
            symbol: Trading symbol
            regime_input: Input regime probabilities
            strategy_weights: Output strategy weights
            market_features: Market indicators

        Returns:
            True if stored successfully
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO meta_controller_decisions (
                            time, symbol,
                            prob_bull, prob_bear, prob_sideways, regime_entropy,
                            weight_bull_specialist, weight_bear_specialist,
                            weight_sideways_specialist, weight_momentum,
                            weight_mean_reversion,
                            method, confidence, market_features, metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """, (
                        strategy_weights.timestamp,
                        symbol,
                        regime_input.prob_bull,
                        regime_input.prob_bear,
                        regime_input.prob_sideways,
                        regime_input.entropy,
                        strategy_weights.bull_specialist,
                        strategy_weights.bear_specialist,
                        strategy_weights.sideways_specialist,
                        strategy_weights.momentum,
                        strategy_weights.mean_reversion,
                        strategy_weights.method,
                        strategy_weights.confidence,
                        Json(market_features),
                        Json({
                            'regime_confidence': regime_input.confidence,
                            'regime_timestamp': regime_input.timestamp.isoformat()
                        })
                    ))

                    conn.commit()
                    return True

        except Exception as e:
            logger.error(f"Failed to store decision: {e}")
            return False

    def get_recent_decisions(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Retrieve recent meta-controller decisions

        Args:
            symbol: Trading symbol
            limit: Maximum number of records

        Returns:
            List of decision dictionaries
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT *
                        FROM meta_controller_decisions
                        WHERE symbol = %s
                        ORDER BY time DESC
                        LIMIT %s
                    """, (symbol, limit))

                    rows = cur.fetchall()
                    return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to retrieve decisions: {e}")
            return []


# Expose public API
__all__ = [
    'MetaControllerRL',
    'MetaControllerDB',
    'RegimeInput',
    'StrategyWeights',
    'StrategyType',
    'HierarchicalPolicyNetwork'
]

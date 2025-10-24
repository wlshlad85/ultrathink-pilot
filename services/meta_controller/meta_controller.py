"""
Meta-Controller Service - Hierarchical RL Strategy Selection
Agent: meta-controller-researcher

Uses PPO (Proximal Policy Optimization) to select optimal trading strategies
based on market regime classifications. Tracks performance in MLflow.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from kafka import KafkaProducer, KafkaConsumer
import json
import logging
from datetime import datetime
import redis
import pickle
from collections import deque
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    """PPO policy network for strategy selection"""

    def __init__(self, state_dim=8, action_dim=5, hidden_dim=128):
        super(PolicyNetwork, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value

class MetaController:
    """
    Hierarchical RL controller for strategy selection:
    - Action Space: [trend_following, mean_reversion, volatility_arbitrage,
                     momentum, market_making]
    - State Space: [regime_id, confidence, returns, volatility, volume_ratio,
                    trend_strength, recent_pnl, strategy_performance]
    """

    def __init__(self, kafka_bootstrap='kafka-1:9092', redis_host='redis',
                 redis_port=6379, mlflow_uri='http://mlflow:5000'):
        self.kafka_bootstrap = kafka_bootstrap
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=2)

        # MLflow setup
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("strategy_selection")

        # Strategy portfolio
        self.strategies = [
            'trend_following',
            'mean_reversion',
            'volatility_arbitrage',
            'momentum',
            'market_making'
        ]

        # Initialize PPO model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = PolicyNetwork(state_dim=8, action_dim=5).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon_clip = 0.2  # PPO clipping parameter
        self.update_frequency = 100  # Update every N steps

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

        # Performance tracking
        self.strategy_performance = {s: deque(maxlen=1000) for s in self.strategies}
        self.current_position = None
        self.episode_steps = 0

        self._load_cached_model()

    def build_state(self, regime_info, market_data):
        """Construct state vector from regime info and market data"""
        try:
            regime_id = regime_info.get('regime_id', 0)
            confidence = regime_info.get('confidence', 0.5)

            returns = market_data.get('returns', 0.0)
            volatility = market_data.get('volatility', 0.01)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            trend_strength = market_data.get('trend_strength', 0.0)

            # Calculate recent PnL and strategy performance
            recent_pnl = self._calculate_recent_pnl()
            strategy_perf = self._calculate_strategy_performance()

            state = np.array([
                regime_id / 3.0,  # Normalize to [0, 1]
                confidence,
                returns,
                volatility,
                volume_ratio,
                trend_strength,
                recent_pnl,
                strategy_perf
            ], dtype=np.float32)

            return state
        except Exception as e:
            logger.error(f"State construction error: {e}")
            return np.zeros(8, dtype=np.float32)

    def select_strategy(self, state):
        """Select strategy using PPO policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs, state_value = self.policy_net(state_tensor)

        # Sample action from policy
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob.item(), state_value.item()

    def update_policy(self):
        """PPO policy update"""
        if len(self.states) < self.update_frequency:
            return

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Calculate returns (Monte Carlo)
        returns = []
        discounted_reward = 0
        for reward in reversed(self.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # PPO update loop
        for _ in range(4):  # K epochs
            action_probs, state_values = self.policy_net(states)
            state_values = state_values.squeeze()

            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()

            # Calculate advantages
            advantages = returns - state_values.detach()

            # Calculate ratios
            ratios = torch.exp(new_log_probs - old_log_probs)

            # PPO clipped objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            value_loss = nn.MSELoss()(state_values, returns)

            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()

        # Log to MLflow
        mlflow.log_metrics({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'mean_return': returns.mean().item()
        }, step=self.episode_steps)

        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()

        self._cache_model()
        logger.info("Policy updated via PPO")

    def _calculate_recent_pnl(self):
        """Calculate recent PnL from performance history"""
        try:
            pnl_data = self.redis_client.get('recent_pnl')
            if pnl_data:
                return float(pnl_data)
        except:
            pass
        return 0.0

    def _calculate_strategy_performance(self):
        """Calculate weighted strategy performance score"""
        try:
            total_score = 0.0
            for strategy, performance in self.strategy_performance.items():
                if len(performance) > 0:
                    total_score += np.mean(performance)
            return total_score / len(self.strategies)
        except:
            return 0.0

    def _cache_model(self):
        """Cache trained model to Redis"""
        try:
            model_state = {
                'policy_net': self.policy_net.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            model_bytes = pickle.dumps(model_state)
            self.redis_client.setex('meta_controller_model', 3600, model_bytes)
        except Exception as e:
            logger.warning(f"Model caching failed: {e}")

    def _load_cached_model(self):
        """Load model from Redis cache"""
        try:
            model_bytes = self.redis_client.get('meta_controller_model')
            if model_bytes:
                model_state = pickle.loads(model_bytes)
                self.policy_net.load_state_dict(model_state['policy_net'])
                self.optimizer.load_state_dict(model_state['optimizer'])
                logger.info("Loaded cached model from Redis")
                return True
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")
        return False

def main():
    controller = MetaController()

    # Kafka consumer for regime events
    consumer = KafkaConsumer(
        'regime_events',
        bootstrap_servers=controller.kafka_bootstrap,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        group_id='meta_controller_group'
    )

    # Kafka producer for strategy decisions
    producer = KafkaProducer(
        bootstrap_servers=controller.kafka_bootstrap,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    logger.info(f"Meta-Controller started on {controller.device}")

    with mlflow.start_run():
        for message in consumer:
            try:
                regime_info = message.value

                # Build state (simplified - would fetch market data in production)
                market_data = {
                    'returns': 0.01,
                    'volatility': 0.02,
                    'volume_ratio': 1.1,
                    'trend_strength': 0.5
                }
                state = controller.build_state(regime_info, market_data)

                # Select strategy
                action_idx, log_prob, value = controller.select_strategy(state)
                selected_strategy = controller.strategies[action_idx]

                # Store experience
                controller.states.append(state)
                controller.actions.append(action_idx)
                controller.log_probs.append(log_prob)
                controller.values.append(value)

                # Placeholder reward (would come from actual trading results)
                reward = np.random.normal(0.01, 0.02)
                controller.rewards.append(reward)

                # Publish decision
                decision = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'symbol': regime_info.get('symbol', 'UNKNOWN'),
                    'regime': regime_info.get('regime'),
                    'selected_strategy': selected_strategy,
                    'confidence': regime_info.get('confidence'),
                    'state': state.tolist(),
                    'action_idx': action_idx
                }
                producer.send('strategy_decisions', value=decision)

                logger.info(f"Strategy selected: {selected_strategy} for {regime_info.get('regime')} regime")

                controller.episode_steps += 1

                # Update policy periodically
                if controller.episode_steps % controller.update_frequency == 0:
                    controller.update_policy()

            except Exception as e:
                logger.error(f"Processing error: {e}")

if __name__ == '__main__':
    main()

"""
Reinforcement Learning Agent for the Dopamine Trading System.
Implements a sophisticated RL agent with experience replay and learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List, Tuple
from collections import deque
import random
import structlog

from ..shared.types import (
    State, ActionType, Experience, TradingSignal, RewardComponents
)
from ..shared.constants import (
    DEFAULT_LEARNING_RATE, DEFAULT_DISCOUNT_FACTOR, DEFAULT_EXPLORATION_RATE,
    DEFAULT_BATCH_SIZE, DEFAULT_MEMORY_SIZE, MIN_REPLAY_SIZE,
    NETWORK_INPUT_DIM, NETWORK_OUTPUT_DIM, ACTION_SPACE_SIZE,
    STATE_PRICE_DIM, STATE_VOLUME_DIM, STATE_ACCOUNT_DIM, 
    STATE_MARKET_DIM, STATE_TECHNICAL_DIM, STATE_SUBSYSTEM_DIM
)

logger = structlog.get_logger(__name__)


class DQN(nn.Module):
    """Deep Q-Network for the RL agent."""
    
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int, 
                 dropout_rate: float = 0.2):
        """Initialize DQN architecture.
        
        Args:
            input_dim: Input state dimension
            hidden_layers: List of hidden layer sizes
            output_dim: Output action dimension
            dropout_rate: Dropout rate for regularization
        """
        super(DQN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class RLAgent:
    """Reinforcement Learning trading agent with dopamine-inspired learning."""
    
    def __init__(self, config: dict, network_manager=None):
        """Initialize RL agent.
        
        Args:
            config: Agent configuration parameters
            network_manager: Optional network manager for neural networks
        """
        self.config = config
        self.network_manager = network_manager
        
        # Calculate state dimension from constants
        self.state_dim = (STATE_PRICE_DIM + STATE_VOLUME_DIM + STATE_ACCOUNT_DIM + 
                         STATE_MARKET_DIM + STATE_TECHNICAL_DIM + STATE_SUBSYSTEM_DIM)
        
        # RL parameters
        self.learning_rate = config.get("learning_rate", DEFAULT_LEARNING_RATE)
        self.discount_factor = config.get("discount_factor", DEFAULT_DISCOUNT_FACTOR)
        self.exploration_rate = config.get("exploration_rate", DEFAULT_EXPLORATION_RATE)
        self.batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)
        self.memory_size = config.get("memory_size", DEFAULT_MEMORY_SIZE)
        self.update_frequency = config.get("update_frequency", 100)
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Neural networks
        hidden_layers = config.get("hidden_layers", [256, 128, 64])
        dropout_rate = config.get("dropout_rate", 0.2)
        
        self.q_network = DQN(self.state_dim, hidden_layers, ACTION_SPACE_SIZE, dropout_rate)
        self.target_network = DQN(self.state_dim, hidden_layers, ACTION_SPACE_SIZE, dropout_rate)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Training state
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0.0
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        
        # Update target network
        self.update_target_network()
        
        logger.info(f"RL agent initialized - state_dim: {self.state_dim}, layers: {hidden_layers}, lr: {self.learning_rate}")
    
    def select_action(self, state: State, training: bool = True) -> Tuple[ActionType, float]:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current market state
            training: Whether in training mode
            
        Returns:
            Tuple[ActionType, float]: Selected action and confidence
        """
        try:
            # Convert state to tensor
            state_tensor = self._state_to_tensor(state)
            
            # Epsilon-greedy action selection
            if training and random.random() < self.exploration_rate:
                # Random exploration
                action_idx = random.randint(0, ACTION_SPACE_SIZE - 1)
                confidence = 0.3  # Low confidence for random actions
            else:
                # Greedy action selection
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                    action_idx = q_values.argmax().item()
                    
                    # Calculate confidence from Q-values
                    q_vals_np = q_values.cpu().numpy()
                    max_q = np.max(q_vals_np)
                    confidence = min(1.0, max(0.1, (max_q + 1.0) / 2.0))  # Normalize to [0.1, 1.0]
            
            action = ActionType(action_idx)
            
            logger.debug(
                "Action selected",
                action=action.name,
                confidence=confidence,
                exploration_rate=self.exploration_rate,
                training=training
            )
            
            return action, confidence
            
        except Exception as e:
            logger.error("Failed to select action", error=str(e))
            return ActionType.HOLD, 0.1
    
    async def select_action_async(self, state: State, training: bool = True) -> Tuple[ActionType, float]:
        """Async version of select_action for compatibility.
        
        Args:
            state: Current market state
            training: Whether in training mode
            
        Returns:
            Tuple[ActionType, float]: Selected action and confidence
        """
        return self.select_action(state, training)
    
    def store_experience(self, state: State, action: ActionType, reward: float,
                        next_state: State, done: bool) -> None:
        """Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            timestamp=self.steps
        )
        
        self.memory.append(experience)
        self.total_reward += reward
        
        logger.debug(
            "Experience stored",
            reward=reward,
            memory_size=len(self.memory),
            total_reward=self.total_reward
        )
    
    def train(self) -> Optional[float]:
        """Train the agent using experience replay.
        
        Returns:
            Optional[float]: Training loss or None if not enough data
        """
        if len(self.memory) < MIN_REPLAY_SIZE:
            return None
        
        try:
            # Sample batch from memory
            batch = random.sample(self.memory, self.batch_size)
            
            # Prepare tensors
            states = torch.stack([self._state_to_tensor(exp.state) for exp in batch])
            actions = torch.tensor([exp.action.value for exp in batch], dtype=torch.long)
            rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)
            next_states = torch.stack([self._state_to_tensor(exp.next_state) for exp in batch])
            dones = torch.tensor([exp.done for exp in batch], dtype=torch.bool)
            
            # Current Q-values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q-values from target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
            
            # Compute loss
            loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
            
            # Update exploration rate
            if self.exploration_rate > self.min_exploration:
                self.exploration_rate *= self.exploration_decay
            
            # Update target network periodically
            if self.steps % self.update_frequency == 0:
                self.update_target_network()
            
            self.steps += 1
            loss_value = loss.item()
            self.loss_history.append(loss_value)
            
            logger.debug(
                "Agent trained",
                loss=loss_value,
                exploration_rate=self.exploration_rate,
                steps=self.steps
            )
            
            return loss_value
            
        except Exception as e:
            logger.error("Training failed", error=str(e))
            return None
    
    def update_target_network(self) -> None:
        """Update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        logger.debug("Target network updated")
    
    def create_trading_signal(self, action: ActionType, confidence: float,
                            current_price: float) -> TradingSignal:
        """Create trading signal from agent action.
        
        Args:
            action: Action selected by agent
            confidence: Confidence in the action
            current_price: Current market price
            
        Returns:
            TradingSignal: Trading signal for NinjaTrader
        """
        # Calculate position size based on confidence
        base_size = 1
        position_size = max(1, min(3, int(base_size * confidence * 2)))
        
        # Calculate stop loss and target prices
        use_stop = confidence > 0.5
        use_target = confidence > 0.6
        
        stop_distance = current_price * 0.01  # 1% stop loss
        target_distance = current_price * 0.02  # 2% profit target
        
        if action == ActionType.BUY:
            stop_price = current_price - stop_distance if use_stop else 0.0
            target_price = current_price + target_distance if use_target else 0.0
        elif action == ActionType.SELL:
            stop_price = current_price + stop_distance if use_stop else 0.0
            target_price = current_price - target_distance if use_target else 0.0
        else:
            stop_price = target_price = 0.0
            use_stop = use_target = False
        
        return TradingSignal(
            action=action.value,
            confidence=confidence,
            position_size=position_size,
            use_stop=use_stop,
            stop_price=stop_price,
            use_target=use_target,
            target_price=target_price
        )
    
    def get_performance_stats(self) -> dict:
        """Get agent performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        return {
            "steps": self.steps,
            "episodes": self.episodes,
            "total_reward": self.total_reward,
            "memory_size": len(self.memory),
            "exploration_rate": self.exploration_rate,
            "avg_loss": np.mean(list(self.loss_history)) if self.loss_history else 0.0,
            "avg_performance": np.mean(list(self.performance_history)) if self.performance_history else 0.0
        }
    
    def save_model(self, filepath: str) -> None:
        """Save agent model to file.
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes,
            'exploration_rate': self.exploration_rate,
            'config': self.config
        }, filepath)
        
        logger.info("Model saved", filepath=filepath)
    
    async def store_experience(self, state: State, action: ActionType, reward: float, next_state: State, done: bool = False) -> None:
        """Store experience in replay buffer (async version).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
        
        self.steps += 1
        self.total_reward += reward
        
        logger.debug(
            "Experience stored",
            reward=reward,
            memory_size=len(self.memory),
            total_reward=self.total_reward
        )
    
    def should_train(self) -> bool:
        """Check if agent should train based on memory size.
        
        Returns:
            bool: True if agent should train
        """
        return len(self.memory) >= MIN_REPLAY_SIZE
    
    async def train_step(self) -> Optional[float]:
        """Perform a single training step (async version).
        
        Returns:
            Optional[float]: Training loss or None if not enough data
        """
        return self.train()
    
    async def update_with_trade_result(self, reward: float) -> None:
        """Update agent with trade result.
        
        Args:
            reward: Trade result reward
        """
        # Update running performance
        self.performance_history.append(reward)
        
        # Update total reward
        self.total_reward += reward
        
        logger.debug("Trade result processed", reward=reward, total_reward=self.total_reward)
    
    def reduce_exploration(self) -> None:
        """Reduce exploration rate (for good performance)."""
        self.exploration_rate *= 0.9
        self.exploration_rate = max(self.exploration_rate, self.min_exploration)
        logger.debug("Exploration reduced", exploration_rate=self.exploration_rate)
    
    def increase_exploration(self) -> None:
        """Increase exploration rate (for poor performance)."""
        self.exploration_rate *= 1.1
        self.exploration_rate = min(self.exploration_rate, 0.5)  # Cap at 50%
        logger.debug("Exploration increased", exploration_rate=self.exploration_rate)
    
    def get_performance_metrics(self) -> dict:
        """Get performance metrics for monitoring.
        
        Returns:
            dict: Performance metrics
        """
        return {
            "steps": self.steps,
            "episodes": self.episodes,
            "total_reward": self.total_reward,
            "memory_size": len(self.memory),
            "exploration_rate": self.exploration_rate,
            "avg_loss": np.mean(list(self.loss_history)) if self.loss_history else 0.0,
            "avg_performance": np.mean(list(self.performance_history)) if self.performance_history else 0.0,
            "recent_performance": np.mean(list(self.performance_history)[-10:]) if len(self.performance_history) >= 10 else 0.0
        }
    
    def load_model(self, filepath: str) -> None:
        """Load agent model from file.
        
        Args:
            filepath: Path to load model from
        """
        checkpoint = torch.load(filepath)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.exploration_rate = checkpoint['exploration_rate']
        
        logger.info("Model loaded", filepath=filepath, steps=self.steps)
    
    def _state_to_tensor(self, state: State) -> torch.Tensor:
        """Convert state to PyTorch tensor.
        
        Args:
            state: State object
            
        Returns:
            torch.Tensor: State as tensor
        """
        # Concatenate all state components
        state_vector = np.concatenate([
            state.prices.flatten(),
            state.volumes.flatten(),
            state.account_metrics.flatten(),
            state.market_conditions.flatten(),
            state.technical_indicators.flatten(),
            state.subsystem_signals.flatten()
        ])
        
        return torch.tensor(state_vector, dtype=torch.float32)
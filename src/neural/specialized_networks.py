"""
Specialized Networks for task-specific neural network architectures.
Implements prediction, classification, and reinforcement learning networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import structlog

from ..shared.types import ActionType, State
from ..shared.constants import (
    DEFAULT_HIDDEN_LAYERS, DEFAULT_ACTIVATION, DEFAULT_DROPOUT_RATE,
    NETWORK_INPUT_DIM, NETWORK_OUTPUT_DIM, ACTION_SPACE_SIZE
)

logger = structlog.get_logger(__name__)


class PredictionNetwork(nn.Module):
    """Neural network for price and volatility prediction."""
    
    def __init__(self, input_dim: int, prediction_horizons: List[int], 
                 hidden_dims: List[int] = None):
        """Initialize prediction network.
        
        Args:
            input_dim: Input feature dimension
            prediction_horizons: List of prediction horizons (steps ahead)
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.prediction_horizons = prediction_horizons
        self.hidden_dims = hidden_dims or DEFAULT_HIDDEN_LAYERS
        
        # Feature extraction layers
        self.feature_extractor = self._build_feature_extractor()
        
        # Prediction heads for different horizons
        self.prediction_heads = nn.ModuleDict()
        for horizon in prediction_horizons:
            self.prediction_heads[f"horizon_{horizon}"] = nn.ModuleDict({
                "price": nn.Linear(self.hidden_dims[-1], 1),
                "volatility": nn.Linear(self.hidden_dims[-1], 1),
                "direction": nn.Linear(self.hidden_dims[-1], 3)  # Up, Down, Sideways
            })
        
        # Attention mechanism for multi-horizon fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dims[-1],
            num_heads=4,
            dropout=0.1
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(self.hidden_dims[-1], len(prediction_horizons))
        
        self.dropout = nn.Dropout(DEFAULT_DROPOUT_RATE)
        
    def _build_feature_extractor(self) -> nn.Sequential:
        """Build feature extraction layers."""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(DEFAULT_DROPOUT_RATE)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through prediction network.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Dict[str, torch.Tensor]: Predictions for each horizon
        """
        # Feature extraction
        features = self.feature_extractor(x)
        features = self.dropout(features)
        
        # Multi-horizon predictions
        predictions = {}
        horizon_features = []
        
        for horizon in self.prediction_horizons:
            horizon_key = f"horizon_{horizon}"
            head = self.prediction_heads[horizon_key]
            
            # Individual predictions
            price_pred = head["price"](features)
            volatility_pred = torch.sigmoid(head["volatility"](features))  # 0-1 range
            direction_pred = F.softmax(head["direction"](features), dim=-1)
            
            predictions[horizon_key] = {
                "price": price_pred,
                "volatility": volatility_pred,
                "direction": direction_pred
            }
            
            horizon_features.append(features)
        
        # Attention-based fusion
        if len(horizon_features) > 1:
            stacked_features = torch.stack(horizon_features, dim=1)
            attended_features, attention_weights = self.attention(
                stacked_features, stacked_features, stacked_features
            )
            
            # Uncertainty estimation
            uncertainty = torch.sigmoid(self.uncertainty_head(attended_features.mean(dim=1)))
            predictions["uncertainty"] = uncertainty
            predictions["attention_weights"] = attention_weights
        
        return predictions
    
    def get_ensemble_prediction(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get ensemble prediction across all horizons."""
        predictions = self.forward(x)
        
        # Weighted average based on uncertainty
        if "uncertainty" in predictions:
            weights = 1.0 / (predictions["uncertainty"] + 1e-8)
            weights = weights / weights.sum(dim=-1, keepdim=True)
        else:
            weights = torch.ones(len(self.prediction_horizons)) / len(self.prediction_horizons)
        
        ensemble_price = torch.zeros_like(predictions[f"horizon_{self.prediction_horizons[0]}"]["price"])
        ensemble_volatility = torch.zeros_like(predictions[f"horizon_{self.prediction_horizons[0]}"]["volatility"])
        ensemble_direction = torch.zeros_like(predictions[f"horizon_{self.prediction_horizons[0]}"]["direction"])
        
        for i, horizon in enumerate(self.prediction_horizons):
            horizon_key = f"horizon_{horizon}"
            w = weights[:, i:i+1] if "uncertainty" in predictions else weights[i]
            
            ensemble_price += w * predictions[horizon_key]["price"]
            ensemble_volatility += w * predictions[horizon_key]["volatility"]
            ensemble_direction += w * predictions[horizon_key]["direction"]
        
        return {
            "price": ensemble_price,
            "volatility": ensemble_volatility,
            "direction": ensemble_direction
        }


class ClassificationNetwork(nn.Module):
    """Neural network for pattern and regime classification."""
    
    def __init__(self, input_dim: int, num_classes: Dict[str, int], 
                 hidden_dims: List[int] = None):
        """Initialize classification network.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Dictionary mapping task names to number of classes
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims or DEFAULT_HIDDEN_LAYERS
        
        # Shared feature extraction
        self.shared_layers = self._build_shared_layers()
        
        # Task-specific classification heads
        self.classification_heads = nn.ModuleDict()
        for task_name, num_class in num_classes.items():
            self.classification_heads[task_name] = nn.Sequential(
                nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Dropout(DEFAULT_DROPOUT_RATE),
                nn.Linear(self.hidden_dims[-1] // 2, num_class)
            )
        
        # Confidence estimation
        self.confidence_head = nn.Linear(self.hidden_dims[-1], len(num_classes))
        
        self.dropout = nn.Dropout(DEFAULT_DROPOUT_RATE)
    
    def _build_shared_layers(self) -> nn.Sequential:
        """Build shared feature extraction layers."""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(DEFAULT_DROPOUT_RATE)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through classification network.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Dict[str, torch.Tensor]: Classifications for each task
        """
        # Shared feature extraction
        shared_features = self.shared_layers(x)
        shared_features = self.dropout(shared_features)
        
        # Task-specific classifications
        predictions = {}
        for task_name, head in self.classification_heads.items():
            logits = head(shared_features)
            predictions[task_name] = {
                "logits": logits,
                "probabilities": F.softmax(logits, dim=-1)
            }
        
        # Confidence estimation
        confidence = torch.sigmoid(self.confidence_head(shared_features))
        predictions["confidence"] = confidence
        
        return predictions


class ReinforcementLearningNetwork(nn.Module):
    """Neural network for reinforcement learning (Q-learning, Actor-Critic)."""
    
    def __init__(self, state_dim: int, action_dim: int, network_type: str = "dqn",
                 hidden_dims: List[int] = None):
        """Initialize RL network.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            network_type: Type of RL network ("dqn", "actor_critic", "ddpg")
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network_type = network_type
        self.hidden_dims = hidden_dims or DEFAULT_HIDDEN_LAYERS
        
        # State representation layers
        self.state_encoder = self._build_state_encoder()
        
        # Network-specific architectures
        if network_type == "dqn":
            self._build_dqn_heads()
        elif network_type == "actor_critic":
            self._build_actor_critic_heads()
        elif network_type == "ddpg":
            self._build_ddpg_heads()
        else:
            raise ValueError(f"Unsupported network type: {network_type}")
        
        # Exploration components
        self.exploration_head = nn.Linear(self.hidden_dims[-1], 1)
        
        self.dropout = nn.Dropout(DEFAULT_DROPOUT_RATE)
    
    def _build_state_encoder(self) -> nn.Sequential:
        """Build state encoding layers."""
        layers = []
        prev_dim = self.state_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(DEFAULT_DROPOUT_RATE)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_dqn_heads(self):
        """Build DQN-specific heads."""
        # Q-value estimation
        self.q_head = nn.Linear(self.hidden_dims[-1], self.action_dim)
        
        # Dueling DQN components
        self.value_head = nn.Linear(self.hidden_dims[-1], 1)
        self.advantage_head = nn.Linear(self.hidden_dims[-1], self.action_dim)
    
    def _build_actor_critic_heads(self):
        """Build Actor-Critic specific heads."""
        # Actor (policy) head
        self.actor_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1] // 2, self.action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) head
        self.critic_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1] // 2, 1)
        )
    
    def _build_ddpg_heads(self):
        """Build DDPG-specific heads."""
        # Actor head (continuous actions)
        self.actor_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1] // 2, self.action_dim),
            nn.Tanh()
        )
        
        # Critic head (Q-value with state-action input)
        self.critic_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1] + self.action_dim, self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1] // 2, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass through RL network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim] (for critic in DDPG)
            
        Returns:
            Dict[str, torch.Tensor]: Network outputs
        """
        # State encoding
        state_features = self.state_encoder(state)
        state_features = self.dropout(state_features)
        
        outputs = {}
        
        if self.network_type == "dqn":
            # Q-values
            q_values = self.q_head(state_features)
            
            # Dueling DQN
            value = self.value_head(state_features)
            advantage = self.advantage_head(state_features)
            
            # Combine value and advantage
            dueling_q = value + advantage - advantage.mean(dim=-1, keepdim=True)
            
            outputs.update({
                "q_values": q_values,
                "dueling_q": dueling_q,
                "value": value,
                "advantage": advantage
            })
        
        elif self.network_type == "actor_critic":
            # Policy and value
            policy = self.actor_head(state_features)
            value = self.critic_head(state_features)
            
            outputs.update({
                "policy": policy,
                "value": value
            })
        
        elif self.network_type == "ddpg":
            # Actor output
            actor_output = self.actor_head(state_features)
            outputs["action"] = actor_output
            
            # Critic output (if action provided)
            if action is not None:
                critic_input = torch.cat([state_features, action], dim=-1)
                critic_output = self.critic_head(critic_input)
                outputs["q_value"] = critic_output
        
        # Exploration bonus
        exploration_bonus = torch.sigmoid(self.exploration_head(state_features))
        outputs["exploration_bonus"] = exploration_bonus
        
        return outputs
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get action from network with exploration.
        
        Args:
            state: State tensor
            epsilon: Exploration probability
            
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Action and additional info
        """
        outputs = self.forward(state)
        
        if self.network_type == "dqn":
            q_values = outputs["dueling_q"]
            
            # Epsilon-greedy exploration
            if np.random.random() < epsilon:
                action = torch.randint(0, self.action_dim, (state.size(0),))
            else:
                action = q_values.argmax(dim=-1)
            
            return action, {"q_values": q_values}
        
        elif self.network_type == "actor_critic":
            policy = outputs["policy"]
            
            # Sample from policy
            action = torch.multinomial(policy, 1).squeeze(-1)
            
            return action, {"policy": policy, "value": outputs["value"]}
        
        elif self.network_type == "ddpg":
            action = outputs["action"]
            
            # Add exploration noise
            if epsilon > 0:
                noise = torch.randn_like(action) * epsilon
                action = torch.clamp(action + noise, -1, 1)
            
            return action, {"action": action}
        
        return torch.zeros(state.size(0)), {}


class EnsembleNetwork(nn.Module):
    """Ensemble of specialized networks for robust predictions."""
    
    def __init__(self, networks: List[nn.Module], ensemble_method: str = "average"):
        """Initialize ensemble network.
        
        Args:
            networks: List of networks to ensemble
            ensemble_method: Method for combining predictions ("average", "weighted", "voting")
        """
        super().__init__()
        
        self.networks = nn.ModuleList(networks)
        self.ensemble_method = ensemble_method
        self.num_networks = len(networks)
        
        # Learnable weights for weighted ensemble
        if ensemble_method == "weighted":
            self.ensemble_weights = nn.Parameter(torch.ones(self.num_networks) / self.num_networks)
        
        # Performance tracking for adaptive weighting
        self.network_performance = torch.ones(self.num_networks)
        self.performance_history = deque(maxlen=1000)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Dict[str, torch.Tensor]: Ensemble predictions
        """
        # Get predictions from all networks
        network_outputs = []
        for network in self.networks:
            output = network(x)
            network_outputs.append(output)
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == "average":
            ensemble_output = self._average_predictions(network_outputs)
        elif self.ensemble_method == "weighted":
            ensemble_output = self._weighted_predictions(network_outputs)
        elif self.ensemble_method == "voting":
            ensemble_output = self._voting_predictions(network_outputs)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
        
        # Add individual network outputs for analysis
        ensemble_output["individual_outputs"] = network_outputs
        ensemble_output["ensemble_weights"] = self.get_current_weights()
        
        return ensemble_output
    
    def _average_predictions(self, outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Average predictions from all networks."""
        ensemble_output = {}
        
        # Find common keys
        common_keys = set(outputs[0].keys())
        for output in outputs[1:]:
            common_keys &= set(output.keys())
        
        # Average common predictions
        for key in common_keys:
            if isinstance(outputs[0][key], torch.Tensor):
                stacked = torch.stack([output[key] for output in outputs])
                ensemble_output[key] = stacked.mean(dim=0)
        
        return ensemble_output
    
    def _weighted_predictions(self, outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Weighted average of predictions."""
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_output = {}
        
        common_keys = set(outputs[0].keys())
        for output in outputs[1:]:
            common_keys &= set(output.keys())
        
        for key in common_keys:
            if isinstance(outputs[0][key], torch.Tensor):
                weighted_sum = torch.zeros_like(outputs[0][key])
                for i, output in enumerate(outputs):
                    weighted_sum += weights[i] * output[key]
                ensemble_output[key] = weighted_sum
        
        return ensemble_output
    
    def _voting_predictions(self, outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Majority voting for discrete predictions."""
        ensemble_output = {}
        
        # This is a simplified voting mechanism
        # In practice, you'd need to handle different output types appropriately
        for key in outputs[0].keys():
            if isinstance(outputs[0][key], torch.Tensor):
                # For continuous outputs, fall back to averaging
                stacked = torch.stack([output[key] for output in outputs])
                ensemble_output[key] = stacked.mean(dim=0)
        
        return ensemble_output
    
    def get_current_weights(self) -> torch.Tensor:
        """Get current ensemble weights."""
        if self.ensemble_method == "weighted":
            return F.softmax(self.ensemble_weights, dim=0)
        else:
            return torch.ones(self.num_networks) / self.num_networks
    
    def update_performance(self, network_idx: int, performance: float):
        """Update performance tracking for adaptive weighting."""
        self.network_performance[network_idx] = performance
        self.performance_history.append({
            "network_idx": network_idx,
            "performance": performance
        })
        
        # Update weights based on performance (for weighted ensemble)
        if self.ensemble_method == "weighted":
            with torch.no_grad():
                # Simple performance-based weight update
                normalized_performance = F.softmax(self.network_performance, dim=0)
                learning_rate = 0.01
                self.ensemble_weights.data = (
                    self.ensemble_weights.data * (1 - learning_rate) + 
                    normalized_performance * learning_rate
                )
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get ensemble information."""
        return {
            "num_networks": self.num_networks,
            "ensemble_method": self.ensemble_method,
            "current_weights": self.get_current_weights().tolist(),
            "network_performance": self.network_performance.tolist(),
            "total_parameters": sum(sum(p.numel() for p in net.parameters()) for net in self.networks)
        }
"""
Adaptive Network for self-adapting neural network architectures.
Implements dynamic architecture modification and online learning capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import asyncio
import structlog

from ..shared.types import State, ActionType
from ..shared.constants import (
    DEFAULT_HIDDEN_LAYERS, DEFAULT_ACTIVATION, DEFAULT_DROPOUT_RATE,
    NETWORK_INPUT_DIM, NETWORK_OUTPUT_DIM
)

logger = structlog.get_logger(__name__)


class AdaptiveModule(nn.Module):
    """A neural network module that can adapt its architecture."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 initial_hidden_dims: List[int], activation: str = "relu"):
        """Initialize adaptive module.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            initial_hidden_dims: Initial hidden layer dimensions
            activation: Activation function name
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        
        # Dynamic layer management
        self.hidden_layers = nn.ModuleList()
        self.layer_importance = []
        self.layer_utilization = []
        
        # Build initial architecture
        self._build_initial_architecture(initial_hidden_dims)
        
        # Adaptation parameters
        self.adaptation_threshold = 0.1
        self.pruning_threshold = 0.05
        self.growth_threshold = 0.8
        
        # Performance tracking
        self.layer_activations = {}
        self.gradient_norms = {}
        self.adaptation_history = deque(maxlen=100)
        
    def _build_initial_architecture(self, hidden_dims: List[int]):
        """Build initial network architecture."""
        prev_dim = self.input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.Linear(prev_dim, hidden_dim)
            self.hidden_layers.append(layer)
            
            # Initialize tracking
            self.layer_importance.append(1.0)
            self.layer_utilization.append(0.0)
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
        
        # Activation function
        self.activation = self._get_activation_fn(self.activation_name)
    
    def _get_activation_fn(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU()
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation tracking."""
        self.layer_activations.clear()
        
        current_x = x
        
        for i, layer in enumerate(self.hidden_layers):
            current_x = layer(current_x)
            
            # Track activation statistics
            self.layer_activations[f"layer_{i}"] = {
                "mean": current_x.mean().item(),
                "std": current_x.std().item(),
                "sparsity": (current_x == 0).float().mean().item()
            }
            
            # Apply activation
            current_x = self.activation(current_x)
        
        # Output layer
        output = self.output_layer(current_x)
        
        return output
    
    def adapt_architecture(self, performance_metrics: Dict[str, float]) -> bool:
        """Adapt network architecture based on performance.
        
        Args:
            performance_metrics: Current performance metrics
            
        Returns:
            bool: True if architecture was modified
        """
        try:
            modified = False
            
            # Calculate layer importance based on gradients and activations
            self._update_layer_importance()
            
            # Prune underutilized layers
            if self._should_prune():
                modified |= self._prune_layers()
            
            # Grow network if needed
            if self._should_grow(performance_metrics):
                modified |= self._grow_network()
            
            # Update layer dimensions if needed
            if self._should_resize():
                modified |= self._resize_layers()
            
            if modified:
                self.adaptation_history.append({
                    "type": "architecture_change",
                    "layers": len(self.hidden_layers),
                    "total_params": sum(p.numel() for p in self.parameters()),
                    "performance": performance_metrics.get("loss", 0.0)
                })
                
                logger.info(
                    "Architecture adapted",
                    layers=len(self.hidden_layers),
                    total_params=sum(p.numel() for p in self.parameters())
                )
            
            return modified
            
        except Exception as e:
            logger.error("Architecture adaptation failed", error=str(e))
            return False
    
    def _update_layer_importance(self):
        """Update layer importance based on activations and gradients."""
        for i, layer in enumerate(self.hidden_layers):
            # Activation-based importance
            if f"layer_{i}" in self.layer_activations:
                activation_stats = self.layer_activations[f"layer_{i}"]
                activation_importance = 1.0 - activation_stats["sparsity"]
                
                # Gradient-based importance
                if layer.weight.grad is not None:
                    grad_norm = layer.weight.grad.norm().item()
                    self.gradient_norms[f"layer_{i}"] = grad_norm
                    gradient_importance = min(1.0, grad_norm)
                else:
                    gradient_importance = 0.5
                
                # Combined importance
                combined_importance = (activation_importance + gradient_importance) / 2.0
                
                # Exponential moving average
                alpha = 0.1
                self.layer_importance[i] = (
                    self.layer_importance[i] * (1 - alpha) + 
                    combined_importance * alpha
                )
            
            # Update utilization
            self.layer_utilization[i] = self.layer_importance[i]
    
    def _should_prune(self) -> bool:
        """Check if pruning should be performed."""
        if len(self.hidden_layers) <= 1:
            return False
        
        # Find layers with low importance
        low_importance_layers = [
            i for i, importance in enumerate(self.layer_importance)
            if importance < self.pruning_threshold
        ]
        
        return len(low_importance_layers) > 0
    
    def _should_grow(self, performance_metrics: Dict[str, float]) -> bool:
        """Check if network should grow."""
        if len(self.hidden_layers) >= 10:  # Max layers
            return False
        
        # Check if performance is stagnating
        recent_loss = performance_metrics.get("loss", 0.0)
        
        if len(self.adaptation_history) >= 5:
            recent_losses = [h["performance"] for h in list(self.adaptation_history)[-5:]]
            if len(recent_losses) > 1:
                loss_improvement = recent_losses[0] - recent_losses[-1]
                if loss_improvement < 0.01:  # Little improvement
                    return True
        
        # Check if current layers are highly utilized
        avg_utilization = np.mean(self.layer_utilization)
        return avg_utilization > self.growth_threshold
    
    def _should_resize(self) -> bool:
        """Check if layers should be resized."""
        # Simple heuristic: resize if utilization varies significantly
        if len(self.layer_utilization) < 2:
            return False
        
        utilization_std = np.std(self.layer_utilization)
        return utilization_std > 0.3
    
    def _prune_layers(self) -> bool:
        """Prune underutilized layers."""
        if len(self.hidden_layers) <= 1:
            return False
        
        # Find least important layer
        min_importance_idx = np.argmin(self.layer_importance)
        
        if self.layer_importance[min_importance_idx] < self.pruning_threshold:
            # Remove layer
            del self.hidden_layers[min_importance_idx]
            del self.layer_importance[min_importance_idx]
            del self.layer_utilization[min_importance_idx]
            
            # Reconnect layers
            self._reconnect_layers()
            
            logger.info("Layer pruned", layer_idx=min_importance_idx)
            return True
        
        return False
    
    def _grow_network(self) -> bool:
        """Add new layer to network."""
        if len(self.hidden_layers) >= 10:
            return False
        
        # Find best position to insert layer
        best_position = len(self.hidden_layers) // 2
        
        # Determine new layer size
        if best_position == 0:
            input_size = self.input_dim
        else:
            input_size = self.hidden_layers[best_position - 1].out_features
        
        if best_position == len(self.hidden_layers):
            output_size = self.output_dim
        else:
            output_size = self.hidden_layers[best_position].in_features
        
        # Create new layer with intermediate size
        new_layer_size = (input_size + output_size) // 2
        new_layer = nn.Linear(input_size, new_layer_size)
        
        # Initialize with Xavier initialization
        nn.init.xavier_uniform_(new_layer.weight)
        nn.init.zeros_(new_layer.bias)
        
        # Insert layer
        self.hidden_layers.insert(best_position, new_layer)
        self.layer_importance.insert(best_position, 0.5)
        self.layer_utilization.insert(best_position, 0.5)
        
        # Reconnect layers
        self._reconnect_layers()
        
        logger.info("Layer added", position=best_position, size=new_layer_size)
        return True
    
    def _resize_layers(self) -> bool:
        """Resize existing layers based on utilization."""
        modified = False
        
        for i, layer in enumerate(self.hidden_layers):
            utilization = self.layer_utilization[i]
            current_size = layer.out_features
            
            # Determine new size
            if utilization > 0.8:
                new_size = min(current_size * 2, 1024)  # Grow
            elif utilization < 0.2:
                new_size = max(current_size // 2, 32)   # Shrink
            else:
                continue
            
            if new_size != current_size:
                # Create new layer
                new_layer = nn.Linear(layer.in_features, new_size)
                
                # Copy weights (truncate or pad as needed)
                with torch.no_grad():
                    min_out = min(current_size, new_size)
                    new_layer.weight[:min_out] = layer.weight[:min_out]
                    new_layer.bias[:min_out] = layer.bias[:min_out]
                
                # Replace layer
                self.hidden_layers[i] = new_layer
                modified = True
                
                logger.info(
                    "Layer resized",
                    layer_idx=i,
                    old_size=current_size,
                    new_size=new_size
                )
        
        if modified:
            self._reconnect_layers()
        
        return modified
    
    def _reconnect_layers(self):
        """Reconnect layers after architecture changes."""
        # Update layer connections
        for i in range(len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            
            # Determine input size
            if i == 0:
                expected_input = self.input_dim
            else:
                expected_input = self.hidden_layers[i - 1].out_features
            
            # Recreate layer if input size doesn't match
            if layer.in_features != expected_input:
                new_layer = nn.Linear(expected_input, layer.out_features)
                
                # Initialize with existing weights where possible
                with torch.no_grad():
                    min_in = min(layer.in_features, expected_input)
                    new_layer.weight[:, :min_in] = layer.weight[:, :min_in]
                    new_layer.bias[:] = layer.bias[:]
                
                self.hidden_layers[i] = new_layer
        
        # Update output layer
        if len(self.hidden_layers) > 0:
            expected_input = self.hidden_layers[-1].out_features
            if self.output_layer.in_features != expected_input:
                new_output_layer = nn.Linear(expected_input, self.output_dim)
                
                with torch.no_grad():
                    min_in = min(self.output_layer.in_features, expected_input)
                    new_output_layer.weight[:, :min_in] = self.output_layer.weight[:, :min_in]
                    new_output_layer.bias[:] = self.output_layer.bias[:]
                
                self.output_layer = new_output_layer
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get current architecture information."""
        return {
            "num_layers": len(self.hidden_layers),
            "layer_sizes": [layer.out_features for layer in self.hidden_layers],
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "layer_importance": self.layer_importance.copy(),
            "layer_utilization": self.layer_utilization.copy(),
            "adaptation_count": len(self.adaptation_history)
        }


class AdaptiveNetwork(nn.Module):
    """Self-adapting neural network with dynamic architecture."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize adaptive network.
        
        Args:
            config: Network configuration
        """
        super().__init__()
        
        self.config = config
        
        # Network parameters
        self.input_dim = config.get("input_dim", NETWORK_INPUT_DIM)
        self.output_dim = config.get("output_dim", NETWORK_OUTPUT_DIM)
        self.hidden_layers = config.get("hidden_layers", DEFAULT_HIDDEN_LAYERS)
        self.activation = config.get("activation", DEFAULT_ACTIVATION)
        self.dropout_rate = config.get("dropout_rate", DEFAULT_DROPOUT_RATE)
        
        # Adaptive components
        self.adaptive_module = AdaptiveModule(
            self.input_dim, self.output_dim, self.hidden_layers, self.activation
        )
        
        # Regularization
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Meta-learning components
        self.meta_learning_rate = 0.01
        self.adaptation_frequency = 100
        self.training_steps = 0
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.adaptation_results = deque(maxlen=50)
        
        logger.info(
            "Adaptive network initialized",
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            initial_layers=len(self.hidden_layers)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adaptive network."""
        # Apply dropout to input
        x = self.dropout(x)
        
        # Forward through adaptive module
        output = self.adaptive_module(x)
        
        return output
    
    def training_step(self, loss: float) -> None:
        """Update training state and trigger adaptation if needed.
        
        Args:
            loss: Current training loss
        """
        self.training_steps += 1
        
        # Track performance
        self.performance_history.append({
            "step": self.training_steps,
            "loss": loss,
            "architecture": self.adaptive_module.get_architecture_info()
        })
        
        # Trigger adaptation periodically
        if self.training_steps % self.adaptation_frequency == 0:
            self._trigger_adaptation()
    
    def _trigger_adaptation(self) -> None:
        """Trigger network adaptation based on recent performance."""
        if len(self.performance_history) < 10:
            return
        
        # Calculate recent performance metrics
        recent_history = list(self.performance_history)[-10:]
        recent_losses = [h["loss"] for h in recent_history]
        
        performance_metrics = {
            "loss": np.mean(recent_losses),
            "loss_std": np.std(recent_losses),
            "loss_trend": recent_losses[-1] - recent_losses[0],
            "training_steps": self.training_steps
        }
        
        # Perform adaptation
        try:
            architecture_changed = self.adaptive_module.adapt_architecture(performance_metrics)
            
            if architecture_changed:
                # Record adaptation result
                self.adaptation_results.append({
                    "step": self.training_steps,
                    "performance_before": performance_metrics,
                    "architecture_info": self.adaptive_module.get_architecture_info()
                })
                
                logger.info(
                    "Network adapted",
                    step=self.training_steps,
                    new_architecture=self.adaptive_module.get_architecture_info()
                )
        
        except Exception as e:
            logger.error("Adaptation failed", error=str(e))
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get network parameters."""
        return {
            "config": self.config,
            "architecture": self.adaptive_module.get_architecture_info(),
            "training_steps": self.training_steps,
            "adaptation_count": len(self.adaptation_results)
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set network parameters."""
        if "training_steps" in params:
            self.training_steps = params["training_steps"]
        
        if "adaptation_frequency" in params:
            self.adaptation_frequency = params["adaptation_frequency"]
        
        if "dropout_rate" in params:
            self.dropout_rate = params["dropout_rate"]
            self.dropout = nn.Dropout(self.dropout_rate)
    
    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get adaptation performance metrics."""
        if not self.adaptation_results:
            return {"adaptations": 0}
        
        return {
            "adaptations": len(self.adaptation_results),
            "last_adaptation_step": self.adaptation_results[-1]["step"],
            "current_architecture": self.adaptive_module.get_architecture_info(),
            "performance_trend": self._calculate_performance_trend()
        }
    
    def _calculate_performance_trend(self) -> Dict[str, float]:
        """Calculate performance trend over time."""
        if len(self.performance_history) < 20:
            return {"trend": 0.0}
        
        recent_losses = [h["loss"] for h in list(self.performance_history)[-20:]]
        
        # Simple linear trend
        x = np.arange(len(recent_losses))
        trend = np.polyfit(x, recent_losses, 1)[0]
        
        return {
            "trend": trend,
            "recent_avg_loss": np.mean(recent_losses),
            "loss_stability": 1.0 / (1.0 + np.std(recent_losses))
        }
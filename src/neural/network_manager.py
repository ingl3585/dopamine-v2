"""
Network Manager for coordinating neural network operations.
Implements clean architecture for neural network lifecycle management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Any, Protocol, Tuple
from collections import deque
import asyncio
import structlog
from pathlib import Path

from ..shared.types import State, AISignal, ActionType
from ..shared.constants import (
    DEFAULT_HIDDEN_LAYERS, DEFAULT_ACTIVATION, DEFAULT_DROPOUT_RATE,
    NETWORK_INPUT_DIM, NETWORK_OUTPUT_DIM, MODEL_SAVE_FREQUENCY,
    MODELS_DIR, MODEL_FILE_EXTENSION
)

logger = structlog.get_logger(__name__)


class NetworkProtocol(Protocol):
    """Protocol defining the interface for neural networks."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        ...
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get network parameters."""
        ...
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set network parameters."""
        ...


class NetworkManager:
    """Manages and coordinates all neural network operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize network manager.
        
        Args:
            config: Network configuration
        """
        self.config = config
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network configurations
        self.hidden_layers = config.get("hidden_layers", DEFAULT_HIDDEN_LAYERS)
        self.activation = config.get("activation", DEFAULT_ACTIVATION)
        self.dropout_rate = config.get("dropout_rate", DEFAULT_DROPOUT_RATE)
        self.learning_rate = config.get("learning_rate", 0.001)
        
        # Network registry
        self.networks: Dict[str, nn.Module] = {}
        self.optimizers: Dict[str, optim.Optimizer] = {}
        self.schedulers: Dict[str, optim.lr_scheduler._LRScheduler] = {}
        
        # Training state
        self.training_metrics: Dict[str, List[float]] = {}
        self.network_states: Dict[str, str] = {}  # "training", "inference", "idle"
        
        # Model persistence
        self.models_dir = Path(MODELS_DIR)
        self.models_dir.mkdir(exist_ok=True)
        self.save_frequency = MODEL_SAVE_FREQUENCY
        self.training_steps = 0
        
        # Performance tracking
        self.inference_times: Dict[str, deque] = {}
        self.memory_usage: Dict[str, float] = {}
        
        logger.info(
            "Network manager initialized",
            device=str(self.device),
            hidden_layers=self.hidden_layers,
            models_dir=str(self.models_dir)
        )
    
    def register_network(self, name: str, network: nn.Module, 
                        optimizer_type: str = "adam") -> None:
        """Register a neural network.
        
        Args:
            name: Network name
            network: Network instance
            optimizer_type: Optimizer type ("adam", "sgd", "rmsprop")
        """
        # Move network to device
        network = network.to(self.device)
        self.networks[name] = network
        
        # Initialize optimizer
        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(network.parameters(), lr=self.learning_rate)
        elif optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(network.parameters(), lr=self.learning_rate, momentum=0.9)
        elif optimizer_type.lower() == "rmsprop":
            optimizer = optim.RMSprop(network.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        self.optimizers[name] = optimizer
        
        # Initialize scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
        self.schedulers[name] = scheduler
        
        # Initialize tracking
        self.training_metrics[name] = []
        self.network_states[name] = "idle"
        self.inference_times[name] = deque(maxlen=100)
        self.memory_usage[name] = 0.0
        
        logger.info(
            "Network registered",
            name=name,
            parameters=sum(p.numel() for p in network.parameters()),
            trainable=sum(p.numel() for p in network.parameters() if p.requires_grad),
            optimizer=optimizer_type
        )
    
    async def predict(self, network_name: str, input_data: torch.Tensor) -> torch.Tensor:
        """Make prediction using specified network.
        
        Args:
            network_name: Name of network to use
            input_data: Input tensor
            
        Returns:
            torch.Tensor: Network output
        """
        if network_name not in self.networks:
            raise ValueError(f"Network not found: {network_name}")
        
        network = self.networks[network_name]
        
        # Time inference
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Set to inference mode
            network.eval()
            self.network_states[network_name] = "inference"
            
            with torch.no_grad():
                # Ensure input is on correct device
                input_data = input_data.to(self.device)
                
                # Forward pass
                output = network(input_data)
                
                # Record inference time
                inference_time = (asyncio.get_event_loop().time() - start_time) * 1000
                self.inference_times[network_name].append(inference_time)
                
                return output
                
        except Exception as e:
            logger.error("Prediction failed", network=network_name, error=str(e))
            raise
        finally:
            self.network_states[network_name] = "idle"
    
    async def train_step(self, network_name: str, input_data: torch.Tensor, 
                        target_data: torch.Tensor) -> float:
        """Perform single training step.
        
        Args:
            network_name: Network to train
            input_data: Input batch
            target_data: Target batch
            
        Returns:
            float: Training loss
        """
        if network_name not in self.networks:
            raise ValueError(f"Network not found: {network_name}")
        
        network = self.networks[network_name]
        optimizer = self.optimizers[network_name]
        
        try:
            # Set to training mode
            network.train()
            self.network_states[network_name] = "training"
            
            # Move data to device
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = network(input_data)
            
            # Calculate loss
            loss = self._calculate_loss(output, target_data)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Update scheduler
            scheduler = self.schedulers[network_name]
            scheduler.step()
            
            # Record metrics
            loss_value = loss.item()
            self.training_metrics[network_name].append(loss_value)
            
            # Trim metrics history
            if len(self.training_metrics[network_name]) > 1000:
                self.training_metrics[network_name] = self.training_metrics[network_name][-1000:]
            
            self.training_steps += 1
            
            # Periodic model saving
            if self.training_steps % self.save_frequency == 0:
                await self._save_all_models()
            
            return loss_value
            
        except Exception as e:
            logger.error("Training step failed", network=network_name, error=str(e))
            raise
        finally:
            self.network_states[network_name] = "idle"
    
    async def train_batch(self, network_name: str, batch_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Train on batch of data.
        
        Args:
            network_name: Network to train
            batch_data: List of (input, target) tuples
            
        Returns:
            float: Average batch loss
        """
        if not batch_data:
            return 0.0
        
        total_loss = 0.0
        batch_size = len(batch_data)
        
        for input_data, target_data in batch_data:
            loss = await self.train_step(network_name, input_data, target_data)
            total_loss += loss
        
        avg_loss = total_loss / batch_size
        
        logger.debug(
            "Batch training completed",
            network=network_name,
            batch_size=batch_size,
            avg_loss=avg_loss
        )
        
        return avg_loss
    
    def get_network_info(self, network_name: str) -> Dict[str, Any]:
        """Get information about a network.
        
        Args:
            network_name: Network name
            
        Returns:
            Dict[str, Any]: Network information
        """
        if network_name not in self.networks:
            return {}
        
        network = self.networks[network_name]
        
        # Calculate memory usage
        param_size = sum(p.numel() * p.element_size() for p in network.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in network.buffers())
        total_memory = (param_size + buffer_size) / 1024 / 1024  # MB
        
        self.memory_usage[network_name] = total_memory
        
        return {
            "parameters": sum(p.numel() for p in network.parameters()),
            "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
            "memory_mb": total_memory,
            "state": self.network_states[network_name],
            "recent_loss": self.training_metrics[network_name][-1] if self.training_metrics[network_name] else 0.0,
            "avg_inference_time_ms": np.mean(self.inference_times[network_name]) if self.inference_times[network_name] else 0.0,
            "current_lr": self.optimizers[network_name].param_groups[0]['lr']
        }
    
    def get_all_networks_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all networks.
        
        Returns:
            Dict[str, Dict[str, Any]]: All network information
        """
        return {name: self.get_network_info(name) for name in self.networks.keys()}
    
    async def save_model(self, network_name: str, filepath: Optional[str] = None) -> str:
        """Save model to disk.
        
        Args:
            network_name: Network to save
            filepath: Optional custom filepath
            
        Returns:
            str: Saved file path
        """
        if network_name not in self.networks:
            raise ValueError(f"Network not found: {network_name}")
        
        if filepath is None:
            filepath = self.models_dir / f"{network_name}_{self.training_steps}{MODEL_FILE_EXTENSION}"
        
        network = self.networks[network_name]
        optimizer = self.optimizers[network_name]
        scheduler = self.schedulers[network_name]
        
        checkpoint = {
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_steps": self.training_steps,
            "config": self.config,
            "metrics": self.training_metrics[network_name][-100:]  # Save last 100 metrics
        }
        
        torch.save(checkpoint, filepath)
        
        logger.info("Model saved", network=network_name, filepath=str(filepath))
        return str(filepath)
    
    async def load_model(self, network_name: str, filepath: str) -> None:
        """Load model from disk.
        
        Args:
            network_name: Network to load into
            filepath: Model file path
        """
        if network_name not in self.networks:
            raise ValueError(f"Network not found: {network_name}")
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network state
        network = self.networks[network_name]
        network.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        optimizer = self.optimizers[network_name]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        scheduler = self.schedulers[network_name]
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Restore training state
        self.training_steps = checkpoint.get("training_steps", 0)
        
        # Restore metrics
        if "metrics" in checkpoint:
            self.training_metrics[network_name] = checkpoint["metrics"]
        
        logger.info("Model loaded", network=network_name, filepath=filepath)
    
    async def _save_all_models(self) -> None:
        """Save all registered models."""
        for network_name in self.networks.keys():
            try:
                await self.save_model(network_name)
            except Exception as e:
                logger.error("Failed to save model", network=network_name, error=str(e))
    
    def _calculate_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss based on output and target.
        
        Args:
            output: Network output
            target: Target values
            
        Returns:
            torch.Tensor: Loss value
        """
        # For classification tasks (action prediction)
        if output.shape[-1] == 3:  # Hold, Buy, Sell
            return nn.CrossEntropyLoss()(output, target.long())
        
        # For regression tasks (Q-values, rewards)
        else:
            return nn.MSELoss()(output, target)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        metrics = {
            "total_networks": len(self.networks),
            "training_steps": self.training_steps,
            "device": str(self.device),
            "total_memory_mb": sum(self.memory_usage.values()),
            "network_states": self.network_states.copy()
        }
        
        # Add per-network metrics
        for name in self.networks.keys():
            info = self.get_network_info(name)
            metrics[f"{name}_info"] = info
        
        return metrics
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reset states
        for name in self.network_states:
            self.network_states[name] = "idle"
        
        logger.info("Network manager cleanup completed")
    
    def set_learning_rate(self, network_name: str, learning_rate: float) -> None:
        """Update learning rate for a network.
        
        Args:
            network_name: Network name
            learning_rate: New learning rate
        """
        if network_name not in self.optimizers:
            raise ValueError(f"Network not found: {network_name}")
        
        optimizer = self.optimizers[network_name]
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        logger.debug("Learning rate updated", network=network_name, lr=learning_rate)
    
    def freeze_network(self, network_name: str) -> None:
        """Freeze network parameters.
        
        Args:
            network_name: Network to freeze
        """
        if network_name not in self.networks:
            raise ValueError(f"Network not found: {network_name}")
        
        network = self.networks[network_name]
        for param in network.parameters():
            param.requires_grad = False
        
        logger.info("Network frozen", network=network_name)
    
    def unfreeze_network(self, network_name: str) -> None:
        """Unfreeze network parameters.
        
        Args:
            network_name: Network to unfreeze
        """
        if network_name not in self.networks:
            raise ValueError(f"Network not found: {network_name}")
        
        network = self.networks[network_name]
        for param in network.parameters():
            param.requires_grad = True
        
        logger.info("Network unfrozen", network=network_name)
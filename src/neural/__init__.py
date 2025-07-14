"""
Neural Network Components for the Dopamine Trading System.

This module provides sophisticated neural network architectures including:
- NetworkManager: Coordinates and manages all neural networks
- AdaptiveNetwork: Self-adapting neural networks with dynamic architecture
- SpecializedNetworks: Task-specific network architectures for different use cases
"""

from .network_manager import NetworkManager, NetworkProtocol
from .adaptive_network import AdaptiveNetwork, AdaptiveModule
from .specialized_networks import (
    PredictionNetwork,
    ClassificationNetwork, 
    ReinforcementLearningNetwork,
    EnsembleNetwork
)

__all__ = [
    # Core management
    "NetworkManager",
    "NetworkProtocol",
    
    # Adaptive networks
    "AdaptiveNetwork",
    "AdaptiveModule",
    
    # Specialized networks
    "PredictionNetwork",
    "ClassificationNetwork",
    "ReinforcementLearningNetwork", 
    "EnsembleNetwork"
]
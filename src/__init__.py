"""
Dopamine Trading System - AI-powered trading with reinforcement learning.
"""

__version__ = "2.0.0"
__author__ = "Dopamine Trading Team"

# Core system components
from .core.trading_system import TradingSystem
from .core.trading_system_integrated import IntegratedTradingSystem

# Shared types and utilities
from .shared.types import (
    ActionType, SignalType, MarketPosition, MessageType,
    State, Experience, AISignal, TradingSignal
)

__all__ = [
    'TradingSystem',
    'IntegratedTradingSystem',
    'ActionType',
    'SignalType',
    'MarketPosition',
    'MessageType',
    'State',
    'Experience',
    'AISignal',
    'TradingSignal'
]
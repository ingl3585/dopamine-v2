"""
AI Subsystem Manager for coordinating specialized intelligence modules.
Implements clean architecture with dependency injection and signal aggregation.
"""

import asyncio
from typing import Dict, List, Optional, Any, Protocol
from abc import ABC, abstractmethod
import numpy as np
import structlog

from ..shared.types import (
    State, AISignal, SignalType, ActionType, SubsystemConfig
)
from ..shared.constants import (
    SUBSYSTEM_NAMES, DEFAULT_SUBSYSTEM_WEIGHT, MIN_CONFIDENCE_THRESHOLD
)
from ..agent.confidence_manager import ConfidenceManager

logger = structlog.get_logger(__name__)


class AISubsystem(Protocol):
    """Protocol defining the interface for AI subsystems."""
    
    @abstractmethod
    async def analyze(self, state: State) -> Optional[AISignal]:
        """Analyze market state and generate signal."""
        ...
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get subsystem performance metrics."""
        ...
    
    @abstractmethod
    def update_parameters(self, params: Dict[str, Any]) -> None:
        """Update subsystem parameters."""
        ...


class SubsystemManager:
    """Manages and coordinates all AI subsystems with clean architecture."""
    
    def __init__(self, config: Dict[str, SubsystemConfig], confidence_config: Optional[Dict[str, Any]] = None):
        """Initialize subsystem manager.
        
        Args:
            config: Configuration for each subsystem
            confidence_config: Configuration for confidence manager
        """
        self.config = config
        self.subsystems: Dict[str, AISubsystem] = {}
        self.weights: Dict[str, float] = {}
        self.enabled: Dict[str, bool] = {}
        
        # Initialize confidence manager
        confidence_config = confidence_config or {}
        self.confidence_manager = ConfidenceManager(confidence_config)
        
        # Performance tracking
        self.signal_history: Dict[str, List[AISignal]] = {name: [] for name in SUBSYSTEM_NAMES}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        self.last_signals: Dict[str, Optional[AISignal]] = {}
        
        # Aggregation state
        self.consensus_threshold = 0.6
        self.conflict_resolution_method = "weighted_average"
        
        # Initialize configurations
        self._initialize_configs()
        
        enabled_names = [name for name, enabled in self.enabled.items() if enabled]
        logger.info(f"Subsystem manager initialized - {len(enabled_names)} subsystems: {', '.join(enabled_names)}")
    
    def register_subsystem(self, name: str, subsystem: AISubsystem) -> None:
        """Register an AI subsystem.
        
        Args:
            name: Subsystem name
            subsystem: Subsystem instance implementing AISubsystem protocol
        """
        if name not in SUBSYSTEM_NAMES:
            raise ValueError(f"Unknown subsystem: {name}")
        
        self.subsystems[name] = subsystem
        self.last_signals[name] = None
        
        logger.debug("Subsystem registered", name=name, type=type(subsystem).__name__)
    
    async def process_state(self, state: State) -> Dict[str, AISignal]:
        """Process market state through all enabled subsystems.
        
        Args:
            state: Current market state
            
        Returns:
            Dict[str, AISignal]: Signals from each subsystem
        """
        signals = {}
        
        # Process subsystems concurrently for performance
        tasks = []
        for name, subsystem in self.subsystems.items():
            if self.enabled.get(name, False):
                task = asyncio.create_task(self._safe_analyze(name, subsystem, state))
                tasks.append((name, task))
        
        # Collect results
        for name, task in tasks:
            try:
                signal = await task
                if signal:
                    signals[name] = signal
                    self.last_signals[name] = signal
                    self.signal_history[name].append(signal)
                    
                    # Trim history
                    if len(self.signal_history[name]) > 1000:
                        self.signal_history[name] = self.signal_history[name][-1000:]
                    
                    logger.debug("Subsystem generated signal", name=name, action=signal.action.name if hasattr(signal.action, 'name') else str(signal.action))
                else:
                    logger.debug("Subsystem returned None (no signal)", name=name)
                        
            except Exception as e:
                logger.error("Subsystem processing failed", name=name, error=str(e))
        
        enabled_names = list(self.enabled.keys())
        active_names = list(signals.keys())
        logger.info(f"Subsystems processed: {len(signals)}/{len(enabled_names)} active ({', '.join(active_names) if active_names else 'none'})")
        return signals
    
    def aggregate_signals(self, signals: Dict[str, AISignal]) -> Optional[AISignal]:
        """Aggregate subsystem signals into consensus signal.
        
        Args:
            signals: Individual subsystem signals
            
        Returns:
            Optional[AISignal]: Aggregated consensus signal or None
        """
        if not signals:
            return None
        
        try:
            # Collect weighted votes
            action_votes: Dict[ActionType, float] = {
                ActionType.HOLD: 0.0,
                ActionType.BUY: 0.0,
                ActionType.SELL: 0.0
            }
            
            total_weight = 0.0
            metadata_combined = {}
            
            for name, signal in signals.items():
                weight = self.weights.get(name, DEFAULT_SUBSYSTEM_WEIGHT)
                
                # Apply confidence weighting
                effective_weight = weight * signal.confidence
                
                action_votes[signal.action] += effective_weight
                total_weight += effective_weight
                
                # Combine metadata
                metadata_combined[f"{name}_confidence"] = signal.confidence
                metadata_combined[f"{name}_strength"] = signal.strength
            
            if total_weight == 0:
                return None
            
            # Determine consensus action
            winning_action = max(action_votes.items(), key=lambda x: x[1])[0]
            consensus_strength = action_votes[winning_action] / total_weight
            
            # Check if consensus meets threshold
            if consensus_strength < self.consensus_threshold and winning_action != ActionType.HOLD:
                winning_action = ActionType.HOLD
                consensus_strength = 0.5
            
            # Use confidence manager to calculate consensus confidence
            consensus_confidence, confidence_metrics = self.confidence_manager.calculate_consensus_confidence(
                signals, self.weights
            )
            
            # Add confidence metrics to metadata
            metadata_combined.update({
                "consensus_method": self.conflict_resolution_method,
                "participating_subsystems": list(signals.keys()),
                "total_weight": total_weight,
                "confidence_metrics": {
                    "consensus_strength": confidence_metrics.consensus_strength,
                    "subsystem_agreement": confidence_metrics.subsystem_agreement,
                    "final_confidence": confidence_metrics.final_confidence
                }
            })
            
            # Create consensus signal
            consensus_signal = AISignal(
                signal_type=SignalType.DOPAMINE,  # Manager signal type
                action=winning_action,
                confidence=consensus_confidence,
                strength=consensus_strength,
                metadata=metadata_combined,
                timestamp=max(signal.timestamp for signal in signals.values())
            )
            
            logger.info(
                "Consensus signal generated",
                action=winning_action.name,
                confidence=consensus_confidence,
                strength=consensus_strength,
                participants=len(signals),
                explanation=self.confidence_manager.get_confidence_explanation(confidence_metrics)
            )
            
            return consensus_signal
            
        except Exception as e:
            logger.error("Signal aggregation failed", error=str(e))
            return None
    
    def detect_conflicts(self, signals: Dict[str, AISignal]) -> List[str]:
        """Detect conflicts between subsystem signals.
        
        Args:
            signals: Subsystem signals to analyze
            
        Returns:
            List[str]: Descriptions of detected conflicts
        """
        conflicts = []
        
        if len(signals) < 2:
            return conflicts
        
        # Group signals by action
        action_groups: Dict[ActionType, List[str]] = {}
        for name, signal in signals.items():
            if signal.action not in action_groups:
                action_groups[signal.action] = []
            action_groups[signal.action].append(name)
        
        # Check for opposing actions with high confidence
        if (ActionType.BUY in action_groups and ActionType.SELL in action_groups):
            buy_subsystems = action_groups[ActionType.BUY]
            sell_subsystems = action_groups[ActionType.SELL]
            
            # Check confidence levels
            high_conf_buys = [name for name in buy_subsystems 
                            if signals[name].confidence > 0.7]
            high_conf_sells = [name for name in sell_subsystems 
                             if signals[name].confidence > 0.7]
            
            if high_conf_buys and high_conf_sells:
                conflicts.append(
                    f"High confidence conflict: {high_conf_buys} want BUY, "
                    f"{high_conf_sells} want SELL"
                )
        
        # Check for unusual confidence patterns
        confidences = [signal.confidence for signal in signals.values()]
        if len(confidences) > 2:
            conf_std = np.std(confidences)
            if conf_std > 0.3:  # High variance in confidence
                conflicts.append(f"High confidence variance: std={conf_std:.3f}")
        
        return conflicts
    
    def update_weights(self, performance_feedback: Dict[str, float]) -> None:
        """Update subsystem weights based on performance feedback.
        
        Args:
            performance_feedback: Performance scores for each subsystem
        """
        # Adaptive weight adjustment based on recent performance
        learning_rate = 0.05
        
        for name, performance in performance_feedback.items():
            if name in self.weights:
                current_weight = self.weights[name]
                
                # Increase weight for good performance, decrease for poor
                if performance > 0.6:  # Good performance
                    adjustment = learning_rate * (performance - 0.6)
                    new_weight = min(0.5, current_weight + adjustment)
                elif performance < 0.4:  # Poor performance
                    adjustment = learning_rate * (0.4 - performance)
                    new_weight = max(0.05, current_weight - adjustment)
                else:
                    new_weight = current_weight
                
                self.weights[name] = new_weight
        
        # Renormalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for name in self.weights:
                self.weights[name] /= total_weight
        
        logger.debug("Weights updated", weights=self.weights)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics.
        
        Returns:
            Dict[str, Any]: System performance metrics
        """
        metrics = {
            "enabled_subsystems": [name for name, enabled in self.enabled.items() if enabled],
            "weights": self.weights.copy(),
            "signal_counts": {name: len(history) for name, history in self.signal_history.items()},
            "last_signal_times": {
                name: signal.timestamp if signal else 0 
                for name, signal in self.last_signals.items()
            }
        }
        
        # Add subsystem-specific metrics
        for name, subsystem in self.subsystems.items():
            try:
                subsystem_metrics = subsystem.get_performance_metrics()
                metrics[f"{name}_metrics"] = subsystem_metrics
            except Exception as e:
                logger.error("Failed to get subsystem metrics", name=name, error=str(e))
                metrics[f"{name}_metrics"] = {"error": str(e)}
        
        return metrics
    
    async def _safe_analyze(self, name: str, subsystem: AISubsystem, state: State) -> Optional[AISignal]:
        """Safely analyze state with error handling.
        
        Args:
            name: Subsystem name
            subsystem: Subsystem instance
            state: Market state
            
        Returns:
            Optional[AISignal]: Analysis result or None if failed
        """
        try:
            return await subsystem.analyze(state)
        except Exception as e:
            logger.error("Subsystem analysis failed", name=name, error=str(e))
            return None
    
    def _initialize_configs(self) -> None:
        """Initialize subsystem configurations from config."""
        for name in SUBSYSTEM_NAMES:
            if name in self.config:
                subsystem_config = self.config[name]
                self.enabled[name] = subsystem_config.enabled
                self.weights[name] = subsystem_config.weight
            else:
                # Default configuration
                self.enabled[name] = True
                self.weights[name] = DEFAULT_SUBSYSTEM_WEIGHT
        
        # Normalize weights
        total_weight = sum(weight for name, weight in self.weights.items() if self.enabled[name])
        if total_weight > 0:
            for name in self.weights:
                if self.enabled[name]:
                    self.weights[name] /= total_weight
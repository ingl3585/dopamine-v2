"""
Centralized confidence calculation and management for the Dopamine Trading System.
Ensures consistent confidence values across all system components.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import structlog

from ..shared.types import ActionType, AISignal
from ..shared.constants import MIN_CONFIDENCE_THRESHOLD

logger = structlog.get_logger(__name__)


@dataclass
class ConfidenceMetrics:
    """Metrics used for confidence calculation."""
    consensus_strength: float
    q_value_spread: float
    subsystem_agreement: float
    exploration_penalty: float
    dopamine_boost: float
    final_confidence: float


class ConfidenceManager:
    """Centralized manager for all confidence calculations in the trading system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize confidence manager.
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        
        # Confidence calculation parameters
        self.min_confidence = config.get("min_confidence", MIN_CONFIDENCE_THRESHOLD)
        self.max_confidence = config.get("max_confidence", 0.95)
        
        # Weighting factors for different confidence components
        self.consensus_weight = config.get("consensus_weight", 0.4)
        self.q_value_weight = config.get("q_value_weight", 0.3)
        self.agreement_weight = config.get("agreement_weight", 0.2)
        self.dopamine_weight = config.get("dopamine_weight", 0.1)
        
        # Normalization parameters
        self.q_value_scale = config.get("q_value_scale", 2.0)
        self.exploration_penalty_factor = config.get("exploration_penalty_factor", 0.5)
        
        logger.info(
            "ConfidenceManager initialized",
            min_confidence=self.min_confidence,
            max_confidence=self.max_confidence,
            weights={
                "consensus": self.consensus_weight,
                "q_value": self.q_value_weight,
                "agreement": self.agreement_weight,
                "dopamine": self.dopamine_weight
            }
        )
    
    def calculate_consensus_confidence(self, signals: Dict[str, AISignal], 
                                     weights: Dict[str, float]) -> Tuple[float, ConfidenceMetrics]:
        """Calculate confidence for consensus signal from subsystem signals.
        
        Args:
            signals: Individual subsystem signals
            weights: Weights for each subsystem
            
        Returns:
            Tuple[float, ConfidenceMetrics]: Final confidence and calculation metrics
        """
        if not signals:
            return 0.0, ConfidenceMetrics(0, 0, 0, 0, 0, 0)
        
        # Calculate weighted average confidence from subsystems
        total_weight = 0.0
        weighted_confidence_sum = 0.0
        
        for name, signal in signals.items():
            weight = weights.get(name, 0.2)  # Default weight
            weighted_confidence_sum += signal.confidence * weight
            total_weight += weight
        
        avg_subsystem_confidence = weighted_confidence_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate agreement between subsystems
        action_counts = {}
        for signal in signals.values():
            action_counts[signal.action] = action_counts.get(signal.action, 0) + 1
        
        max_agreement = max(action_counts.values()) if action_counts else 0
        agreement_ratio = max_agreement / len(signals) if signals else 0
        
        # Calculate consensus strength based on weighted votes
        action_votes = {ActionType.HOLD: 0.0, ActionType.BUY: 0.0, ActionType.SELL: 0.0}
        for name, signal in signals.items():
            weight = weights.get(name, 0.2)
            effective_weight = weight * signal.confidence
            action_votes[signal.action] += effective_weight
        
        total_votes = sum(action_votes.values())
        consensus_strength = max(action_votes.values()) / total_votes if total_votes > 0 else 0.0
        
        # Combine factors into final confidence
        final_confidence = (
            self.consensus_weight * consensus_strength +
            self.agreement_weight * agreement_ratio +
            (1.0 - self.consensus_weight - self.agreement_weight) * avg_subsystem_confidence
        )
        
        # Apply bounds
        final_confidence = max(self.min_confidence, min(self.max_confidence, final_confidence))
        
        metrics = ConfidenceMetrics(
            consensus_strength=consensus_strength,
            q_value_spread=0.0,  # Not applicable for consensus
            subsystem_agreement=agreement_ratio,
            exploration_penalty=0.0,  # Not applicable for consensus
            dopamine_boost=0.0,  # Not applicable for consensus
            final_confidence=final_confidence
        )
        
        logger.debug(
            "Consensus confidence calculated",
            avg_subsystem_confidence=avg_subsystem_confidence,
            consensus_strength=consensus_strength,
            agreement_ratio=agreement_ratio,
            final_confidence=final_confidence
        )
        
        return final_confidence, metrics
    
    def calculate_rl_confidence(self, q_values, 
                               is_exploration: bool = False,
                               dopamine_boost: float = 0.0) -> Tuple[float, ConfidenceMetrics]:
        """Calculate confidence for RL agent decision based on Q-values.
        
        Args:
            q_values: Q-values for all actions
            is_exploration: Whether action was selected through exploration
            dopamine_boost: Additional confidence boost from dopamine system
            
        Returns:
            Tuple[float, ConfidenceMetrics]: Final confidence and calculation metrics
        """
        if len(q_values) == 0:
            return self.min_confidence, ConfidenceMetrics(0, 0, 0, 0, 0, self.min_confidence)
        
        # Calculate Q-value spread (higher spread = more confident decision)
        max_q = max(q_values)
        if len(q_values) > 1:
            sorted_q = sorted(q_values, reverse=True)
            second_max_q = sorted_q[1]
        else:
            second_max_q = max_q
        q_value_spread = max_q - second_max_q
        
        # Normalize Q-value confidence (map Q-values to [0, 1] range)
        # Higher max Q-value and larger spread indicate higher confidence
        q_confidence = min(1.0, max(0.0, (max_q + 1.0) / self.q_value_scale))
        spread_confidence = min(1.0, q_value_spread / 2.0)  # Normalize spread
        
        # Combine Q-value metrics
        base_q_confidence = (q_confidence + spread_confidence) / 2.0
        
        # Apply exploration penalty
        exploration_penalty = self.exploration_penalty_factor if is_exploration else 0.0
        
        # Calculate final confidence
        final_confidence = (
            base_q_confidence * (1.0 - exploration_penalty) + 
            dopamine_boost * self.dopamine_weight
        )
        
        # Apply bounds
        final_confidence = max(self.min_confidence, min(self.max_confidence, final_confidence))
        
        metrics = ConfidenceMetrics(
            consensus_strength=0.0,  # Not applicable for RL
            q_value_spread=q_value_spread,
            subsystem_agreement=0.0,  # Not applicable for RL
            exploration_penalty=exploration_penalty,
            dopamine_boost=dopamine_boost,
            final_confidence=final_confidence
        )
        
        logger.debug(
            "RL confidence calculated",
            max_q=max_q,
            q_value_spread=q_value_spread,
            base_q_confidence=base_q_confidence,
            is_exploration=is_exploration,
            dopamine_boost=dopamine_boost,
            final_confidence=final_confidence
        )
        
        return final_confidence, metrics
    
    def combine_confidences(self, consensus_confidence: float, 
                           rl_confidence: float,
                           consensus_metrics: ConfidenceMetrics,
                           rl_metrics: ConfidenceMetrics) -> Tuple[float, ConfidenceMetrics]:
        """Combine consensus and RL confidences into final trading confidence.
        
        Args:
            consensus_confidence: Confidence from subsystem consensus
            rl_confidence: Confidence from RL agent
            consensus_metrics: Metrics from consensus calculation
            rl_metrics: Metrics from RL calculation
            
        Returns:
            Tuple[float, ConfidenceMetrics]: Final combined confidence and metrics
        """
        # Weight the two confidence sources
        # Consensus gets higher weight as it represents multiple AI systems
        consensus_weight = 0.6
        rl_weight = 0.4
        
        # Calculate weighted average
        combined_confidence = (
            consensus_weight * consensus_confidence + 
            rl_weight * rl_confidence
        )
        
        # Apply conservative adjustment - both should be reasonably confident
        # If either confidence is very low, reduce the combined confidence
        min_individual = min(consensus_confidence, rl_confidence)
        if min_individual < 0.3:
            confidence_penalty = (0.3 - min_individual) * 0.5
            combined_confidence = max(0.0, combined_confidence - confidence_penalty)
        
        # Apply bounds
        final_confidence = max(self.min_confidence, min(self.max_confidence, combined_confidence))
        
        # Create combined metrics
        combined_metrics = ConfidenceMetrics(
            consensus_strength=consensus_metrics.consensus_strength,
            q_value_spread=rl_metrics.q_value_spread,
            subsystem_agreement=consensus_metrics.subsystem_agreement,
            exploration_penalty=rl_metrics.exploration_penalty,
            dopamine_boost=rl_metrics.dopamine_boost,
            final_confidence=final_confidence
        )
        
        logger.info(
            "Confidences combined",
            consensus_confidence=consensus_confidence,
            rl_confidence=rl_confidence,
            combined_confidence=combined_confidence,
            final_confidence=final_confidence,
            min_individual=min_individual
        )
        
        return final_confidence, combined_metrics
    
    def should_trade(self, confidence: float, action: ActionType) -> bool:
        """Determine if a trade should be executed based on confidence and action.
        
        Args:
            confidence: Final trading confidence
            action: Proposed trading action
            
        Returns:
            bool: Whether trade should be executed
        """
        if action == ActionType.HOLD:
            return False
        
        should_trade = confidence >= self.min_confidence
        
        logger.debug(
            "Trade decision",
            confidence=confidence,
            min_confidence=self.min_confidence,
            action=action.name,
            should_trade=should_trade
        )
        
        return should_trade
    
    def get_confidence_explanation(self, metrics: ConfidenceMetrics) -> str:
        """Generate human-readable explanation of confidence calculation.
        
        Args:
            metrics: Confidence calculation metrics
            
        Returns:
            str: Explanation of confidence factors
        """
        explanation_parts = []
        
        if metrics.consensus_strength > 0:
            explanation_parts.append(f"Consensus: {metrics.consensus_strength:.2f}")
        
        if metrics.q_value_spread > 0:
            explanation_parts.append(f"Q-spread: {metrics.q_value_spread:.2f}")
        
        if metrics.subsystem_agreement > 0:
            explanation_parts.append(f"Agreement: {metrics.subsystem_agreement:.2f}")
        
        if metrics.exploration_penalty > 0:
            explanation_parts.append(f"Exploration penalty: -{metrics.exploration_penalty:.2f}")
        
        if metrics.dopamine_boost > 0:
            explanation_parts.append(f"Dopamine boost: +{metrics.dopamine_boost:.2f}")
        
        explanation = " | ".join(explanation_parts)
        return f"Final confidence: {metrics.final_confidence:.3f} ({explanation})"
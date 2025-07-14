"""
Dopamine Subsystem for reward optimization and learning enhancement.
Implements dopamine-inspired learning mechanisms for adaptive reward processing.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import asyncio
import structlog

from ..shared.types import State, AISignal, SignalType, ActionType
from ..shared.constants import (
    DOPAMINE_DEFAULT_BASELINE, DOPAMINE_DEFAULT_SURPRISE_THRESHOLD, 
    DOPAMINE_DECAY_FACTOR, REWARD_PNL_WEIGHT, REWARD_RISK_WEIGHT
)

logger = structlog.get_logger(__name__)


class RewardPredictor:
    """Predicts expected rewards based on current state."""
    
    def __init__(self, learning_rate: float = 0.01):
        """Initialize reward predictor.
        
        Args:
            learning_rate: Learning rate for prediction updates
        """
        self.learning_rate = learning_rate
        self.prediction_model = {}  # Simple state -> reward mapping
        self.prediction_history = deque(maxlen=1000)
        self.baseline_reward = 0.0
        
    def predict_reward(self, state_features: Dict[str, float]) -> float:
        """Predict expected reward for given state.
        
        Args:
            state_features: Current state features
            
        Returns:
            float: Predicted reward
        """
        # Simple linear prediction based on key features
        prediction = self.baseline_reward
        
        # Price momentum contribution
        if "price_momentum" in state_features:
            prediction += state_features["price_momentum"] * 0.5
        
        # Volume contribution
        if "volume_ratio" in state_features:
            prediction += (state_features["volume_ratio"] - 1.0) * 0.2
        
        # Risk contribution (negative)
        if "risk_score" in state_features:
            prediction -= state_features["risk_score"] * 0.3
        
        return prediction
    
    def update_prediction(self, state_features: Dict[str, float], actual_reward: float) -> float:
        """Update prediction model with actual reward.
        
        Args:
            state_features: State features
            actual_reward: Actual reward received
            
        Returns:
            float: Prediction error
        """
        predicted_reward = self.predict_reward(state_features)
        prediction_error = actual_reward - predicted_reward
        
        # Update baseline with exponential moving average
        self.baseline_reward += self.learning_rate * prediction_error
        
        # Store prediction performance
        self.prediction_history.append({
            "predicted": predicted_reward,
            "actual": actual_reward,
            "error": prediction_error
        })
        
        return prediction_error


class SurpriseDetector:
    """Detects surprising events in market data."""
    
    def __init__(self, threshold: float = 0.1, adaptation_rate: float = 0.05):
        """Initialize surprise detector.
        
        Args:
            threshold: Threshold for surprise detection
            adaptation_rate: Rate of adaptation to new data
        """
        self.threshold = threshold
        self.adaptation_rate = adaptation_rate
        self.expectations = {}  # Feature -> expected value
        self.surprises = deque(maxlen=500)
        
    def detect_surprise(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Detect surprising events in features.
        
        Args:
            features: Current feature values
            
        Returns:
            Tuple[bool, float]: (is_surprising, surprise_magnitude)
        """
        surprise_scores = []
        
        for feature, value in features.items():
            expected = self.expectations.get(feature, value)
            
            # Calculate surprise as normalized deviation
            if feature in self.expectations:
                surprise = abs(value - expected) / max(abs(expected), 1e-8)
                surprise_scores.append(surprise)
            
            # Update expectation with exponential smoothing
            self.expectations[feature] = (
                self.expectations.get(feature, value) * (1 - self.adaptation_rate) + 
                value * self.adaptation_rate
            )
        
        if not surprise_scores:
            return False, 0.0
        
        max_surprise = max(surprise_scores)
        is_surprising = max_surprise > self.threshold
        
        if is_surprising:
            self.surprises.append({
                "magnitude": max_surprise,
                "features": features.copy()
            })
        
        return is_surprising, max_surprise


class DopamineSubsystem:
    """Dopamine-inspired reward optimization and learning enhancement."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize dopamine subsystem.
        
        Args:
            config: Subsystem configuration
        """
        self.config = config
        
        # Configuration
        self.baseline_dopamine = config.get("baseline_dopamine", DOPAMINE_DEFAULT_BASELINE)
        self.surprise_threshold = config.get("surprise_threshold", DOPAMINE_DEFAULT_SURPRISE_THRESHOLD)
        self.decay_factor = config.get("decay_factor", DOPAMINE_DECAY_FACTOR)
        
        # Components
        self.reward_predictor = RewardPredictor()
        self.surprise_detector = SurpriseDetector(self.surprise_threshold)
        
        # Dopamine state
        self.current_dopamine_level = self.baseline_dopamine
        self.dopamine_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        
        # Learning modulation
        self.learning_rate_multiplier = 1.0
        self.exploration_bonus = 0.0
        self.motivation_level = 0.5
        
        # Performance tracking
        self.total_rewards = 0.0
        self.prediction_accuracy = 0.0
        self.surprise_count = 0
        self.dopamine_peaks = deque(maxlen=100)
        
        logger.info(
            "Dopamine subsystem initialized",
            baseline=self.baseline_dopamine,
            surprise_threshold=self.surprise_threshold,
            decay_factor=self.decay_factor
        )
    
    async def analyze(self, state: State) -> Optional[AISignal]:
        """Analyze market state for dopamine-enhanced signals.
        
        Args:
            state: Current market state
            
        Returns:
            Optional[AISignal]: Dopamine-enhanced signal
        """
        try:
            # Extract features for dopamine analysis
            features = self._extract_dopamine_features(state)
            
            if not features:
                return None
            
            # Calculate current reward
            current_reward = self._calculate_current_reward(features)
            self.reward_history.append(current_reward)
            self.total_rewards += current_reward
            
            # Predict expected reward
            prediction_error = self.reward_predictor.update_prediction(features, current_reward)
            
            # Detect surprises
            is_surprising, surprise_magnitude = self.surprise_detector.detect_surprise(features)
            if is_surprising:
                self.surprise_count += 1
            
            # Update dopamine level
            self._update_dopamine_level(prediction_error, surprise_magnitude)
            
            # Generate dopamine-enhanced signal
            signal = await self._generate_dopamine_signal(features, state.timestamp)
            
            return signal
            
        except Exception as e:
            logger.error("Dopamine analysis failed", error=str(e))
            return None
    
    def _extract_dopamine_features(self, state: State) -> Dict[str, float]:
        """Extract features relevant to dopamine system."""
        features = {}
        
        # Price-based features
        if len(state.prices) > 0:
            features["current_price"] = state.prices[-1]
            
            if len(state.prices) >= 5:
                recent_prices = state.prices[-5:]
                features["price_momentum"] = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                features["price_volatility"] = np.std(recent_prices) / np.mean(recent_prices)
        
        # Volume features
        if len(state.volumes) > 0:
            features["current_volume"] = state.volumes[-1]
            
            if len(state.volumes) >= 5:
                recent_volumes = state.volumes[-5:]
                avg_volume = np.mean(recent_volumes[:-1])
                features["volume_ratio"] = recent_volumes[-1] / max(avg_volume, 1.0)
        
        # Account features
        if len(state.account_metrics) >= 5:
            features["unrealized_pnl"] = state.account_metrics[3]
            features["daily_pnl"] = state.account_metrics[2]
            features["position_size"] = state.account_metrics[8]
        
        # Risk features
        if len(state.market_conditions) >= 3:
            features["risk_score"] = state.market_conditions[0]  # Market volatility as risk proxy
            features["drawdown"] = state.market_conditions[1]
        
        return features
    
    def _calculate_current_reward(self, features: Dict[str, float]) -> float:
        """Calculate current reward based on features."""
        reward = 0.0
        
        # PnL-based reward
        if "daily_pnl" in features:
            reward += features["daily_pnl"] * REWARD_PNL_WEIGHT
        
        # Risk-adjusted reward
        if "risk_score" in features:
            risk_penalty = features["risk_score"] * REWARD_RISK_WEIGHT
            reward -= risk_penalty
        
        # Momentum reward
        if "price_momentum" in features:
            momentum_reward = abs(features["price_momentum"]) * 0.1
            reward += momentum_reward
        
        # Volume consistency reward
        if "volume_ratio" in features:
            volume_consistency = 1.0 - abs(features["volume_ratio"] - 1.0)
            reward += volume_consistency * 0.05
        
        return reward
    
    def _update_dopamine_level(self, prediction_error: float, surprise_magnitude: float):
        """Update dopamine level based on prediction error and surprise."""
        # Dopamine responds to positive prediction errors
        dopamine_signal = max(0, prediction_error) * 2.0
        
        # Add surprise bonus
        surprise_bonus = surprise_magnitude * 0.5
        dopamine_signal += surprise_bonus
        
        # Update current level with decay
        self.current_dopamine_level = (
            self.current_dopamine_level * self.decay_factor + 
            dopamine_signal * (1 - self.decay_factor)
        )
        
        # Store history
        self.dopamine_history.append(self.current_dopamine_level)
        
        # Track peaks
        if self.current_dopamine_level > self.baseline_dopamine + 0.1:
            self.dopamine_peaks.append(self.current_dopamine_level)
        
        # Update learning modulation
        self._update_learning_modulation()
    
    def _update_learning_modulation(self):
        """Update learning rate and exploration based on dopamine level."""
        # Higher dopamine = higher learning rate
        dopamine_norm = self.current_dopamine_level - self.baseline_dopamine
        self.learning_rate_multiplier = 1.0 + max(0, dopamine_norm) * 0.5
        
        # Exploration bonus based on dopamine variability
        if len(self.dopamine_history) >= 10:
            recent_dopamine = list(self.dopamine_history)[-10:]
            dopamine_var = np.std(recent_dopamine)
            self.exploration_bonus = min(0.2, dopamine_var * 0.1)
        
        # Motivation level
        self.motivation_level = 0.5 + self.current_dopamine_level * 0.3
        self.motivation_level = max(0.1, min(1.0, self.motivation_level))
    
    async def _generate_dopamine_signal(self, features: Dict[str, float], 
                                      timestamp: int) -> Optional[AISignal]:
        """Generate dopamine-enhanced trading signal."""
        # Base action determination
        action = ActionType.HOLD
        base_confidence = 0.3
        
        # Dopamine-enhanced decision making
        if self.current_dopamine_level > self.baseline_dopamine + 0.05:
            # High dopamine = more optimistic/aggressive
            if features.get("price_momentum", 0) > 0:
                action = ActionType.BUY
                base_confidence = 0.6
            elif features.get("daily_pnl", 0) > 0:
                action = ActionType.BUY
                base_confidence = 0.5
        
        elif self.current_dopamine_level < self.baseline_dopamine - 0.05:
            # Low dopamine = more conservative/defensive
            if features.get("risk_score", 0) > 0.3:
                action = ActionType.SELL
                base_confidence = 0.7
            elif features.get("drawdown", 0) > 0.02:
                action = ActionType.SELL
                base_confidence = 0.6
        
        # Modulate confidence with dopamine and motivation
        dopamine_boost = (self.current_dopamine_level - self.baseline_dopamine) * 0.5
        motivation_boost = (self.motivation_level - 0.5) * 0.2
        
        final_confidence = base_confidence + dopamine_boost + motivation_boost
        final_confidence = max(0.1, min(0.9, final_confidence))
        
        # Calculate signal strength
        strength = final_confidence * self.motivation_level
        
        # Add exploration bonus to strength
        strength += self.exploration_bonus
        strength = max(0.1, min(1.0, strength))
        
        # Metadata
        metadata = {
            "dopamine_level": self.current_dopamine_level,
            "learning_rate_multiplier": self.learning_rate_multiplier,
            "motivation_level": self.motivation_level,
            "exploration_bonus": self.exploration_bonus,
            "surprise_count": self.surprise_count,
            "prediction_accuracy": self.prediction_accuracy,
            "recent_reward": self.reward_history[-1] if self.reward_history else 0.0
        }
        
        return AISignal(
            signal_type=SignalType.DOPAMINE,
            action=action,
            confidence=final_confidence,
            strength=strength,
            metadata=metadata,
            timestamp=timestamp
        )
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get dopamine subsystem performance metrics."""
        avg_dopamine = np.mean(self.dopamine_history) if self.dopamine_history else 0.0
        dopamine_volatility = np.std(self.dopamine_history) if len(self.dopamine_history) > 1 else 0.0
        
        # Calculate prediction accuracy
        if len(self.reward_predictor.prediction_history) > 0:
            recent_predictions = list(self.reward_predictor.prediction_history)[-100:]
            errors = [abs(p["error"]) for p in recent_predictions]
            self.prediction_accuracy = 1.0 - min(1.0, np.mean(errors))
        
        return {
            "current_dopamine_level": self.current_dopamine_level,
            "avg_dopamine_level": avg_dopamine,
            "dopamine_volatility": dopamine_volatility,
            "learning_rate_multiplier": self.learning_rate_multiplier,
            "motivation_level": self.motivation_level,
            "exploration_bonus": self.exploration_bonus,
            "total_rewards": self.total_rewards,
            "prediction_accuracy": self.prediction_accuracy,
            "surprise_count": self.surprise_count,
            "dopamine_peaks": len(self.dopamine_peaks)
        }
    
    def update_parameters(self, params: Dict[str, Any]) -> None:
        """Update subsystem parameters."""
        if "surprise_threshold" in params:
            self.surprise_threshold = params["surprise_threshold"]
            self.surprise_detector.threshold = self.surprise_threshold
        
        if "decay_factor" in params:
            self.decay_factor = params["decay_factor"]
        
        if "baseline_dopamine" in params:
            self.baseline_dopamine = params["baseline_dopamine"]
        
        logger.debug("Dopamine parameters updated", params=params)
    
    def get_learning_enhancement(self) -> Dict[str, float]:
        """Get current learning enhancement parameters."""
        return {
            "learning_rate_multiplier": self.learning_rate_multiplier,
            "exploration_bonus": self.exploration_bonus,
            "motivation_level": self.motivation_level,
            "dopamine_level": self.current_dopamine_level
        }
    
    def reset_dopamine_state(self) -> None:
        """Reset dopamine state to baseline."""
        self.current_dopamine_level = self.baseline_dopamine
        self.learning_rate_multiplier = 1.0
        self.exploration_bonus = 0.0
        self.motivation_level = 0.5
        
        logger.info("Dopamine state reset to baseline")
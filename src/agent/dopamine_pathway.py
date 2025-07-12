"""
Dopamine-inspired learning pathway for the RL agent.
Implements prediction error-based learning modulation and exploration strategies.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import structlog

from ..shared.types import RewardComponents, State, ActionType
from ..shared.constants import DOPAMINE_DECAY_FACTOR

logger = structlog.get_logger(__name__)


class DopaminePathway:
    """Dopamine-inspired learning enhancement system."""
    
    def __init__(self, config: dict):
        """Initialize dopamine pathway.
        
        Args:
            config: Dopamine pathway configuration
        """
        self.config = config
        
        # Dopamine parameters
        self.baseline_reward = config.get("baseline_reward", 0.0)
        self.surprise_threshold = config.get("surprise_threshold", 0.1)
        self.decay_factor = config.get("decay_factor", DOPAMINE_DECAY_FACTOR)
        self.learning_modulation = config.get("learning_modulation", True)
        
        # Prediction tracking
        self.reward_predictions = deque(maxlen=1000)
        self.prediction_errors = deque(maxlen=1000)
        self.dopamine_signals = deque(maxlen=1000)
        
        # Adaptive baselines
        self.running_reward_mean = 0.0
        self.running_reward_std = 1.0
        self.prediction_confidence = 0.5
        
        # Learning modulation state
        self.current_dopamine = 0.0
        self.learning_rate_multiplier = 1.0
        self.exploration_bonus = 0.0
        
        # Novelty detection
        self.state_history = deque(maxlen=500)
        self.novelty_threshold = 0.8
        
        # Performance tracking
        self.prediction_accuracy = deque(maxlen=100)
        self.surprise_events = 0
        self.total_predictions = 0
        
        logger.info(
            "Dopamine pathway initialized",
            baseline_reward=self.baseline_reward,
            surprise_threshold=self.surprise_threshold
        )
    
    def predict_reward(self, state: State, action: ActionType) -> float:
        """Predict expected reward for state-action pair.
        
        Args:
            state: Current market state
            action: Proposed action
            
        Returns:
            float: Predicted reward
        """
        try:
            # Simple prediction based on recent history and state features
            if len(self.reward_predictions) < 10:
                prediction = self.baseline_reward
            else:
                # Use recent reward history as baseline
                recent_rewards = list(self.dopamine_signals)[-10:]
                prediction = np.mean(recent_rewards)
                
                # Adjust based on state features
                prediction += self._state_based_adjustment(state, action)
            
            self.reward_predictions.append(prediction)
            self.total_predictions += 1
            
            logger.debug(
                "Reward predicted",
                prediction=prediction,
                action=action.name,
                confidence=self.prediction_confidence
            )
            
            return prediction
            
        except Exception as e:
            logger.error("Failed to predict reward", error=str(e))
            return self.baseline_reward
    
    def calculate_prediction_error(self, predicted_reward: float, 
                                 actual_reward: RewardComponents) -> float:
        """Calculate prediction error (dopamine signal).
        
        Args:
            predicted_reward: Previously predicted reward
            actual_reward: Actual reward components received
            
        Returns:
            float: Prediction error (dopamine signal)
        """
        try:
            # Use total reward for prediction error
            actual = actual_reward.total_reward
            
            # Calculate raw prediction error
            prediction_error = actual - predicted_reward
            
            # Apply surprise threshold
            if abs(prediction_error) < self.surprise_threshold:
                dopamine_signal = prediction_error * 0.5  # Reduced signal for expected outcomes
            else:
                dopamine_signal = prediction_error
                self.surprise_events += 1
                logger.debug("Surprise event detected", error=prediction_error)
            
            # Store signals
            self.prediction_errors.append(prediction_error)
            self.dopamine_signals.append(dopamine_signal)
            
            # Update current dopamine level
            self.current_dopamine = dopamine_signal
            
            # Update prediction accuracy
            accuracy = 1.0 - min(abs(prediction_error), 1.0)
            self.prediction_accuracy.append(accuracy)
            
            # Update running statistics
            self._update_running_stats(actual)
            
            logger.debug(
                "Prediction error calculated",
                predicted=predicted_reward,
                actual=actual,
                error=prediction_error,
                dopamine=dopamine_signal
            )
            
            return dopamine_signal
            
        except Exception as e:
            logger.error("Failed to calculate prediction error", error=str(e))
            return 0.0
    
    def modulate_learning_rate(self, base_learning_rate: float) -> float:
        """Modulate learning rate based on dopamine signal.
        
        Args:
            base_learning_rate: Base learning rate
            
        Returns:
            float: Modulated learning rate
        """
        if not self.learning_modulation:
            return base_learning_rate
        
        # Positive dopamine increases learning, negative decreases it
        if self.current_dopamine > 0:
            # Boost learning for positive prediction errors
            multiplier = 1.0 + (self.current_dopamine * 2.0)
        else:
            # Reduce learning for negative prediction errors, but don't eliminate it
            multiplier = max(0.1, 1.0 + (self.current_dopamine * 0.5))
        
        # Apply decay to dopamine influence
        self.current_dopamine *= self.decay_factor
        
        # Store multiplier for tracking
        self.learning_rate_multiplier = multiplier
        
        modulated_rate = base_learning_rate * multiplier
        
        logger.debug(
            "Learning rate modulated",
            base_rate=base_learning_rate,
            multiplier=multiplier,
            modulated_rate=modulated_rate
        )
        
        return modulated_rate
    
    def calculate_exploration_bonus(self, state: State) -> float:
        """Calculate exploration bonus based on state novelty.
        
        Args:
            state: Current market state
            
        Returns:
            float: Exploration bonus
        """
        try:
            # Calculate state novelty
            novelty = self._calculate_state_novelty(state)
            
            # Convert novelty to exploration bonus
            if novelty > self.novelty_threshold:
                bonus = (novelty - self.novelty_threshold) * 0.1
                self.exploration_bonus = bonus
                
                logger.debug("High novelty state detected", novelty=novelty, bonus=bonus)
                return bonus
            
            self.exploration_bonus = 0.0
            return 0.0
            
        except Exception as e:
            logger.error("Failed to calculate exploration bonus", error=str(e))
            return 0.0
    
    def get_curiosity_signal(self) -> float:
        """Get curiosity-driven exploration signal.
        
        Returns:
            float: Curiosity signal for exploration
        """
        # High curiosity when prediction accuracy is low
        if len(self.prediction_accuracy) < 10:
            return 0.5
        
        recent_accuracy = np.mean(list(self.prediction_accuracy)[-10:])
        curiosity = 1.0 - recent_accuracy
        
        # Amplify curiosity when dopamine is low (boring periods)
        if abs(self.current_dopamine) < 0.1:
            curiosity *= 1.5
        
        return np.clip(curiosity, 0.0, 1.0)
    
    def _state_based_adjustment(self, state: State, action: ActionType) -> float:
        """Calculate state-based reward prediction adjustment.
        
        Args:
            state: Current market state
            action: Proposed action
            
        Returns:
            float: Prediction adjustment
        """
        adjustment = 0.0
        
        # Account balance influence
        if len(state.account_metrics) > 0:
            # Higher account balance might indicate recent success
            balance_norm = state.account_metrics[0]  # Normalized balance
            if balance_norm > 1.0:  # Above baseline
                adjustment += 0.1
            elif balance_norm < 0.8:  # Below baseline
                adjustment -= 0.1
        
        # Market conditions influence
        if len(state.market_conditions) > 0:
            volatility = state.market_conditions[0]
            # High volatility periods are less predictable
            if volatility > 0.03:
                adjustment -= 0.05
        
        # Technical indicators influence
        if len(state.technical_indicators) > 5:
            # Use momentum and volatility features
            momentum = state.technical_indicators[0]  # Short-term momentum
            vol_change = state.technical_indicators[4]  # Volatility change
            
            # Align action with momentum
            if action == ActionType.BUY and momentum > 0:
                adjustment += 0.05
            elif action == ActionType.SELL and momentum < 0:
                adjustment += 0.05
            elif action != ActionType.HOLD:
                adjustment -= 0.02  # Penalty for counter-momentum trades
        
        return np.clip(adjustment, -0.2, 0.2)
    
    def _calculate_state_novelty(self, state: State) -> float:
        """Calculate novelty of current state.
        
        Args:
            state: Current market state
            
        Returns:
            float: State novelty score
        """
        if len(self.state_history) < 10:
            self.state_history.append(state)
            return 0.5  # Medium novelty for early states
        
        # Create state signature for comparison
        current_signature = self._create_state_signature(state)
        
        # Compare with recent state history
        similarities = []
        for past_state in list(self.state_history)[-50:]:  # Compare with last 50 states
            past_signature = self._create_state_signature(past_state)
            similarity = self._calculate_signature_similarity(current_signature, past_signature)
            similarities.append(similarity)
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        novelty = 1.0 - max_similarity
        
        # Store current state
        self.state_history.append(state)
        
        return novelty
    
    def _create_state_signature(self, state: State) -> np.ndarray:
        """Create a compact signature for state comparison.
        
        Args:
            state: Market state
            
        Returns:
            np.ndarray: State signature
        """
        # Use key features for signature
        signature_features = []
        
        # Price trends (last few values from each timeframe)
        if len(state.prices) >= 10:
            signature_features.extend(state.prices[-5:])
        
        # Volume patterns
        if len(state.volumes) >= 10:
            signature_features.extend(state.volumes[-3:])
        
        # Account state
        if len(state.account_metrics) >= 5:
            signature_features.extend(state.account_metrics[:5])
        
        # Technical features
        if len(state.technical_indicators) >= 10:
            signature_features.extend(state.technical_indicators[:10])
        
        return np.array(signature_features)
    
    def _calculate_signature_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Calculate similarity between two state signatures.
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            float: Similarity score (0 to 1)
        """
        if len(sig1) != len(sig2):
            return 0.0
        
        # Use cosine similarity
        dot_product = np.dot(sig1, sig2)
        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)  # Ensure non-negative
    
    def _update_running_stats(self, reward: float) -> None:
        """Update running statistics for normalization.
        
        Args:
            reward: New reward value
        """
        # Exponential moving average for mean
        alpha = 0.01
        self.running_reward_mean = (alpha * reward + 
                                   (1 - alpha) * self.running_reward_mean)
        
        # Update standard deviation estimate
        squared_diff = (reward - self.running_reward_mean) ** 2
        self.running_reward_std = (alpha * squared_diff + 
                                  (1 - alpha) * self.running_reward_std ** 2) ** 0.5
        
        # Ensure std doesn't get too small
        self.running_reward_std = max(0.1, self.running_reward_std)
    
    def get_pathway_stats(self) -> Dict[str, float]:
        """Get dopamine pathway statistics.
        
        Returns:
            Dict[str, float]: Pathway statistics
        """
        return {
            "current_dopamine": self.current_dopamine,
            "learning_rate_multiplier": self.learning_rate_multiplier,
            "exploration_bonus": self.exploration_bonus,
            "prediction_accuracy": np.mean(self.prediction_accuracy) if self.prediction_accuracy else 0.0,
            "surprise_rate": self.surprise_events / max(self.total_predictions, 1),
            "curiosity_signal": self.get_curiosity_signal(),
            "running_reward_mean": self.running_reward_mean,
            "running_reward_std": self.running_reward_std,
            "prediction_confidence": self.prediction_confidence
        }
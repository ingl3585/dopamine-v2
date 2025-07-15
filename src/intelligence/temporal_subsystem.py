"""
Temporal Subsystem for time-based pattern detection and cycle analysis.
Implements sophisticated temporal pattern recognition without classic indicators.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import asyncio
from scipy import signal as scipy_signal
import structlog

from ..shared.types import State, AISignal, SignalType, ActionType
from ..shared.constants import TEMPORAL_DEFAULT_CYCLES, TEMPORAL_DEFAULT_LOOKBACK

logger = structlog.get_logger(__name__)


class CycleDetector:
    """Detects and tracks market cycles without traditional indicators."""
    
    def __init__(self, min_period: int = 5, max_period: int = 240):
        """Initialize cycle detector.
        
        Args:
            min_period: Minimum cycle period
            max_period: Maximum cycle period
        """
        self.min_period = min_period
        self.max_period = max_period
        self.detected_cycles: List[Dict[str, float]] = []
        self.cycle_confidence = 0.0
    
    def detect_cycles(self, data: np.ndarray) -> List[Dict[str, float]]:
        """Detect cycles in time series data.
        
        Args:
            data: Time series data
            
        Returns:
            List[Dict[str, float]]: Detected cycles with periods and strengths
        """
        if len(data) < self.max_period * 2:
            return []
        
        cycles = []
        
        # Use autocorrelation to detect cycles
        autocorr = self._calculate_autocorrelation(data)
        
        # Find peaks in autocorrelation
        peaks, _ = scipy_signal.find_peaks(
            autocorr[self.min_period:self.max_period], 
            height=0.3, 
            distance=self.min_period
        )
        
        for peak in peaks:
            period = peak + self.min_period
            strength = autocorr[period]
            
            cycles.append({
                "period": period,
                "strength": strength,
                "confidence": min(1.0, strength * 2.0)
            })
        
        # Sort by strength
        cycles.sort(key=lambda x: x["strength"], reverse=True)
        
        self.detected_cycles = cycles[:5]  # Keep top 5 cycles
        return self.detected_cycles
    
    def _calculate_autocorrelation(self, data: np.ndarray) -> np.ndarray:
        """Calculate autocorrelation function."""
        n = len(data)
        data_normalized = (data - np.mean(data)) / np.std(data)
        
        autocorr = np.correlate(data_normalized, data_normalized, mode='full')
        autocorr = autocorr[n-1:]  # Take only positive lags
        autocorr = autocorr / autocorr[0]  # Normalize
        
        return autocorr


class TemporalPatternMatcher:
    """Matches temporal patterns in market data."""
    
    def __init__(self, pattern_length: int = 10):  # Reduced from 20
        """Initialize pattern matcher.
        
        Args:
            pattern_length: Length of patterns to match
        """
        self.pattern_length = pattern_length
        self.reference_patterns: List[Tuple[np.ndarray, ActionType, float]] = []
        self.pattern_scores = deque(maxlen=100)
    
    def add_reference_pattern(self, pattern: np.ndarray, action: ActionType, performance: float):
        """Add a reference pattern with performance feedback.
        
        Args:
            pattern: Pattern data
            action: Associated action
            performance: Pattern performance score
        """
        self.reference_patterns.append((pattern, action, performance))
        
        # Keep only best performing patterns
        if len(self.reference_patterns) > 50:
            self.reference_patterns.sort(key=lambda x: x[2], reverse=True)
            self.reference_patterns = self.reference_patterns[:50]
    
    def find_matches(self, current_pattern: np.ndarray) -> List[Tuple[ActionType, float]]:
        """Find matching patterns and their actions.
        
        Args:
            current_pattern: Current pattern to match
            
        Returns:
            List[Tuple[ActionType, float]]: Matched actions and similarities
        """
        matches = []
        
        for ref_pattern, action, performance in self.reference_patterns:
            similarity = self._calculate_pattern_similarity(current_pattern, ref_pattern)
            
            if similarity > 0.7:  # High similarity threshold
                confidence = similarity * performance
                matches.append((action, confidence))
        
        return matches
    
    def _calculate_pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between patterns."""
        if len(pattern1) != len(pattern2):
            return 0.0
        
        # Dynamic Time Warping similarity (simplified)
        correlation = np.corrcoef(pattern1, pattern2)[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        
        return max(0.0, correlation)


class TemporalSubsystem:
    """Temporal pattern recognition and cycle analysis subsystem."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize temporal subsystem.
        
        Args:
            config: Subsystem configuration
        """
        self.config = config
        
        # Configuration
        self.cycle_lengths = config.get("cycle_lengths", TEMPORAL_DEFAULT_CYCLES)
        self.lookback_periods = config.get("lookback_periods", TEMPORAL_DEFAULT_LOOKBACK)
        self.min_cycle_confidence = config.get("min_cycle_confidence", 0.4)
        
        # Components
        self.cycle_detector = CycleDetector(min(self.cycle_lengths), max(self.cycle_lengths))
        self.pattern_matcher = TemporalPatternMatcher()
        
        # Data storage
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        self.timestamp_history = deque(maxlen=1000)
        
        # Pattern analysis
        self.seasonal_patterns: Dict[str, float] = {}
        self.time_of_day_patterns: Dict[int, Dict[str, float]] = {}
        
        # Performance tracking
        self.total_signals = 0
        self.successful_predictions = 0
        self.cycle_accuracy = deque(maxlen=100)
        
        logger.info(f"Temporal subsystem initialized - cycles: {self.cycle_lengths}, lookback: {self.lookback_periods}")
    
    async def analyze(self, state: State) -> Optional[AISignal]:
        """Analyze temporal patterns in market state.
        
        Args:
            state: Current market state
            
        Returns:
            Optional[AISignal]: Temporal analysis signal
        """
        try:
            # Extract and store temporal data
            self._update_history(state)
            
            if len(self.price_history) < self.lookback_periods:
                return None
            
            # Multi-timeframe analysis
            signals = []
            
            # Cycle analysis
            cycle_signal = await self._analyze_cycles()
            if cycle_signal:
                signals.append(cycle_signal)
            
            # Pattern matching
            pattern_signal = await self._analyze_patterns()
            if pattern_signal:
                signals.append(pattern_signal)
            
            # Time-of-day analysis
            time_signal = await self._analyze_time_patterns(state.timestamp)
            if time_signal:
                signals.append(time_signal)
            
            # Seasonal analysis (weekly/monthly patterns)
            seasonal_signal = await self._analyze_seasonal_patterns(state.timestamp)
            if seasonal_signal:
                signals.append(seasonal_signal)
            
            # Combine signals
            if signals:
                combined_signal = self._combine_temporal_signals(signals, state.timestamp)
                self.total_signals += 1
                return combined_signal
            
            # Generate occasional test signal when enough data is available
            if len(self.price_history) >= 15 and len(self.price_history) % 25 == 0:
                # Generate a low-confidence signal to stay active
                action = ActionType.HOLD  # Conservative default
                return AISignal(
                    signal_type=SignalType.TEMPORAL,
                    action=action,
                    confidence=0.35,  # Low confidence
                    strength=0.15,    # Low strength
                    metadata={"type": "temporal_test", "data_points": len(self.price_history)},
                    timestamp=state.timestamp
                )
            
            return None
            
        except Exception as e:
            logger.error("Temporal analysis failed", error=str(e))
            return None
    
    async def _analyze_cycles(self) -> Optional[Dict[str, Any]]:
        """Analyze market cycles."""
        if len(self.price_history) < max(self.cycle_lengths) * 2:
            return None
        
        price_data = np.array(list(self.price_history))
        
        # Detect cycles
        cycles = self.cycle_detector.detect_cycles(price_data)
        
        if not cycles:
            return None
        
        # Analyze current position in dominant cycle
        dominant_cycle = cycles[0]
        period = int(dominant_cycle["period"])
        
        if len(price_data) >= period:
            # Determine cycle phase
            cycle_phase = len(price_data) % period
            cycle_position = cycle_phase / period  # 0 to 1
            
            # Predict next direction based on cycle
            if 0.2 <= cycle_position <= 0.8:  # Middle of cycle
                if cycle_position < 0.5:
                    direction = ActionType.BUY  # Upward phase
                else:
                    direction = ActionType.SELL  # Downward phase
                
                confidence = dominant_cycle["strength"] * dominant_cycle["confidence"]
                
                return {
                    "action": direction,
                    "confidence": confidence,
                    "source": "cycle_analysis",
                    "cycle_period": period,
                    "cycle_phase": cycle_position
                }
        
        return None
    
    async def _analyze_patterns(self) -> Optional[Dict[str, Any]]:
        """Analyze temporal patterns."""
        if len(self.price_history) < self.pattern_matcher.pattern_length:
            return None
        
        # Extract current pattern
        current_pattern = np.array(list(self.price_history)[-self.pattern_matcher.pattern_length:])
        
        # Normalize pattern
        pattern_normalized = (current_pattern - np.mean(current_pattern)) / (np.std(current_pattern) + 1e-8)
        
        # Find matches
        matches = self.pattern_matcher.find_matches(pattern_normalized)
        
        if not matches:
            return None
        
        # Vote on action
        action_votes = {ActionType.HOLD: 0.0, ActionType.BUY: 0.0, ActionType.SELL: 0.0}
        total_confidence = 0.0
        
        for action, confidence in matches:
            action_votes[action] += confidence
            total_confidence += confidence
        
        if total_confidence > 0:
            best_action = max(action_votes.items(), key=lambda x: x[1])[0]
            confidence = action_votes[best_action] / total_confidence
            
            if confidence > 0.5:  # Minimum confidence threshold
                return {
                    "action": best_action,
                    "confidence": confidence,
                    "source": "pattern_matching",
                    "match_count": len(matches)
                }
        
        return None
    
    async def _analyze_time_patterns(self, timestamp: int) -> Optional[Dict[str, Any]]:
        """Analyze time-of-day patterns."""
        try:
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamp / 1e7 - 62135596800)  # .NET ticks to datetime
            hour = dt.hour
            
            # Check if we have learned patterns for this hour
            if hour in self.time_of_day_patterns:
                patterns = self.time_of_day_patterns[hour]
                
                if patterns.get("confidence", 0) > 0.3:
                    action_str = patterns.get("dominant_action", "HOLD")
                    action = ActionType[action_str] if action_str in ["BUY", "SELL", "HOLD"] else ActionType.HOLD
                    
                    return {
                        "action": action,
                        "confidence": patterns["confidence"],
                        "source": "time_of_day",
                        "hour": hour
                    }
            
            return None
            
        except Exception:
            return None
    
    async def _analyze_seasonal_patterns(self, timestamp: int) -> Optional[Dict[str, Any]]:
        """Analyze seasonal (weekly/monthly) patterns."""
        try:
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamp / 1e7 - 62135596800)
            day_of_week = dt.weekday()  # 0 = Monday
            
            # Simple day-of-week analysis
            seasonal_key = f"dow_{day_of_week}"
            
            if seasonal_key in self.seasonal_patterns:
                pattern_strength = self.seasonal_patterns[seasonal_key]
                
                if abs(pattern_strength) > 0.2:  # Significant pattern
                    action = ActionType.BUY if pattern_strength > 0 else ActionType.SELL
                    confidence = min(abs(pattern_strength), 0.6)  # Cap confidence
                    
                    return {
                        "action": action,
                        "confidence": confidence,
                        "source": "seasonal",
                        "day_of_week": day_of_week
                    }
            
            return None
            
        except Exception:
            return None
    
    def _combine_temporal_signals(self, signals: List[Dict[str, Any]], timestamp: int) -> AISignal:
        """Combine multiple temporal signals."""
        # Weight signals by source reliability
        source_weights = {
            "cycle_analysis": 0.4,
            "pattern_matching": 0.3,
            "time_of_day": 0.2,
            "seasonal": 0.1
        }
        
        action_votes = {ActionType.HOLD: 0.0, ActionType.BUY: 0.0, ActionType.SELL: 0.0}
        total_weight = 0.0
        metadata = {}
        
        for signal in signals:
            source = signal["source"]
            weight = source_weights.get(source, 0.1)
            confidence = signal["confidence"]
            
            effective_weight = weight * confidence
            action_votes[signal["action"]] += effective_weight
            total_weight += effective_weight
            
            # Collect metadata
            metadata[f"{source}_confidence"] = confidence
            metadata.update({k: v for k, v in signal.items() if k not in ["action", "confidence", "source"]})
        
        # Determine final action
        if total_weight > 0:
            best_action = max(action_votes.items(), key=lambda x: x[1])[0]
            final_confidence = action_votes[best_action] / total_weight
        else:
            best_action = ActionType.HOLD
            final_confidence = 0.1
        
        # Calculate strength based on agreement
        agreement = len([s for s in signals if s["action"] == best_action]) / len(signals)
        strength = agreement * final_confidence
        
        metadata["signal_count"] = len(signals)
        metadata["agreement_ratio"] = agreement
        
        return AISignal(
            signal_type=SignalType.TEMPORAL,
            action=best_action,
            confidence=final_confidence,
            strength=strength,
            metadata=metadata,
            timestamp=timestamp
        )
    
    def _update_history(self, state: State) -> None:
        """Update temporal data history."""
        # Extract representative price (use last price from longest timeframe)
        if len(state.prices) > 0:
            self.price_history.append(state.prices[-1])
        
        # Extract representative volume
        if len(state.volumes) > 0:
            self.volume_history.append(state.volumes[-1])
        
        self.timestamp_history.append(state.timestamp if hasattr(state, 'timestamp') else 0)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get temporal subsystem performance metrics."""
        success_rate = self.successful_predictions / max(self.total_signals, 1)
        avg_cycle_accuracy = np.mean(self.cycle_accuracy) if self.cycle_accuracy else 0.0
        
        return {
            "total_signals": self.total_signals,
            "success_rate": success_rate,
            "cycle_accuracy": avg_cycle_accuracy,
            "detected_cycles": len(self.cycle_detector.detected_cycles),
            "pattern_library_size": len(self.pattern_matcher.reference_patterns),
            "seasonal_patterns": len(self.seasonal_patterns),
            "time_patterns": len(self.time_of_day_patterns)
        }
    
    def update_parameters(self, params: Dict[str, Any]) -> None:
        """Update subsystem parameters."""
        if "min_cycle_confidence" in params:
            self.min_cycle_confidence = params["min_cycle_confidence"]
        
        if "lookback_periods" in params:
            self.lookback_periods = params["lookback_periods"]
        
        logger.debug("Temporal parameters updated", params=params)
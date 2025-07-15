"""
Immune Subsystem for risk assessment and anomaly detection.
Implements adaptive immune system principles for market threat detection.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import deque
import asyncio
import structlog

from ..shared.types import State, AISignal, SignalType, ActionType
from ..shared.constants import (
    IMMUNE_DEFAULT_THRESHOLD, IMMUNE_DEFAULT_ADAPTATION_RATE, IMMUNE_MEMORY_SIZE
)

logger = structlog.get_logger(__name__)


class AnomalyDetector:
    """Detects market anomalies using statistical methods."""
    
    def __init__(self, threshold: float = 2.0, memory_size: int = 500):
        """Initialize anomaly detector.
        
        Args:
            threshold: Standard deviation threshold for anomalies
            memory_size: Size of historical data memory
        """
        self.threshold = threshold
        self.memory = deque(maxlen=memory_size)
        self.running_stats = {"mean": 0.0, "std": 1.0, "count": 0}
    
    def detect_anomaly(self, value: float) -> Tuple[bool, float]:
        """Detect if value is anomalous.
        
        Args:
            value: Value to check
            
        Returns:
            Tuple[bool, float]: (is_anomaly, severity)
        """
        self.memory.append(value)
        self._update_stats(value)
        
        # Calculate z-score
        z_score = abs(value - self.running_stats["mean"]) / max(self.running_stats["std"], 1e-8)
        
        is_anomaly = z_score > self.threshold
        severity = min(z_score / self.threshold, 3.0)  # Cap at 3x threshold
        
        return is_anomaly, severity
    
    def _update_stats(self, value: float) -> None:
        """Update running statistics."""
        self.running_stats["count"] += 1
        
        # Exponential moving average
        alpha = min(0.1, 2.0 / self.running_stats["count"])
        
        old_mean = self.running_stats["mean"]
        self.running_stats["mean"] += alpha * (value - old_mean)
        
        # Update variance estimate
        variance = (1 - alpha) * (self.running_stats["std"] ** 2) + alpha * ((value - old_mean) ** 2)
        self.running_stats["std"] = max(np.sqrt(variance), 0.01)


class RiskPatternMatcher:
    """Matches current conditions against known risky patterns."""
    
    def __init__(self):
        """Initialize risk pattern matcher."""
        self.risk_patterns: List[Dict[str, Any]] = []
        self.pattern_hits = deque(maxlen=100)
    
    def add_risk_pattern(self, pattern: Dict[str, Any], severity: float) -> None:
        """Add a risk pattern to memory.
        
        Args:
            pattern: Risk pattern features
            severity: Pattern severity (0-1)
        """
        pattern["severity"] = severity
        pattern["hits"] = 0
        pattern["last_seen"] = 0
        self.risk_patterns.append(pattern)
        
        # Keep only most severe patterns
        if len(self.risk_patterns) > 50:
            self.risk_patterns.sort(key=lambda x: x["severity"], reverse=True)
            self.risk_patterns = self.risk_patterns[:50]
    
    def match_patterns(self, current_features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Match current features against risk patterns.
        
        Args:
            current_features: Current market features
            
        Returns:
            List[Dict[str, Any]]: Matched risk patterns
        """
        matches = []
        
        for pattern in self.risk_patterns:
            similarity = self._calculate_pattern_similarity(current_features, pattern)
            
            if similarity > 0.7:  # High similarity threshold
                pattern["hits"] += 1
                matches.append({
                    "pattern": pattern,
                    "similarity": similarity,
                    "severity": pattern["severity"] * similarity
                })
        
        return sorted(matches, key=lambda x: x["severity"], reverse=True)
    
    def _calculate_pattern_similarity(self, features1: Dict[str, float], 
                                    features2: Dict[str, Any]) -> float:
        """Calculate similarity between feature sets."""
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            if isinstance(features2[key], (int, float)):
                # Normalized absolute difference
                diff = abs(features1[key] - features2[key])
                max_val = max(abs(features1[key]), abs(features2[key]), 1.0)
                similarity = 1.0 - min(diff / max_val, 1.0)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0


class ImmuneSubsystem:
    """Immune system-inspired risk assessment and anomaly detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize immune subsystem.
        
        Args:
            config: Subsystem configuration
        """
        self.config = config
        
        # Configuration
        self.anomaly_threshold = config.get("anomaly_threshold", IMMUNE_DEFAULT_THRESHOLD)
        self.adaptation_rate = config.get("adaptation_rate", IMMUNE_DEFAULT_ADAPTATION_RATE)
        self.memory_size = config.get("memory_size", IMMUNE_MEMORY_SIZE)
        
        # Components
        self.price_anomaly_detector = AnomalyDetector(self.anomaly_threshold, self.memory_size)
        self.volume_anomaly_detector = AnomalyDetector(self.anomaly_threshold, self.memory_size)
        self.volatility_anomaly_detector = AnomalyDetector(self.anomaly_threshold, self.memory_size)
        self.risk_pattern_matcher = RiskPatternMatcher()
        
        # Risk assessment
        self.current_risk_level = 0.0
        self.risk_history = deque(maxlen=100)
        self.threat_memory: Set[str] = set()
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            "price_volatility": 0.02,
            "volume_spike": 2.0,
            "drawdown_warning": 0.03,
            "position_risk": 0.5
        }
        
        # Performance tracking
        self.total_alerts = 0
        self.true_positives = 0
        self.false_positives = 0
        
        logger.info(
            "Immune subsystem initialized",
            anomaly_threshold=self.anomaly_threshold,
            memory_size=self.memory_size
        )
    
    async def analyze(self, state: State) -> Optional[AISignal]:
        """Analyze market state for risks and anomalies.
        
        Args:
            state: Current market state
            
        Returns:
            Optional[AISignal]: Risk assessment signal
        """
        try:
            # Extract risk features
            risk_features = self._extract_risk_features(state)
            
            # Multi-layer anomaly detection
            anomalies = []
            
            # Price anomaly detection
            price_anomaly = await self._detect_price_anomalies(risk_features)
            if price_anomaly:
                anomalies.append(price_anomaly)
            
            # Volume anomaly detection
            volume_anomaly = await self._detect_volume_anomalies(risk_features)
            if volume_anomaly:
                anomalies.append(volume_anomaly)
            
            # Market stress detection
            stress_signal = await self._detect_market_stress(risk_features)
            if stress_signal:
                anomalies.append(stress_signal)
            
            # Risk pattern matching
            pattern_risks = await self._match_risk_patterns(risk_features)
            anomalies.extend(pattern_risks)
            
            # Portfolio risk assessment
            portfolio_risk = await self._assess_portfolio_risk(state)
            if portfolio_risk:
                anomalies.append(portfolio_risk)
            
            # Generate combined risk signal
            if anomalies:
                risk_signal = self._generate_risk_signal(anomalies, state.timestamp)
                self.total_alerts += 1
                return risk_signal
            
            # Update normal market conditions
            self._update_baseline(risk_features)
            
            # Occasionally generate a neutral signal to stay active
            if len(self.risk_history) > 0 and len(self.risk_history) % 20 == 0:
                # Generate a low-confidence HOLD signal to indicate system is working
                return AISignal(
                    signal_type=SignalType.IMMUNE,
                    action=ActionType.HOLD,
                    confidence=0.3,  # Low confidence
                    strength=0.1,    # Low strength
                    metadata={"type": "system_health", "anomalies_detected": 0},
                    timestamp=state.timestamp
                )
            
            return None
            
        except Exception as e:
            logger.error("Immune analysis failed", error=str(e))
            return None
    
    def _extract_risk_features(self, state: State) -> Dict[str, float]:
        """Extract risk-relevant features from market state."""
        features = {}
        
        # Price-based risk features
        if len(state.prices) > 0:
            features["current_price"] = state.prices[-1] if len(state.prices) > 0 else 0.0
            
            if len(state.prices) >= 10:
                recent_prices = state.prices[-10:]
                features["price_volatility"] = np.std(recent_prices) / max(np.mean(recent_prices), 1e-8)
                features["price_momentum"] = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Volume-based risk features
        if len(state.volumes) > 0:
            features["current_volume"] = state.volumes[-1] if len(state.volumes) > 0 else 0.0
            
            if len(state.volumes) >= 10:
                recent_volumes = state.volumes[-10:]
                avg_volume = np.mean(recent_volumes[:-1])
                features["volume_spike"] = recent_volumes[-1] / max(avg_volume, 1.0)
        
        # Account-based risk features
        if len(state.account_metrics) >= 5:
            features["unrealized_pnl"] = state.account_metrics[3]  # Normalized unrealized P&L
            features["daily_pnl"] = state.account_metrics[2]  # Normalized daily P&L
            
        if len(state.account_metrics) >= 9:
            features["position_size"] = state.account_metrics[8]  # Normalized position size
        else:
            features["position_size"] = 0.0  # Default when position size not available
        
        # Market condition features
        if len(state.market_conditions) >= 3:
            features["market_volatility"] = state.market_conditions[0]
            features["drawdown_pct"] = state.market_conditions[1]
            features["portfolio_heat"] = state.market_conditions[2]
        
        return features
    
    async def _detect_price_anomalies(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Detect price-related anomalies."""
        if "price_volatility" not in features:
            return None
        
        is_anomaly, severity = self.price_anomaly_detector.detect_anomaly(features["price_volatility"])
        
        if is_anomaly and severity > 1.5:
            return {
                "type": "price_anomaly",
                "severity": severity,
                "feature": "price_volatility",
                "value": features["price_volatility"],
                "action_suggestion": ActionType.HOLD  # Conservative during anomalies
            }
        
        return None
    
    async def _detect_volume_anomalies(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Detect volume-related anomalies."""
        if "volume_spike" not in features:
            return None
        
        is_anomaly, severity = self.volume_anomaly_detector.detect_anomaly(features["volume_spike"])
        
        if is_anomaly and features["volume_spike"] > 3.0:  # 3x normal volume
            return {
                "type": "volume_anomaly",
                "severity": severity,
                "feature": "volume_spike",
                "value": features["volume_spike"],
                "action_suggestion": ActionType.HOLD  # Wait and see during volume spikes
            }
        
        return None
    
    async def _detect_market_stress(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Detect market stress conditions."""
        stress_score = 0.0
        stress_factors = []
        
        # High volatility stress
        if features.get("market_volatility", 0) > 0.05:
            stress_score += 0.3
            stress_factors.append("high_volatility")
        
        # Drawdown stress
        if features.get("drawdown_pct", 0) > 0.02:
            stress_score += 0.4
            stress_factors.append("drawdown")
        
        # Portfolio heat stress
        if features.get("portfolio_heat", 0) > 0.7:
            stress_score += 0.3
            stress_factors.append("portfolio_heat")
        
        if stress_score > 0.5:
            return {
                "type": "market_stress",
                "severity": min(stress_score, 1.0),
                "factors": stress_factors,
                "action_suggestion": ActionType.SELL if stress_score > 0.8 else ActionType.HOLD
            }
        
        return None
    
    async def _match_risk_patterns(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Match against known risk patterns."""
        matches = self.risk_pattern_matcher.match_patterns(features)
        
        risk_signals = []
        for match in matches[:3]:  # Top 3 matches
            if match["severity"] > 0.6:
                risk_signals.append({
                    "type": "risk_pattern",
                    "severity": match["severity"],
                    "pattern_type": match["pattern"].get("type", "unknown"),
                    "action_suggestion": ActionType.SELL if match["severity"] > 0.8 else ActionType.HOLD
                })
        
        return risk_signals
    
    async def _assess_portfolio_risk(self, state: State) -> Optional[Dict[str, Any]]:
        """Assess portfolio-level risks."""
        if len(state.account_metrics) < 5:
            return None
        
        # Calculate portfolio risk metrics
        daily_pnl = state.account_metrics[2]  # Normalized daily P&L
        position_size = state.account_metrics[8]  # Normalized position size
        
        risk_score = 0.0
        risk_factors = []
        
        # Large negative P&L
        if daily_pnl < -0.3:  # 30% of normalized range
            risk_score += 0.4
            risk_factors.append("large_loss")
        
        # Oversized positions
        if position_size > 0.8:  # 80% of max position
            risk_score += 0.3
            risk_factors.append("large_position")
        
        if risk_score > 0.4:
            return {
                "type": "portfolio_risk",
                "severity": min(risk_score, 1.0),
                "factors": risk_factors,
                "action_suggestion": ActionType.SELL if risk_score > 0.7 else ActionType.HOLD
            }
        
        return None
    
    def _generate_risk_signal(self, anomalies: List[Dict[str, Any]], timestamp: int) -> AISignal:
        """Generate combined risk signal from detected anomalies."""
        # Calculate overall risk level
        max_severity = max(anomaly["severity"] for anomaly in anomalies)
        avg_severity = np.mean([anomaly["severity"] for anomaly in anomalies])
        
        # Determine action based on risk consensus
        action_votes = {ActionType.HOLD: 0, ActionType.BUY: 0, ActionType.SELL: 0}
        
        for anomaly in anomalies:
            action = anomaly.get("action_suggestion", ActionType.HOLD)
            weight = anomaly["severity"]
            action_votes[action] += weight
        
        # Select action with highest weight
        consensus_action = max(action_votes.items(), key=lambda x: x[1])[0]
        
        # Risk signals tend to be conservative/defensive
        if consensus_action == ActionType.BUY and max_severity > 0.7:
            consensus_action = ActionType.HOLD  # Override risky buy signals
        
        # Calculate confidence based on severity and consensus
        confidence = min(avg_severity, 0.9)  # Cap confidence for risk signals
        strength = max_severity
        
        # Update risk level
        self.current_risk_level = max_severity
        self.risk_history.append(max_severity)
        
        # Metadata
        metadata = {
            "anomaly_count": len(anomalies),
            "max_severity": max_severity,
            "avg_severity": avg_severity,
            "risk_types": [a["type"] for a in anomalies],
            "current_risk_level": self.current_risk_level
        }
        
        return AISignal(
            signal_type=SignalType.IMMUNE,
            action=consensus_action,
            confidence=confidence,
            strength=strength,
            metadata=metadata,
            timestamp=timestamp
        )
    
    def _update_baseline(self, features: Dict[str, float]) -> None:
        """Update baseline expectations during normal conditions."""
        # Adapt thresholds based on recent market conditions
        for key, detector in [
            ("price_volatility", self.price_anomaly_detector),
            ("volume_spike", self.volume_anomaly_detector)
        ]:
            if key in features:
                detector.detect_anomaly(features[key])  # Updates internal stats
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get immune subsystem performance metrics."""
        accuracy = self.true_positives / max(self.total_alerts, 1)
        false_positive_rate = self.false_positives / max(self.total_alerts, 1)
        avg_risk_level = np.mean(self.risk_history) if self.risk_history else 0.0
        
        return {
            "total_alerts": self.total_alerts,
            "accuracy": accuracy,
            "false_positive_rate": false_positive_rate,
            "avg_risk_level": avg_risk_level,
            "current_risk_level": self.current_risk_level,
            "risk_patterns": len(self.risk_pattern_matcher.risk_patterns),
            "threat_memory_size": len(self.threat_memory)
        }
    
    def update_parameters(self, params: Dict[str, Any]) -> None:
        """Update subsystem parameters."""
        if "anomaly_threshold" in params:
            self.anomaly_threshold = params["anomaly_threshold"]
            # Update detector thresholds
            self.price_anomaly_detector.threshold = self.anomaly_threshold
            self.volume_anomaly_detector.threshold = self.anomaly_threshold
        
        if "adaptation_rate" in params:
            self.adaptation_rate = params["adaptation_rate"]
        
        logger.debug("Immune parameters updated", params=params)
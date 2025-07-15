"""
Microstructure Subsystem for market regime analysis and order flow detection.
Implements sophisticated market microstructure analysis without classic indicators.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import asyncio
from sklearn.cluster import KMeans
import structlog

from ..shared.types import State, AISignal, SignalType, ActionType
from ..shared.constants import (
    MICROSTRUCTURE_DEFAULT_WINDOW, MICROSTRUCTURE_DEFAULT_VOL_THRESHOLD,
    MICROSTRUCTURE_REGIME_TYPES
)

logger = structlog.get_logger(__name__)


class RegimeDetector:
    """Detects market regimes based on price action and volume patterns."""
    
    def __init__(self, window_size: int = 50):
        """Initialize regime detector.
        
        Args:
            window_size: Window size for regime analysis
        """
        self.window_size = window_size
        self.regimes = ["trending", "ranging", "volatile", "calm"]
        self.current_regime = "ranging"
        self.regime_confidence = 0.5
        self.regime_history = deque(maxlen=200)
        
        # Clustering for regime detection
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.feature_history = deque(maxlen=500)
    
    def detect_regime(self, prices: np.ndarray, volumes: np.ndarray) -> Tuple[str, float]:
        """Detect current market regime.
        
        Args:
            prices: Price array
            volumes: Volume array
            
        Returns:
            Tuple[str, float]: (regime, confidence)
        """
        if len(prices) < self.window_size:
            return self.current_regime, self.regime_confidence
        
        # Extract regime features
        features = self._extract_regime_features(prices, volumes)
        self.feature_history.append(features)
        
        if len(self.feature_history) < 8:  # Reduced from 20 to 8
            return self.current_regime, self.regime_confidence
        
        # Use recent feature history for clustering
        recent_features = np.array(list(self.feature_history)[-100:])
        
        # Clean NaN values from feature history
        recent_features = np.nan_to_num(recent_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            # Fit clustering model
            self.kmeans.fit(recent_features)
            
            # Predict current regime
            current_cluster = self.kmeans.predict([features])[0]
            
            # Map cluster to regime type
            regime, confidence = self._map_cluster_to_regime(current_cluster, features)
            
            self.current_regime = regime
            self.regime_confidence = confidence
            self.regime_history.append((regime, confidence))
            
            return regime, confidence
            
        except Exception as e:
            logger.error("Regime detection failed", error=str(e))
            return self.current_regime, self.regime_confidence
    
    def _extract_regime_features(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Extract features for regime classification."""
        features = []
        
        # Price-based features
        if len(prices) >= 20:
            returns = np.diff(prices) / prices[:-1]
            
            # Volatility (without traditional indicators)
            features.append(np.std(returns[-20:]))
            
            # Trend strength (directional persistence)
            trend_consistency = np.mean(np.sign(returns[-20:]))
            features.append(abs(trend_consistency))
            
            # Price range expansion/contraction
            recent_range = np.max(prices[-20:]) - np.min(prices[-20:])
            avg_range = np.mean([np.max(prices[i:i+10]) - np.min(prices[i:i+10]) 
                               for i in range(len(prices)-30, len(prices)-20)])
            range_expansion = recent_range / max(avg_range, 1e-8)
            features.append(range_expansion)
            
            # Price acceleration (second derivative)
            if len(returns) >= 2:
                acceleration = np.std(np.diff(returns[-10:]))
                features.append(acceleration)
        
        # Volume-based features  
        if len(volumes) >= 20:
            # Volume consistency
            vol_std = np.std(volumes[-20:]) / max(np.mean(volumes[-20:]), 1.0)
            features.append(vol_std)
            
            # Volume trend
            vol_trend = (volumes[-1] - volumes[-20]) / max(volumes[-20], 1.0)
            features.append(vol_trend)
        
        # Pad features if necessary
        while len(features) < 6:
            features.append(0.0)
        
        # Replace any NaN or inf values with 0.0
        features_array = np.array(features[:6])
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_array
    
    def _map_cluster_to_regime(self, cluster: int, features: np.ndarray) -> Tuple[str, float]:
        """Map cluster to regime type based on features."""
        volatility = features[0] if len(features) > 0 else 0.0
        trend_strength = features[1] if len(features) > 1 else 0.0
        range_expansion = features[2] if len(features) > 2 else 1.0
        
        # Simple rule-based mapping
        confidence = 0.7  # Base confidence
        
        if volatility > 0.03:  # High volatility
            if trend_strength > 0.6:
                return "trending", confidence
            else:
                return "volatile", confidence
        else:  # Low volatility
            if trend_strength > 0.4:
                return "trending", confidence * 0.8  # Lower confidence
            else:
                return "ranging" if range_expansion < 1.5 else "calm", confidence
    
    def get_regime_transition_probability(self) -> Dict[str, float]:
        """Get probability of regime transitions."""
        if len(self.regime_history) < 10:
            return {regime: 0.25 for regime in self.regimes}
        
        # Analyze recent regime stability
        recent_regimes = [r[0] for r in list(self.regime_history)[-10:]]
        regime_changes = sum(1 for i in range(1, len(recent_regimes)) 
                           if recent_regimes[i] != recent_regimes[i-1])
        
        stability = 1.0 - (regime_changes / len(recent_regimes))
        
        # Higher stability = lower transition probability
        base_prob = (1.0 - stability) / len(self.regimes)
        current_prob = stability + base_prob
        
        probs = {regime: base_prob for regime in self.regimes}
        probs[self.current_regime] = current_prob
        
        return probs


class OrderFlowAnalyzer:
    """Analyzes order flow patterns without traditional order book data."""
    
    def __init__(self):
        """Initialize order flow analyzer."""
        self.price_volume_history = deque(maxlen=100)
        self.flow_patterns = deque(maxlen=50)
    
    def analyze_flow(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Analyze order flow from price-volume relationship.
        
        Args:
            prices: Price array
            volumes: Volume array
            
        Returns:
            Dict[str, float]: Flow analysis metrics
        """
        if len(prices) < 10 or len(volumes) < 10:
            return {"buying_pressure": 0.5, "selling_pressure": 0.5}
        
        # Store price-volume data
        for i in range(min(len(prices), len(volumes))):
            self.price_volume_history.append((prices[i], volumes[i]))
        
        # Analyze recent flow patterns
        recent_data = list(self.price_volume_history)[-20:]
        
        if len(recent_data) < 10:
            return {"buying_pressure": 0.5, "selling_pressure": 0.5}
        
        # Calculate flow metrics
        buying_pressure = self._calculate_buying_pressure(recent_data)
        selling_pressure = 1.0 - buying_pressure
        
        # Volume-price divergence
        divergence = self._calculate_divergence(recent_data)
        
        return {
            "buying_pressure": buying_pressure,
            "selling_pressure": selling_pressure,
            "divergence": divergence,
            "flow_strength": abs(buying_pressure - 0.5) * 2
        }
    
    def _calculate_buying_pressure(self, price_volume_data: List[Tuple[float, float]]) -> float:
        """Calculate buying pressure from price-volume relationship."""
        if len(price_volume_data) < 2:
            return 0.5
        
        # Simple heuristic: rising prices with increasing volume = buying pressure
        pressure_signals = []
        
        for i in range(1, len(price_volume_data)):
            prev_price, prev_volume = price_volume_data[i-1]
            curr_price, curr_volume = price_volume_data[i]
            
            price_change = (curr_price - prev_price) / max(prev_price, 1e-8)
            volume_change = (curr_volume - prev_volume) / max(prev_volume, 1.0)
            
            # Positive price change with volume increase = buying
            if price_change > 0 and volume_change > 0:
                pressure_signals.append(0.8)
            elif price_change > 0 and volume_change < 0:
                pressure_signals.append(0.6)  # Price up, volume down = weaker buying
            elif price_change < 0 and volume_change > 0:
                pressure_signals.append(0.2)  # Price down, volume up = selling
            else:
                pressure_signals.append(0.4)  # Price down, volume down = neutral
        
        return np.mean(pressure_signals) if pressure_signals else 0.5
    
    def _calculate_divergence(self, price_volume_data: List[Tuple[float, float]]) -> float:
        """Calculate price-volume divergence."""
        if len(price_volume_data) < 5:
            return 0.0
        
        prices = [pv[0] for pv in price_volume_data]
        volumes = [pv[1] for pv in price_volume_data]
        
        # Calculate trends
        price_trend = (prices[-1] - prices[0]) / max(prices[0], 1e-8)
        volume_trend = (volumes[-1] - volumes[0]) / max(volumes[0], 1.0)
        
        # Divergence when trends oppose
        if price_trend > 0 and volume_trend < -0.1:
            return 0.7  # Price up, volume down = bearish divergence
        elif price_trend < 0 and volume_trend > 0.1:
            return -0.7  # Price down, volume up = bullish divergence
        else:
            return 0.0  # No significant divergence


class MicrostructureSubsystem:
    """Market microstructure analysis subsystem."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize microstructure subsystem.
        
        Args:
            config: Subsystem configuration
        """
        self.config = config
        
        # Configuration
        self.window_size = config.get("regime_detection_window", MICROSTRUCTURE_DEFAULT_WINDOW)
        self.volatility_threshold = config.get("volatility_threshold", MICROSTRUCTURE_DEFAULT_VOL_THRESHOLD)
        
        # Components
        self.regime_detector = RegimeDetector(self.window_size)
        self.order_flow_analyzer = OrderFlowAnalyzer()
        
        # Market state tracking
        self.liquidity_score = 0.5
        self.market_efficiency = 0.5
        self.institutional_activity = 0.5
        
        # Performance tracking
        self.total_regime_predictions = 0
        self.correct_regime_predictions = 0
        self.regime_stability_score = deque(maxlen=100)
        
        logger.info(
            "Microstructure subsystem initialized",
            window_size=self.window_size,
            volatility_threshold=self.volatility_threshold
        )
    
    async def analyze(self, state: State) -> Optional[AISignal]:
        """Analyze market microstructure patterns.
        
        Args:
            state: Current market state
            
        Returns:
            Optional[AISignal]: Microstructure analysis signal
        """
        try:
            # Extract microstructure data
            prices, volumes = self._extract_microstructure_data(state)
            
            if len(prices) < 8:  # Reduced from 20 to 8
                return None
            
            # Multi-component analysis
            analyses = []
            
            # Regime detection
            regime_analysis = await self._analyze_regime(prices, volumes)
            if regime_analysis:
                analyses.append(regime_analysis)
            
            # Order flow analysis
            flow_analysis = await self._analyze_order_flow(prices, volumes)
            if flow_analysis:
                analyses.append(flow_analysis)
            
            # Liquidity assessment
            liquidity_analysis = await self._analyze_liquidity(prices, volumes)
            if liquidity_analysis:
                analyses.append(liquidity_analysis)
            
            # Market efficiency assessment
            efficiency_analysis = await self._analyze_efficiency(prices, volumes)
            if efficiency_analysis:
                analyses.append(efficiency_analysis)
            
            # Combine analyses
            if analyses:
                combined_signal = self._combine_microstructure_signals(analyses, state.timestamp)
                return combined_signal
            
            # Generate occasional test signal when enough data is available
            if len(prices) >= 8 and len(prices) % 15 == 0:
                # Generate a low-confidence signal to stay active
                return AISignal(
                    signal_type=SignalType.MICROSTRUCTURE,
                    action=ActionType.HOLD,
                    confidence=0.35,  # Low confidence
                    strength=0.15,    # Low strength
                    metadata={"type": "microstructure_test", "data_points": len(prices)},
                    timestamp=state.timestamp
                )
            
            return None
            
        except Exception as e:
            logger.error("Microstructure analysis failed", error=str(e))
            return None
    
    def _extract_microstructure_data(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        """Extract price and volume data for microstructure analysis."""
        # Use 15-minute timeframe for microstructure analysis
        prices = np.array(state.prices) if len(state.prices) > 0 else np.array([])
        volumes = np.array(state.volumes) if len(state.volumes) > 0 else np.array([])
        
        # Ensure arrays are same length
        min_len = min(len(prices), len(volumes))
        if min_len > 0:
            prices = prices[-min_len:]
            volumes = volumes[-min_len:]
        
        return prices, volumes
    
    async def _analyze_regime(self, prices: np.ndarray, volumes: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze market regime."""
        regime, confidence = self.regime_detector.detect_regime(prices, volumes)
        
        if confidence > 0.4:  # Minimum confidence threshold
            # Map regime to trading action
            action_map = {
                "trending": ActionType.BUY,  # Follow trend
                "ranging": ActionType.HOLD,  # Wait for breakout
                "volatile": ActionType.HOLD,  # Avoid volatility
                "calm": ActionType.BUY      # Opportunity in calm markets
            }
            
            action = action_map.get(regime, ActionType.HOLD)
            
            return {
                "source": "regime_analysis",
                "action": action,
                "confidence": confidence,
                "regime": regime,
                "regime_stability": confidence
            }
        
        return None
    
    async def _analyze_order_flow(self, prices: np.ndarray, volumes: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze order flow patterns."""
        flow_metrics = self.order_flow_analyzer.analyze_flow(prices, volumes)
        
        buying_pressure = flow_metrics["buying_pressure"]
        flow_strength = flow_metrics["flow_strength"]
        
        if flow_strength > 0.3:  # Significant flow detected
            if buying_pressure > 0.65:
                action = ActionType.BUY
                confidence = min(buying_pressure, 0.8)
            elif buying_pressure < 0.35:
                action = ActionType.SELL
                confidence = min(1.0 - buying_pressure, 0.8)
            else:
                action = ActionType.HOLD
                confidence = 0.4
            
            return {
                "source": "order_flow",
                "action": action,
                "confidence": confidence,
                "buying_pressure": buying_pressure,
                "flow_strength": flow_strength
            }
        
        return None
    
    async def _analyze_liquidity(self, prices: np.ndarray, volumes: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze market liquidity conditions."""
        if len(prices) < 10 or len(volumes) < 10:
            return None
        
        # Estimate liquidity from price-volume relationship
        recent_prices = prices[-10:]
        recent_volumes = volumes[-10:]
        
        # Price impact measure (price change per unit volume)
        price_changes = np.abs(np.diff(recent_prices))
        volume_changes = recent_volumes[1:]
        
        # Avoid division by zero
        volume_changes = np.where(volume_changes == 0, 1.0, volume_changes)
        price_impact = np.mean(price_changes / volume_changes)
        
        # Lower price impact = higher liquidity
        liquidity_score = 1.0 / (1.0 + price_impact * 1000)  # Normalize
        self.liquidity_score = liquidity_score
        
        if liquidity_score > 0.7:  # High liquidity
            return {
                "source": "liquidity_analysis",
                "action": ActionType.BUY,  # Favor trading in liquid markets
                "confidence": 0.6,
                "liquidity_score": liquidity_score
            }
        elif liquidity_score < 0.3:  # Low liquidity
            return {
                "source": "liquidity_analysis", 
                "action": ActionType.HOLD,  # Avoid illiquid markets
                "confidence": 0.7,
                "liquidity_score": liquidity_score
            }
        
        return None
    
    async def _analyze_efficiency(self, prices: np.ndarray, volumes: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze market efficiency (mean reversion vs momentum)."""
        if len(prices) < 20:
            return None
        
        # Calculate autocorrelation in returns to measure efficiency
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) < 10:
            return None
        
        # Lag-1 autocorrelation
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        
        if np.isnan(autocorr):
            autocorr = 0.0
        
        self.market_efficiency = 1.0 - abs(autocorr)  # Perfect efficiency = 0 autocorr
        
        # Strong negative autocorr = mean reversion
        # Strong positive autocorr = momentum
        if autocorr < -0.3:  # Mean reversion
            return {
                "source": "efficiency_analysis",
                "action": ActionType.SELL,  # Expect reversal
                "confidence": min(abs(autocorr), 0.7),
                "efficiency_score": self.market_efficiency,
                "autocorrelation": autocorr
            }
        elif autocorr > 0.3:  # Momentum
            return {
                "source": "efficiency_analysis",
                "action": ActionType.BUY,  # Follow momentum
                "confidence": min(autocorr, 0.7),
                "efficiency_score": self.market_efficiency,
                "autocorrelation": autocorr
            }
        
        return None
    
    def _combine_microstructure_signals(self, analyses: List[Dict[str, Any]], 
                                      timestamp: int) -> AISignal:
        """Combine microstructure analyses into single signal."""
        # Weight different sources
        source_weights = {
            "regime_analysis": 0.4,
            "order_flow": 0.3,
            "liquidity_analysis": 0.2,
            "efficiency_analysis": 0.1
        }
        
        action_votes = {ActionType.HOLD: 0.0, ActionType.BUY: 0.0, ActionType.SELL: 0.0}
        total_weight = 0.0
        metadata = {}
        
        for analysis in analyses:
            source = analysis["source"]
            weight = source_weights.get(source, 0.1)
            confidence = analysis["confidence"]
            
            effective_weight = weight * confidence
            action_votes[analysis["action"]] += effective_weight
            total_weight += effective_weight
            
            # Collect metadata
            metadata[f"{source}_confidence"] = confidence
            metadata.update({k: v for k, v in analysis.items() 
                           if k not in ["source", "action", "confidence"]})
        
        # Determine final action
        if total_weight > 0:
            best_action = max(action_votes.items(), key=lambda x: x[1])[0]
            final_confidence = action_votes[best_action] / total_weight
        else:
            best_action = ActionType.HOLD
            final_confidence = 0.3
        
        # Calculate strength based on consensus
        consensus = len([a for a in analyses if a["action"] == best_action]) / len(analyses)
        strength = consensus * final_confidence
        
        metadata.update({
            "analysis_count": len(analyses),
            "consensus_ratio": consensus,
            "liquidity_score": self.liquidity_score,
            "market_efficiency": self.market_efficiency
        })
        
        return AISignal(
            signal_type=SignalType.MICROSTRUCTURE,
            action=best_action,
            confidence=final_confidence,
            strength=strength,
            metadata=metadata,
            timestamp=timestamp
        )
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get microstructure subsystem performance metrics."""
        regime_accuracy = (self.correct_regime_predictions / 
                          max(self.total_regime_predictions, 1))
        
        avg_stability = np.mean(self.regime_stability_score) if self.regime_stability_score else 0.5
        
        return {
            "regime_accuracy": regime_accuracy,
            "regime_stability": avg_stability,
            "current_liquidity": self.liquidity_score,
            "market_efficiency": self.market_efficiency,
            "institutional_activity": self.institutional_activity,
            "total_predictions": self.total_regime_predictions
        }
    
    def update_parameters(self, params: Dict[str, Any]) -> None:
        """Update subsystem parameters."""
        if "volatility_threshold" in params:
            self.volatility_threshold = params["volatility_threshold"]
        
        if "window_size" in params:
            self.window_size = params["window_size"]
            self.regime_detector.window_size = self.window_size
        
        logger.debug("Microstructure parameters updated", params=params)
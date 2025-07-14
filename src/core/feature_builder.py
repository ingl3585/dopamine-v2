"""
Feature builder for AI-focused market data features.
Separated from market_data.py for clean code compliance (< 600 lines per file).
"""

import numpy as np
import time
from typing import List, Dict, Any
from collections import deque
import structlog

from ..shared.constants import (
    TIMEFRAMES, FEATURE_WINDOWS, STATE_PRICE_DIM, STATE_VOLUME_DIM, 
    STATE_ACCOUNT_DIM, STATE_MARKET_DIM, STATE_TECHNICAL_DIM
)
from ..shared.utils import (
    calculate_returns, calculate_volatility, normalize_array
)
from ..shared.types import AccountInfo, MarketConditions, State

logger = structlog.get_logger(__name__)


class FeatureBuilder:
    """Builds AI-focused features from market data without classic indicators."""
    
    def __init__(self):
        """Initialize feature builder."""
        self.last_feature_update = 0
        
    def build_state(self, price_cache: Dict[str, deque], volume_cache: Dict[str, deque],
                   account_history: deque, market_history: deque) -> State:
        """Build complete state vector for RL agent.
        
        Args:
            price_cache: Price data cache by timeframe
            volume_cache: Volume data cache by timeframe  
            account_history: Account information history
            market_history: Market conditions history
            
        Returns:
            State: Complete state vector for RL agent
        """
        prices = self._build_price_features(price_cache)
        volumes = self._build_volume_features(volume_cache)
        account = self._build_account_features(account_history)
        market = self._build_market_features(market_history)
        technical = self._build_technical_features(price_cache, volume_cache)
        subsystem = np.zeros(25)  # Placeholder for subsystem signals
        
        return State(
            prices=prices,
            volumes=volumes,
            account_metrics=account,
            market_conditions=market,
            technical_indicators=technical,
            subsystem_signals=subsystem,
            timestamp=int(time.time() * 1000)  # Current timestamp in milliseconds
        )
    
    def _build_price_features(self, price_cache: Dict[str, deque]) -> np.ndarray:
        """Build price feature vector."""
        features = []
        
        # Get recent prices from each timeframe
        for tf in TIMEFRAMES:
            prices = list(price_cache[tf])[-STATE_PRICE_DIM//len(TIMEFRAMES):]
            
            # Pad if insufficient data
            while len(prices) < STATE_PRICE_DIM//len(TIMEFRAMES):
                prices.insert(0, prices[0] if prices else 0.0)
            
            # Normalize prices
            if prices:
                normalized = normalize_array(np.array(prices))
                features.extend(normalized)
        
        # Ensure correct dimension
        features = features[:STATE_PRICE_DIM]
        while len(features) < STATE_PRICE_DIM:
            features.append(0.0)
        
        return np.array(features)
    
    def _build_volume_features(self, volume_cache: Dict[str, deque]) -> np.ndarray:
        """Build volume feature vector."""
        features = []
        
        # Get recent volumes from each timeframe
        for tf in TIMEFRAMES:
            volumes = list(volume_cache[tf])[-STATE_VOLUME_DIM//len(TIMEFRAMES):]
            
            # Pad if insufficient data
            while len(volumes) < STATE_VOLUME_DIM//len(TIMEFRAMES):
                volumes.insert(0, volumes[0] if volumes else 0.0)
            
            # Normalize volumes
            if volumes:
                normalized = normalize_array(np.array(volumes))
                features.extend(normalized)
        
        # Ensure correct dimension
        features = features[:STATE_VOLUME_DIM]
        while len(features) < STATE_VOLUME_DIM:
            features.append(0.0)
        
        return np.array(features)
    
    def _build_account_features(self, account_history: deque) -> np.ndarray:
        """Build account metrics feature vector."""
        if not account_history:
            return np.zeros(STATE_ACCOUNT_DIM)
        
        account = account_history[-1]
        
        # Normalize account metrics
        features = [
            account.account_balance / 100000.0,  # Normalize to 100k
            account.buying_power / 100000.0,
            account.daily_pnl / 10000.0,  # Normalize to 10k
            account.unrealized_pnl / 10000.0,
            account.net_liquidation / 100000.0,
            account.margin_used / 100000.0,
            account.available_margin / 100000.0,
            account.open_positions / 10.0,  # Normalize to max 10 positions
            account.total_position_size / 10.0,
            0.0  # Reserved for future use
        ]
        
        return np.array(features[:STATE_ACCOUNT_DIM])
    
    def _build_market_features(self, market_history: deque) -> np.ndarray:
        """Build market conditions feature vector."""
        if not market_history:
            return np.zeros(STATE_MARKET_DIM)
        
        market = market_history[-1]
        
        features = [
            market.volatility,
            market.drawdown_pct / 100.0,
            market.portfolio_heat / 100.0,
            market.trend_strength,
            1.0 if market.regime == "trending" else 0.0
        ]
        
        return np.array(features[:STATE_MARKET_DIM])
    
    def _build_technical_features(self, price_cache: Dict[str, deque], 
                                 volume_cache: Dict[str, deque]) -> np.ndarray:
        """Build raw price/volume features without classic indicators."""
        features = []
        
        # Use 15m timeframe for technical features
        prices = list(price_cache["15m"])
        volumes = list(volume_cache["15m"])
        
        if len(prices) < 20:  # Need minimum data
            return np.zeros(STATE_TECHNICAL_DIM)
        
        # Price momentum features (raw returns over different windows)
        returns = calculate_returns(prices)
        if len(returns) >= 10:
            # Short-term momentum (last 3 periods)
            short_momentum = sum(returns[-3:]) if len(returns) >= 3 else 0.0
            features.append(short_momentum)
            
            # Medium-term momentum (last 10 periods)
            medium_momentum = sum(returns[-10:]) if len(returns) >= 10 else 0.0
            features.append(medium_momentum)
            
            # Return acceleration (change in momentum)
            if len(returns) >= 6:
                recent_momentum = sum(returns[-3:])
                previous_momentum = sum(returns[-6:-3])
                momentum_change = recent_momentum - previous_momentum
                features.append(momentum_change)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Volatility (essential metric)
        if returns:
            vol = calculate_volatility(returns)
            features.append(vol)
            
            # Volatility change
            if len(returns) >= 20:
                recent_vol = np.std(returns[-10:]) if len(returns) >= 10 else 0.0
                historical_vol = np.std(returns[-20:-10]) if len(returns) >= 20 else 0.0
                vol_change = recent_vol - historical_vol
                features.append(vol_change)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Volume features
        if volumes and len(volumes) >= 10:
            # Volume momentum
            volume_returns = calculate_returns(volumes)
            if volume_returns:
                vol_momentum = sum(volume_returns[-5:]) if len(volume_returns) >= 5 else 0.0
                features.append(vol_momentum)
            else:
                features.append(0.0)
            
            # Price-volume correlation
            if len(prices) == len(volumes) and len(returns) >= 5:
                recent_price_changes = returns[-5:]
                recent_volume_changes = volume_returns[-5:] if len(volume_returns) >= 5 else [0.0] * 5
                
                if len(recent_price_changes) == len(recent_volume_changes):
                    price_vol_corr = np.corrcoef(recent_price_changes, recent_volume_changes)[0, 1]
                    features.append(price_vol_corr if not np.isnan(price_vol_corr) else 0.0)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Raw price patterns
        if len(prices) >= 10:
            recent_prices = prices[-10:]
            price_std = np.std(recent_prices)
            price_mean = np.mean(recent_prices)
            price_dispersion = price_std / price_mean if price_mean > 0 else 0.0
            features.append(price_dispersion)
            
            # Price position in recent range
            price_min = min(recent_prices)
            price_max = max(recent_prices)
            if price_max > price_min:
                price_position = (prices[-1] - price_min) / (price_max - price_min)
                features.append(price_position)
            else:
                features.append(0.5)
            
            # Price acceleration
            if len(returns) >= 3:
                return_changes = calculate_returns(returns[-3:])
                if return_changes:
                    price_acceleration = return_changes[-1] if return_changes else 0.0
                    features.append(price_acceleration)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Microstructure features
        if len(prices) >= 5:
            recent_returns = returns[-5:] if len(returns) >= 5 else returns
            if recent_returns:
                max_return = max(recent_returns)
                min_return = min(recent_returns)
                return_range = max_return - min_return
                features.append(return_range)
            else:
                features.append(0.0)
            
            # Trend consistency
            if len(returns) >= 5:
                positive_returns = sum(1 for r in returns[-5:] if r > 0)
                trend_consistency = (positive_returns / 5.0) - 0.5
                features.append(trend_consistency)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Pad remaining features
        while len(features) < STATE_TECHNICAL_DIM:
            features.append(0.0)
        
        return np.array(features[:STATE_TECHNICAL_DIM])
    
    def has_sufficient_data(self, price_cache: Dict[str, deque]) -> bool:
        """Check if we have sufficient data to build state."""
        min_required = max(FEATURE_WINDOWS) + 1
        
        for tf in TIMEFRAMES:
            if len(price_cache[tf]) < min_required:
                return False
        
        return True
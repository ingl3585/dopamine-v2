"""
Market data processor for the Dopamine Trading System.
Handles validation, normalization, and feature extraction from incoming market data.
"""

from typing import Dict, List, Optional, Callable, Any
from collections import deque
import numpy as np
import structlog

from ..shared.types import (
    LiveDataMessage, HistoricalData, TradeCompletion, MarketData, 
    AccountInfo, MarketConditions, BarData, State
)
from ..shared.constants import (
    TIMEFRAMES, MAX_CACHE_SIZE, MAX_PRICE_HISTORY, FEATURE_WINDOWS,
    INDICATOR_PERIODS, STATE_PRICE_DIM, STATE_VOLUME_DIM, 
    STATE_ACCOUNT_DIM, STATE_MARKET_DIM, STATE_TECHNICAL_DIM
)
from ..shared.utils import (
    validate_price_data, validate_volume_data, calculate_returns,
    calculate_volatility, normalize_array, extract_features
)

logger = structlog.get_logger(__name__)


class MarketDataProcessor:
    """Processes and validates incoming market data."""
    
    def __init__(self, cache_size: int = MAX_CACHE_SIZE):
        """Initialize market data processor.
        
        Args:
            cache_size: Maximum number of data points to cache per timeframe
        """
        self.cache_size = cache_size
        
        # Data caches for each timeframe
        self._price_cache = {tf: deque(maxlen=cache_size) for tf in TIMEFRAMES}
        self._volume_cache = {tf: deque(maxlen=cache_size) for tf in TIMEFRAMES}
        self._timestamp_cache = {tf: deque(maxlen=cache_size) for tf in TIMEFRAMES}
        
        # Account and market condition history
        self._account_history = deque(maxlen=cache_size)
        self._market_history = deque(maxlen=cache_size)
        
        # Trade completion history
        self._trade_history = deque(maxlen=1000)
        
        # Feature cache
        self._features_cache = {}
        self._last_feature_update = 0
        
        # Callbacks
        self._data_callbacks: List[Callable[[LiveDataMessage], None]] = []
        self._historical_callbacks: List[Callable[[HistoricalData], None]] = []
        self._trade_callbacks: List[Callable[[TradeCompletion], None]] = []
        
        # Statistics
        self._stats = {
            "messages_processed": 0,
            "validation_errors": 0,
            "feature_updates": 0
        }
        
        logger.info("Market data processor initialized", cache_size=cache_size)
    
    def add_data_callback(self, callback: Callable[[LiveDataMessage], None]) -> None:
        """Add callback for live data updates."""
        self._data_callbacks.append(callback)
    
    def add_historical_callback(self, callback: Callable[[HistoricalData], None]) -> None:
        """Add callback for historical data updates."""
        self._historical_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[TradeCompletion], None]) -> None:
        """Add callback for trade completion updates."""
        self._trade_callbacks.append(callback)
    
    def process_message(self, message_data: Dict[str, Any]) -> bool:
        """Process incoming message from TCP bridge.
        
        Args:
            message_data: Raw message data from NinjaTrader
            
        Returns:
            bool: True if message was processed successfully
        """
        try:
            message_type = message_data.get("type")
            
            if message_type == "live_data":
                return self._process_live_data(message_data)
            elif message_type == "historical_data":
                return self._process_historical_data(message_data)
            elif message_type == "trade_completion":
                return self._process_trade_completion(message_data)
            else:
                logger.warning("Unknown message type", message_type=message_type)
                return False
                
        except Exception as e:
            logger.error("Failed to process message", error=str(e))
            self._stats["validation_errors"] += 1
            return False
    
    def get_current_state(self) -> Optional[State]:
        """Get current market state for RL agent.
        
        Returns:
            State: Current market state or None if insufficient data
        """
        try:
            # Check if we have sufficient data
            if not self._has_sufficient_data():
                return None
            
            # Build state components
            prices = self._build_price_features()
            volumes = self._build_volume_features()
            account = self._build_account_features()
            market = self._build_market_features()
            technical = self._build_technical_features()
            subsystem = np.zeros(25)  # Placeholder for subsystem signals
            
            return State(
                prices=prices,
                volumes=volumes,
                account_metrics=account,
                market_conditions=market,
                technical_indicators=technical,
                subsystem_signals=subsystem
            )
            
        except Exception as e:
            logger.error("Failed to build current state", error=str(e))
            return None
    
    def get_price_data(self, timeframe: str, lookback: int = 100) -> List[float]:
        """Get price data for specific timeframe.
        
        Args:
            timeframe: Timeframe identifier (1m, 5m, 15m, 1h, 4h)
            lookback: Number of periods to return
            
        Returns:
            List[float]: Price data
        """
        if timeframe not in self._price_cache:
            return []
        
        cache = self._price_cache[timeframe]
        return list(cache)[-lookback:] if cache else []
    
    def get_volume_data(self, timeframe: str, lookback: int = 100) -> List[float]:
        """Get volume data for specific timeframe.
        
        Args:
            timeframe: Timeframe identifier
            lookback: Number of periods to return
            
        Returns:
            List[float]: Volume data
        """
        if timeframe not in self._volume_cache:
            return []
        
        cache = self._volume_cache[timeframe]
        return list(cache)[-lookback:] if cache else []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dict[str, Any]: Processing statistics
        """
        return {
            **self._stats,
            "cache_sizes": {tf: len(cache) for tf, cache in self._price_cache.items()},
            "account_history_size": len(self._account_history),
            "trade_history_size": len(self._trade_history)
        }
    
    def _process_live_data(self, data: Dict[str, Any]) -> bool:
        """Process live market data message."""
        try:
            # Validate and store price data
            price_data = {
                "1m": data.get("price_1m", []),
                "5m": data.get("price_5m", []),
                "15m": data.get("price_15m", []),
                "1h": data.get("price_1h", []),
                "4h": data.get("price_4h", [])
            }
            
            volume_data = {
                "1m": data.get("volume_1m", []),
                "5m": data.get("volume_5m", []),
                "15m": data.get("volume_15m", []),
                "1h": data.get("volume_1h", []),
                "4h": data.get("volume_4h", [])
            }
            
            # Validate data
            for tf in TIMEFRAMES:
                if not validate_price_data(price_data.get(tf, [])):
                    logger.warning("Invalid price data", timeframe=tf)
                    return False
                if not validate_volume_data(volume_data.get(tf, [])):
                    logger.warning("Invalid volume data", timeframe=tf)
                    return False
            
            # Update caches with latest data point from each timeframe
            timestamp = data.get("timestamp", 0)
            for tf in TIMEFRAMES:
                if price_data[tf]:  # If we have data for this timeframe
                    self._price_cache[tf].append(price_data[tf][-1])
                    self._volume_cache[tf].append(volume_data[tf][-1])
                    self._timestamp_cache[tf].append(timestamp)
            
            # Store account info
            account_info = AccountInfo(
                account_balance=data.get("account_balance", 0.0),
                buying_power=data.get("buying_power", 0.0),
                daily_pnl=data.get("daily_pnl", 0.0),
                unrealized_pnl=data.get("unrealized_pnl", 0.0),
                net_liquidation=data.get("net_liquidation", 0.0),
                margin_used=data.get("margin_used", 0.0),
                available_margin=data.get("available_margin", 0.0),
                open_positions=data.get("open_positions", 0),
                total_position_size=data.get("total_position_size", 0)
            )
            self._account_history.append(account_info)
            
            # Store market conditions
            market_conditions = MarketConditions(
                current_price=data.get("current_price", 0.0),
                volatility=data.get("volatility", 0.0),
                drawdown_pct=data.get("drawdown_pct", 0.0),
                portfolio_heat=data.get("portfolio_heat", 0.0),
                regime=data.get("regime", "normal"),
                trend_strength=data.get("trend_strength", 0.5)
            )
            self._market_history.append(market_conditions)
            
            # Create live data message
            market_data = MarketData(
                price_1m=price_data["1m"],
                price_5m=price_data["5m"],
                price_15m=price_data["15m"],
                price_1h=price_data["1h"],
                price_4h=price_data["4h"],
                volume_1m=volume_data["1m"],
                volume_5m=volume_data["5m"],
                volume_15m=volume_data["15m"],
                volume_1h=volume_data["1h"],
                volume_4h=volume_data["4h"],
                timestamp=timestamp
            )
            
            live_message = LiveDataMessage(
                type="live_data",
                market_data=market_data,
                account_info=account_info,
                market_conditions=market_conditions
            )
            
            # Update features
            self._update_features()
            
            # Notify callbacks
            for callback in self._data_callbacks:
                callback(live_message)
            
            self._stats["messages_processed"] += 1
            return True
            
        except Exception as e:
            logger.error("Failed to process live data", error=str(e))
            return False
    
    def _process_historical_data(self, data: Dict[str, Any]) -> bool:
        """Process historical data message."""
        try:
            # Parse bar data for each timeframe
            def parse_bars(bars_data: List[Dict]) -> List[BarData]:
                bars = []
                for bar in bars_data:
                    bars.append(BarData(
                        timestamp=bar.get("timestamp", 0),
                        open=bar.get("open", 0.0),
                        high=bar.get("high", 0.0),
                        low=bar.get("low", 0.0),
                        close=bar.get("close", 0.0),
                        volume=bar.get("volume", 0)
                    ))
                return bars
            
            historical_data = HistoricalData(
                type="historical_data",
                bars_1m=parse_bars(data.get("bars_1m", [])),
                bars_5m=parse_bars(data.get("bars_5m", [])),
                bars_15m=parse_bars(data.get("bars_15m", [])),
                bars_1h=parse_bars(data.get("bars_1h", [])),
                bars_4h=parse_bars(data.get("bars_4h", [])),
                timestamp=data.get("timestamp", 0)
            )
            
            # Load historical data into caches
            timeframe_bars = {
                "1m": historical_data.bars_1m,
                "5m": historical_data.bars_5m,
                "15m": historical_data.bars_15m,
                "1h": historical_data.bars_1h,
                "4h": historical_data.bars_4h
            }
            
            for tf, bars in timeframe_bars.items():
                if bars:
                    # Clear existing cache and load historical data
                    self._price_cache[tf].clear()
                    self._volume_cache[tf].clear()
                    self._timestamp_cache[tf].clear()
                    
                    for bar in bars:
                        self._price_cache[tf].append(bar.close)
                        self._volume_cache[tf].append(bar.volume)
                        self._timestamp_cache[tf].append(bar.timestamp)
            
            # Update features with historical data
            self._update_features()
            
            # Notify callbacks
            for callback in self._historical_callbacks:
                callback(historical_data)
            
            logger.info(
                "Historical data processed",
                bars_1m=len(historical_data.bars_1m),
                bars_5m=len(historical_data.bars_5m),
                bars_15m=len(historical_data.bars_15m),
                bars_1h=len(historical_data.bars_1h),
                bars_4h=len(historical_data.bars_4h)
            )
            
            self._stats["messages_processed"] += 1
            return True
            
        except Exception as e:
            logger.error("Failed to process historical data", error=str(e))
            return False
    
    def _process_trade_completion(self, data: Dict[str, Any]) -> bool:
        """Process trade completion message."""
        try:
            trade_completion = TradeCompletion(
                type="trade_completion",
                pnl=data.get("pnl", 0.0),
                exit_price=data.get("exit_price", 0.0),
                entry_price=data.get("entry_price", 0.0),
                size=data.get("size", 0),
                exit_reason=data.get("exit_reason", "unknown"),
                entry_time=data.get("entry_time", 0),
                exit_time=data.get("exit_time", 0),
                account_balance=data.get("account_balance", 0.0),
                net_liquidation=data.get("net_liquidation", 0.0),
                margin_used=data.get("margin_used", 0.0),
                daily_pnl=data.get("daily_pnl", 0.0),
                trade_duration_minutes=data.get("trade_duration_minutes", 0.0),
                price_move_pct=data.get("price_move_pct", 0.0),
                volatility=data.get("volatility", 0.0),
                regime=data.get("regime", "normal"),
                trend_strength=data.get("trend_strength", 0.5),
                confidence=data.get("confidence", 0.5),
                consensus_strength=data.get("consensus_strength", 0.5),
                primary_tool=data.get("primary_tool", "unknown"),
                timestamp=data.get("timestamp", 0)
            )
            
            self._trade_history.append(trade_completion)
            
            # Notify callbacks
            for callback in self._trade_callbacks:
                callback(trade_completion)
            
            logger.info(
                "Trade completion processed",
                pnl=trade_completion.pnl,
                exit_reason=trade_completion.exit_reason,
                duration_minutes=trade_completion.trade_duration_minutes
            )
            
            self._stats["messages_processed"] += 1
            return True
            
        except Exception as e:
            logger.error("Failed to process trade completion", error=str(e))
            return False
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data to build state."""
        min_required = max(FEATURE_WINDOWS) + 1
        
        for tf in TIMEFRAMES:
            if len(self._price_cache[tf]) < min_required:
                return False
        
        return len(self._account_history) > 0 and len(self._market_history) > 0
    
    def _build_price_features(self) -> np.ndarray:
        """Build price feature vector."""
        features = []
        
        # Get recent prices from each timeframe
        for tf in TIMEFRAMES:
            prices = list(self._price_cache[tf])[-STATE_PRICE_DIM//len(TIMEFRAMES):]
            
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
    
    def _build_volume_features(self) -> np.ndarray:
        """Build volume feature vector."""
        features = []
        
        # Get recent volumes from each timeframe
        for tf in TIMEFRAMES:
            volumes = list(self._volume_cache[tf])[-STATE_VOLUME_DIM//len(TIMEFRAMES):]
            
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
    
    def _build_account_features(self) -> np.ndarray:
        """Build account metrics feature vector."""
        if not self._account_history:
            return np.zeros(STATE_ACCOUNT_DIM)
        
        account = self._account_history[-1]
        
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
    
    def _build_market_features(self) -> np.ndarray:
        """Build market conditions feature vector."""
        if not self._market_history:
            return np.zeros(STATE_MARKET_DIM)
        
        market = self._market_history[-1]
        
        features = [
            market.volatility,
            market.drawdown_pct / 100.0,
            market.portfolio_heat / 100.0,
            market.trend_strength,
            1.0 if market.regime == "trending" else 0.0
        ]
        
        return np.array(features[:STATE_MARKET_DIM])
    
    def _build_technical_features(self) -> np.ndarray:
        """Build raw price/volume features without classic indicators."""
        features = []
        
        # Use 15m timeframe for technical features
        prices = list(self._price_cache["15m"])
        volumes = list(self._volume_cache["15m"])
        
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
        
        # Volatility (essential metric - kept as requested)
        if returns:
            vol = calculate_volatility(returns)
            features.append(vol)
            
            # Volatility change (current vs historical)
            if len(returns) >= 20:
                recent_vol = np.std(returns[-10:]) if len(returns) >= 10 else 0.0
                historical_vol = np.std(returns[-20:-10]) if len(returns) >= 20 else 0.0
                vol_change = recent_vol - historical_vol
                features.append(vol_change)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Volume features (essential metric - kept as requested)
        if volumes and len(volumes) >= 10:
            # Volume momentum
            volume_returns = calculate_returns(volumes)
            if volume_returns:
                vol_momentum = sum(volume_returns[-5:]) if len(volume_returns) >= 5 else 0.0
                features.append(vol_momentum)
            else:
                features.append(0.0)
            
            # Price-volume relationship
            if len(prices) == len(volumes) and len(returns) >= 5:
                # Correlation between recent price and volume changes
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
        
        # Raw price patterns (let AI discover patterns)
        if len(prices) >= 10:
            # Price dispersion (how spread out recent prices are)
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
                features.append(0.5)  # Middle position if no range
            
            # Price acceleration (second derivative)
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
        
        # Microstructure features (price action patterns)
        if len(prices) >= 5:
            # Gap analysis (price jumps)
            recent_returns = returns[-5:] if len(returns) >= 5 else returns
            if recent_returns:
                max_return = max(recent_returns)
                min_return = min(recent_returns)
                return_range = max_return - min_return
                features.append(return_range)
            else:
                features.append(0.0)
            
            # Trend consistency (how consistent the direction is)
            if len(returns) >= 5:
                positive_returns = sum(1 for r in returns[-5:] if r > 0)
                trend_consistency = (positive_returns / 5.0) - 0.5  # Center around 0
                features.append(trend_consistency)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Pad remaining features with zeros
        while len(features) < STATE_TECHNICAL_DIM:
            features.append(0.0)
        
        return np.array(features[:STATE_TECHNICAL_DIM])
    
    def _update_features(self) -> None:
        """Update cached features."""
        try:
            # Extract features from each timeframe
            for tf in TIMEFRAMES:
                prices = list(self._price_cache[tf])
                volumes = list(self._volume_cache[tf])
                
                if len(prices) >= max(FEATURE_WINDOWS):
                    self._features_cache[tf] = extract_features(prices, volumes)
            
            self._stats["feature_updates"] += 1
            
        except Exception as e:
            logger.error("Failed to update features", error=str(e))
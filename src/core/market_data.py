"""
Market data processor for the Dopamine Trading System.
Handles validation, normalization, and feature extraction from incoming market data.
"""

from typing import Dict, List, Optional, Callable, Any, Union, Awaitable
from collections import deque
import numpy as np
import asyncio
import inspect
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
    validate_price_data, validate_volume_data, extract_features
)
from .feature_builder import FeatureBuilder

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
        
        # Feature builder
        self._feature_builder = FeatureBuilder()
        
        # Feature cache
        self._features_cache = {}
        self._last_feature_update = 0
        
        # Callbacks (can be sync or async)
        self._data_callbacks: List[Union[Callable[[LiveDataMessage], None], Callable[[LiveDataMessage], Awaitable[None]]]] = []
        self._historical_callbacks: List[Union[Callable[[HistoricalData], None], Callable[[HistoricalData], Awaitable[None]]]] = []
        self._trade_callbacks: List[Union[Callable[[TradeCompletion], None], Callable[[TradeCompletion], Awaitable[None]]]] = []
        
        # Statistics
        self._stats = {
            "messages_processed": 0,
            "validation_errors": 0,
            "feature_updates": 0
        }
        
        logger.debug("Market data processor initialized", cache_size=cache_size)
    
    def add_data_callback(self, callback: Callable[[LiveDataMessage], None]) -> None:
        """Add callback for live data updates."""
        self._data_callbacks.append(callback)
    
    def add_historical_callback(self, callback: Callable[[HistoricalData], None]) -> None:
        """Add callback for historical data updates."""
        self._historical_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[TradeCompletion], None]) -> None:
        """Add callback for trade completion updates."""
        self._trade_callbacks.append(callback)
    
    async def _call_callbacks(self, callbacks: List, data: Any) -> None:
        """Call a list of callbacks, handling both sync and async functions."""
        for callback in callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    # Async callback - await it
                    await callback(data)
                else:
                    # Sync callback - call directly
                    callback(data)
            except Exception as e:
                logger.error(
                    "Callback failed",
                    callback_name=getattr(callback, '__name__', 'unknown'),
                    error=str(e)
                )
    
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
            elif message_type == "test_connection":
                return self._process_test_message(message_data)
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
            if not self._feature_builder.has_sufficient_data(self._price_cache):
                return None
            
            state = self._feature_builder.build_state(
                self._price_cache, self._volume_cache,
                self._account_history, self._market_history
            )
            
            logger.debug(
                "Built market state",
                price_features=len(state.prices),
                volume_features=len(state.volumes),
                account_features=len(state.account_metrics),
                technical_features=len(state.technical_indicators)
            )
            
            # Add cache status debugging
            cache_status = {tf: len(self._price_cache[tf]) for tf in TIMEFRAMES}
            logger.debug(
                "Market state cache status",
                cache_sizes=cache_status,
                sufficient_data=self._feature_builder.has_sufficient_data(self._price_cache)
            )
            
            return state
            
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
            
            logger.debug(
                "Live data received",
                price_data_lengths={tf: len(price_data[tf]) for tf in TIMEFRAMES},
                volume_data_lengths={tf: len(volume_data[tf]) for tf in TIMEFRAMES}
            )
            
            for tf in TIMEFRAMES:
                if price_data[tf]:  # If we have data for this timeframe
                    latest_price = price_data[tf][-1]
                    latest_volume = volume_data[tf][-1]
                    
                    self._price_cache[tf].append(latest_price)
                    self._volume_cache[tf].append(latest_volume)
                    self._timestamp_cache[tf].append(timestamp)
                    
                    logger.debug(
                        f"Updated {tf} cache",
                        latest_price=latest_price,
                        latest_volume=latest_volume,
                        cache_size=len(self._price_cache[tf])
                    )
            
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
            
            # Show cache status after live data update
            cache_status = {tf: len(self._price_cache[tf]) for tf in TIMEFRAMES}
            logger.debug(
                "Live data processed",
                cache_sizes=cache_status,
                sufficient_for_ai=self._has_sufficient_data_for_trading()
            )
            
            # Notify callbacks (async)
            asyncio.create_task(self._call_callbacks(self._data_callbacks, live_message))
            
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
            
            total_loaded = 0
            for tf, bars in timeframe_bars.items():
                if bars:
                    # Clear existing cache and load historical data
                    self._price_cache[tf].clear()
                    self._volume_cache[tf].clear()
                    self._timestamp_cache[tf].clear()
                    
                    # Validate and load bars
                    valid_bars = 0
                    for bar in bars:
                        if bar.close > 0 and bar.volume >= 0:  # Basic validation
                            self._price_cache[tf].append(bar.close)
                            self._volume_cache[tf].append(bar.volume)
                            self._timestamp_cache[tf].append(bar.timestamp)
                            valid_bars += 1
                    
                    total_loaded += valid_bars
                    logger.debug(
                        "Loaded historical bars",
                        timeframe=tf,
                        bars_loaded=valid_bars,
                        cache_size=len(self._price_cache[tf])
                    )
            
            # Update features with historical data
            self._update_features()
            
            # Notify callbacks (async)
            asyncio.create_task(self._call_callbacks(self._historical_callbacks, historical_data))
            
            # Show cache status after historical data loading
            cache_status = {tf: len(self._price_cache[tf]) for tf in TIMEFRAMES}
            
            logger.info(
                "Historical data processed and cached",
                total_bars_loaded=total_loaded,
                bars_1m=len(historical_data.bars_1m),
                bars_5m=len(historical_data.bars_5m),
                bars_15m=len(historical_data.bars_15m),
                bars_1h=len(historical_data.bars_1h),
                bars_4h=len(historical_data.bars_4h),
                cache_sizes=cache_status,
                cache_ready=self._has_sufficient_data_for_trading()
            )
            
            self._stats["messages_processed"] += 1
            return True
            
        except Exception as e:
            logger.error("Failed to process historical data", error=str(e))
            import traceback
            logger.error("Historical data processing traceback", traceback=traceback.format_exc())
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
            
            # Notify callbacks (async)
            asyncio.create_task(self._call_callbacks(self._trade_callbacks, trade_completion))
            
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
    
    def _process_test_message(self, data: Dict[str, Any]) -> bool:
        """Process test connection message from NinjaTrader."""
        try:
            message = data.get("message", "Unknown test message")
            timestamp = data.get("timestamp", 0)
            
            logger.info(
                "Test message received from NinjaTrader",
                message=message,
                timestamp=timestamp
            )
            
            self._stats["messages_processed"] += 1
            return True
            
        except Exception as e:
            logger.error("Failed to process test message", error=str(e))
            return False
    
    def _has_sufficient_data_for_trading(self) -> bool:
        """Check if we have sufficient historical data to start AI processing."""
        min_bars_required = 50  # Minimum bars needed for meaningful analysis
        
        for tf in ["1m", "5m", "15m"]:
            if len(self._price_cache[tf]) < min_bars_required:
                return False
        
        return True
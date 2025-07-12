"""
TCP bridge for communication with NinjaTrader ResearchStrategy.
Handles bidirectional data flow: receives market data and sends trading signals.
"""

import asyncio
import json
import struct
import threading
from typing import Callable, Optional, Dict, Any
import structlog

from ..shared.types import (
    LiveDataMessage, HistoricalData, TradeCompletion, TradingSignal, 
    MessageType, MarketData, AccountInfo, MarketConditions, BarData
)
from ..shared.constants import (
    TCP_HOST, TCP_PORT_DATA, TCP_PORT_SIGNALS, TCP_BUFFER_SIZE, 
    TCP_TIMEOUT, MESSAGE_HEADER_SIZE, MAX_MESSAGE_SIZE, JSON_ENCODING
)
from ..shared.utils import parse_json_message, serialize_to_json

logger = structlog.get_logger(__name__)


class TCPBridge:
    """TCP bridge for NinjaTrader communication."""
    
    def __init__(self, 
                 data_port: int = TCP_PORT_DATA,
                 signals_port: int = TCP_PORT_SIGNALS,
                 host: str = TCP_HOST):
        """Initialize TCP bridge.
        
        Args:
            data_port: Port for receiving market data from NinjaTrader
            signals_port: Port for sending trading signals to NinjaTrader
            host: Host address for TCP connections
        """
        self.host = host
        self.data_port = data_port
        self.signals_port = signals_port
        
        # Server instances
        self._data_server: Optional[asyncio.Server] = None
        self._signals_server: Optional[asyncio.Server] = None
        
        # Client connections
        self._ninja_data_writer: Optional[asyncio.StreamWriter] = None
        self._ninja_signals_writer: Optional[asyncio.StreamWriter] = None
        
        # Event loop and running state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Callback functions
        self._data_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._connection_callback: Optional[Callable[[bool], None]] = None
        
        # Connection status
        self._ninja_connected = False
        
    def set_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Set callback function for incoming market data.
        
        Args:
            callback: Function to call when market data is received
        """
        self._data_callback = callback
    
    def set_connection_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback function for connection status changes.
        
        Args:
            callback: Function to call when connection status changes
        """
        self._connection_callback = callback
    
    def start(self) -> None:
        """Start the TCP bridge servers."""
        if self._running:
            logger.warning("TCP bridge is already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        
        logger.info(
            "TCP bridge started",
            data_port=self.data_port,
            signals_port=self.signals_port
        )
    
    def stop(self) -> None:
        """Stop the TCP bridge servers."""
        if not self._running:
            logger.warning("TCP bridge is not running")
            return
        
        self._running = False
        
        if self._loop and not self._loop.is_closed():
            # Schedule shutdown in the event loop
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        
        if self._thread:
            self._thread.join(timeout=5.0)
        
        logger.info("TCP bridge stopped")
    
    def send_signal(self, signal: TradingSignal) -> bool:
        """Send trading signal to NinjaTrader.
        
        Args:
            signal: Trading signal to send
            
        Returns:
            bool: True if signal was sent successfully
        """
        if not self._ninja_signals_writer or self._ninja_signals_writer.is_closing():
            logger.warning("No signals connection to NinjaTrader")
            return False
        
        try:
            # Convert signal to JSON
            signal_dict = {
                "action": signal.action,
                "confidence": signal.confidence,
                "position_size": signal.position_size,
                "use_stop": signal.use_stop,
                "stop_price": signal.stop_price,
                "use_target": signal.use_target,
                "target_price": signal.target_price
            }
            
            json_str = serialize_to_json(signal_dict)
            
            # Send via event loop
            if self._loop and not self._loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(
                    self._send_message(self._ninja_signals_writer, json_str),
                    self._loop
                )
                return future.result(timeout=1.0)
            
            return False
            
        except Exception as e:
            logger.error("Failed to send trading signal", error=str(e))
            return False
    
    def is_connected(self) -> bool:
        """Check if NinjaTrader is connected.
        
        Returns:
            bool: True if NinjaTrader is connected
        """
        return self._ninja_connected
    
    def _run_event_loop(self) -> None:
        """Run the asyncio event loop in a separate thread."""
        try:
            # Create new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            # Start servers
            self._loop.run_until_complete(self._start_servers())
            
            # Run until stopped
            self._loop.run_forever()
            
        except Exception as e:
            logger.error("Event loop error", error=str(e))
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()
    
    async def _start_servers(self) -> None:
        """Start TCP servers for data and signals."""
        try:
            # Start data server (receives from NinjaTrader)
            self._data_server = await asyncio.start_server(
                self._handle_data_connection,
                self.host,
                self.data_port
            )
            
            # Start signals server (sends to NinjaTrader)
            self._signals_server = await asyncio.start_server(
                self._handle_signals_connection,
                self.host,
                self.signals_port
            )
            
            logger.info(
                "TCP servers started",
                data_address=f"{self.host}:{self.data_port}",
                signals_address=f"{self.host}:{self.signals_port}"
            )
            
        except Exception as e:
            logger.error("Failed to start TCP servers", error=str(e))
            raise
    
    async def _handle_data_connection(self, 
                                    reader: asyncio.StreamReader, 
                                    writer: asyncio.StreamWriter) -> None:
        """Handle incoming data connection from NinjaTrader."""
        client_addr = writer.get_extra_info('peername')
        logger.info("Data connection established", client=client_addr)
        
        self._ninja_data_writer = writer
        self._ninja_connected = True
        
        if self._connection_callback:
            self._connection_callback(True)
        
        try:
            while self._running and not reader.at_eof():
                # Read message header (4 bytes for length)
                header_data = await reader.read(MESSAGE_HEADER_SIZE)
                if len(header_data) != MESSAGE_HEADER_SIZE:
                    break
                
                # Extract message length
                message_length = struct.unpack('<I', header_data)[0]
                
                # Validate message length
                if message_length <= 0 or message_length > MAX_MESSAGE_SIZE:
                    logger.warning(
                        "Invalid message length",
                        length=message_length,
                        max_length=MAX_MESSAGE_SIZE
                    )
                    continue
                
                # Read message data
                message_data = await reader.read(message_length)
                if len(message_data) != message_length:
                    break
                
                # Process message
                try:
                    json_str = message_data.decode(JSON_ENCODING)
                    message_dict = parse_json_message(json_str)
                    
                    if message_dict and self._data_callback:
                        self._data_callback(message_dict)
                        
                except Exception as e:
                    logger.error("Failed to process data message", error=str(e))
                    
        except Exception as e:
            logger.error("Data connection error", error=str(e))
        finally:
            writer.close()
            await writer.wait_closed()
            self._ninja_data_writer = None
            self._ninja_connected = False
            
            if self._connection_callback:
                self._connection_callback(False)
            
            logger.info("Data connection closed", client=client_addr)
    
    async def _handle_signals_connection(self, 
                                       reader: asyncio.StreamReader, 
                                       writer: asyncio.StreamWriter) -> None:
        """Handle outgoing signals connection to NinjaTrader."""
        client_addr = writer.get_extra_info('peername')
        logger.info("Signals connection established", client=client_addr)
        
        self._ninja_signals_writer = writer
        
        try:
            # Keep connection alive - NinjaTrader will close when done
            while self._running and not reader.at_eof():
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error("Signals connection error", error=str(e))
        finally:
            writer.close()
            await writer.wait_closed()
            self._ninja_signals_writer = None
            logger.info("Signals connection closed", client=client_addr)
    
    async def _send_message(self, writer: asyncio.StreamWriter, message: str) -> bool:
        """Send message with length header.
        
        Args:
            writer: Stream writer to send message through
            message: JSON message to send
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            # Encode message
            message_bytes = message.encode(JSON_ENCODING)
            message_length = len(message_bytes)
            
            # Create header with message length
            header = struct.pack('<I', message_length)
            
            # Send header + message
            writer.write(header)
            writer.write(message_bytes)
            await writer.drain()
            
            logger.debug(
                "Message sent",
                length=message_length,
                message_preview=message[:100] + "..." if len(message) > 100 else message
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to send message", error=str(e))
            return False
    
    async def _shutdown(self) -> None:
        """Shutdown TCP servers and connections."""
        logger.info("Shutting down TCP bridge")
        
        # Close client connections
        if self._ninja_data_writer and not self._ninja_data_writer.is_closing():
            self._ninja_data_writer.close()
            await self._ninja_data_writer.wait_closed()
        
        if self._ninja_signals_writer and not self._ninja_signals_writer.is_closing():
            self._ninja_signals_writer.close()
            await self._ninja_signals_writer.wait_closed()
        
        # Close servers
        if self._data_server:
            self._data_server.close()
            await self._data_server.wait_closed()
        
        if self._signals_server:
            self._signals_server.close()
            await self._signals_server.wait_closed()
        
        # Stop event loop
        self._loop.stop()


def parse_live_data_message(data: Dict[str, Any]) -> Optional[LiveDataMessage]:
    """Parse live data message from NinjaTrader.
    
    Args:
        data: Raw message data from NinjaTrader
        
    Returns:
        LiveDataMessage: Parsed live data message or None if invalid
    """
    try:
        if data.get("type") != MessageType.LIVE_DATA.value:
            return None
        
        # Parse market data
        market_data = MarketData(
            price_1m=data.get("price_1m", []),
            price_5m=data.get("price_5m", []),
            price_15m=data.get("price_15m", []),
            price_1h=data.get("price_1h", []),
            price_4h=data.get("price_4h", []),
            volume_1m=data.get("volume_1m", []),
            volume_5m=data.get("volume_5m", []),
            volume_15m=data.get("volume_15m", []),
            volume_1h=data.get("volume_1h", []),
            volume_4h=data.get("volume_4h", []),
            timestamp=data.get("timestamp", 0)
        )
        
        # Parse account info
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
        
        # Parse market conditions
        market_conditions = MarketConditions(
            current_price=data.get("current_price", 0.0),
            volatility=data.get("volatility", 0.0),
            drawdown_pct=data.get("drawdown_pct", 0.0),
            portfolio_heat=data.get("portfolio_heat", 0.0),
            regime=data.get("regime", "normal"),
            trend_strength=data.get("trend_strength", 0.5)
        )
        
        return LiveDataMessage(
            type=data["type"],
            market_data=market_data,
            account_info=account_info,
            market_conditions=market_conditions
        )
        
    except Exception as e:
        logger.error("Failed to parse live data message", error=str(e))
        return None


def parse_historical_data_message(data: Dict[str, Any]) -> Optional[HistoricalData]:
    """Parse historical data message from NinjaTrader.
    
    Args:
        data: Raw message data from NinjaTrader
        
    Returns:
        HistoricalData: Parsed historical data or None if invalid
    """
    try:
        if data.get("type") != MessageType.HISTORICAL_DATA.value:
            return None
        
        def parse_bars(bars_data: list) -> list[BarData]:
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
        
        return HistoricalData(
            type=data["type"],
            bars_1m=parse_bars(data.get("bars_1m", [])),
            bars_5m=parse_bars(data.get("bars_5m", [])),
            bars_15m=parse_bars(data.get("bars_15m", [])),
            bars_1h=parse_bars(data.get("bars_1h", [])),
            bars_4h=parse_bars(data.get("bars_4h", [])),
            timestamp=data.get("timestamp", 0)
        )
        
    except Exception as e:
        logger.error("Failed to parse historical data message", error=str(e))
        return None


def parse_trade_completion_message(data: Dict[str, Any]) -> Optional[TradeCompletion]:
    """Parse trade completion message from NinjaTrader.
    
    Args:
        data: Raw message data from NinjaTrader
        
    Returns:
        TradeCompletion: Parsed trade completion data or None if invalid
    """
    try:
        if data.get("type") != MessageType.TRADE_COMPLETION.value:
            return None
        
        return TradeCompletion(
            type=data["type"],
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
        
    except Exception as e:
        logger.error("Failed to parse trade completion message", error=str(e))
        return None
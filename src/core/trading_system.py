"""
Main trading system coordinator for the Dopamine Trading System.
Orchestrates all components and manages the system lifecycle.
"""

import asyncio
import signal
import sys
from typing import Dict, Any, Optional
import structlog

from .config_manager import ConfigManager
from .tcp_bridge import TCPBridge
from .market_data import MarketDataProcessor
from ..agent.rl_agent import RLAgent
from ..intelligence.subsystem_manager import SubsystemManager
from ..intelligence.dna_subsystem import DNASubsystem
from ..intelligence.temporal_subsystem import TemporalSubsystem
from ..intelligence.immune_subsystem import ImmuneSubsystem
from ..intelligence.microstructure_subsystem import MicrostructureSubsystem
from ..intelligence.dopamine_subsystem import DopamineSubsystem
from ..neural.network_manager import NetworkManager
from ..neural.adaptive_network import AdaptiveNetwork
from ..neural.specialized_networks import ReinforcementLearningNetwork
from ..shared.types import (
    SystemConfig, LiveDataMessage, HistoricalData, TradeCompletion, 
    TradingSignal, ActionType, State, SubsystemConfig
)
from ..shared.constants import (
    SUCCESS_SYSTEM_START, SUCCESS_TCP_CONNECTED, SUCCESS_DATA_PROCESSED,
    ERROR_TCP_CONNECTION, ERROR_CONFIG_LOAD, ERROR_INSUFFICIENT_DATA,
    SYSTEM_STATES, SUBSYSTEM_NAMES
)
from ..shared.utils import setup_logging, ensure_directory_exists

logger = structlog.get_logger(__name__)


class TradingSystem:
    """Main system coordinator that orchestrates all components."""
    
    def __init__(self, config_file: str = "config/system_config.json"):
        """Initialize the trading system.
        
        Args:
            config_file: Path to system configuration file
        """
        self.config_file = config_file
        self.state = "initializing"
        
        # Core components
        self.config_manager: Optional[ConfigManager] = None
        self.tcp_bridge: Optional[TCPBridge] = None
        self.market_processor: Optional[MarketDataProcessor] = None
        
        # AI components
        self.rl_agent: Optional[RLAgent] = None
        self.subsystem_manager: Optional[SubsystemManager] = None
        self.network_manager: Optional[NetworkManager] = None
        
        # System state
        self.config: Optional[SystemConfig] = None
        self.is_running = False
        self.ninja_connected = False
        self.has_historical_data = False
        
        # Statistics
        self.stats = {
            "startup_time": 0.0,
            "messages_processed": 0,
            "signals_sent": 0,
            "errors": 0,
            "uptime_seconds": 0
        }
        
        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "profitable_trades": 0,
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_latency_ms": 0.0
        }
        
        logger.debug("Trading system initialized", config_file=config_file)
    
    async def start(self) -> bool:
        """Start the trading system.
        
        Returns:
            bool: True if system started successfully
        """
        try:
            start_time = asyncio.get_event_loop().time()
            self.state = "loading"
            
            # Load configuration
            if not await self._load_configuration():
                logger.error("Failed to load system configuration")
                self.state = "error"
                return False
            
            # Setup logging
            self._setup_logging()
            
            # Initialize components
            if not await self._initialize_components():
                logger.error("Failed to initialize core components")
                self.state = "error"
                return False
            
            logger.info("Loading reinforcement learning agent")
            # Initialize AI components
            if not await self._initialize_ai_components():
                logger.error("Failed to initialize AI subsystems")
                self.state = "error"
                return False
            
            logger.info("Establishing NinjaTrader connection")
            # Start TCP bridge
            self.state = "connecting"
            if not await self._start_tcp_bridge():
                logger.error("Failed to establish NinjaTrader connection")
                self.state = "error"
                return False
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Mark as running
            self.is_running = True
            self.state = "running"
            
            # Calculate startup time
            self.stats["startup_time"] = asyncio.get_event_loop().time() - start_time
            
            logger.info(
                "Trading system started successfully",
                startup_time=f"{self.stats['startup_time']:.2f}s",
                state=self.state
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to start trading system", error=str(e))
            self.state = "error"
            self.stats["errors"] += 1
            return False
    
    async def stop(self) -> None:
        """Stop the trading system gracefully."""
        if not self.is_running:
            logger.warning("Trading system is not running")
            return
        
        logger.info("Stopping trading system")
        self.state = "stopping"
        self.is_running = False
        
        try:
            # Stop TCP bridge
            if self.tcp_bridge:
                self.tcp_bridge.stop()
            
            # Stop AI components
            if self.network_manager:
                self.network_manager.cleanup()
            
            # Clean up other components
            if self.rl_agent:
                # RL agent cleanup (if implemented)
                pass
            if self.subsystem_manager:
                # Subsystem manager cleanup (if implemented)
                pass
            
            self.state = "stopped"
            logger.info("Trading system stopped successfully")
            
        except Exception as e:
            logger.error("Error during system shutdown", error=str(e))
            self.state = "error"
    
    async def run(self) -> None:
        """Run the trading system until stopped."""
        if not await self.start():
            logger.error("Failed to start trading system")
            return
        
        try:
            # Main event loop
            while self.is_running:
                await asyncio.sleep(1.0)
                self.stats["uptime_seconds"] += 1
                
                # Periodic health checks
                if self.stats["uptime_seconds"] % 60 == 0:  # Every minute
                    await self._health_check()
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error("Runtime error", error=str(e))
            self.stats["errors"] += 1
        finally:
            await self.stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status.
        
        Returns:
            Dict[str, Any]: System status information
        """
        status = {
            "state": self.state,
            "is_running": self.is_running,
            "ninja_connected": self.ninja_connected,
            "has_historical_data": self.has_historical_data,
            "statistics": self.stats.copy()
        }
        
        # Add component status
        if self.tcp_bridge:
            status["tcp_connected"] = self.tcp_bridge.is_connected()
        
        if self.market_processor:
            status["market_data"] = self.market_processor.get_statistics()
        
        return status
    
    def send_trading_signal(self, action: ActionType, confidence: float, 
                          position_size: int = 1, use_stop: bool = False,
                          stop_price: float = 0.0, use_target: bool = False,
                          target_price: float = 0.0) -> bool:
        """Send trading signal to NinjaTrader.
        
        Args:
            action: Trading action (hold, buy, sell)
            confidence: Signal confidence (0.0 to 1.0)
            position_size: Number of contracts
            use_stop: Whether to use stop loss
            stop_price: Stop loss price
            use_target: Whether to use profit target
            target_price: Profit target price
            
        Returns:
            bool: True if signal was sent successfully
        """
        if not self.tcp_bridge or not self.ninja_connected:
            logger.warning("Cannot send signal - NinjaTrader not connected")
            return False
        
        try:
            signal = TradingSignal(
                action=action.value,
                confidence=confidence,
                position_size=position_size,
                use_stop=use_stop,
                stop_price=stop_price,
                use_target=use_target,
                target_price=target_price
            )
            
            success = self.tcp_bridge.send_signal(signal)
            
            if success:
                self.stats["signals_sent"] += 1
                logger.info(
                    "Trading signal sent",
                    action=action.name,
                    confidence=confidence,
                    position_size=position_size
                )
            
            return success
            
        except Exception as e:
            logger.error("Failed to send trading signal", error=str(e))
            self.stats["errors"] += 1
            return False
    
    async def _load_configuration(self) -> bool:
        """Load system configuration."""
        try:
            self.config_manager = ConfigManager(self.config_file)
            self.config = self.config_manager.load_config()
            
            logger.debug(
                "Configuration loaded",
                environment=self.config.system.get("environment"),
                subsystems_enabled=sum(1 for s in self.config.subsystems.values() if s.enabled)
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            return False
    
    def _setup_logging(self) -> None:
        """Setup system logging."""
        try:
            log_level = self.config.system.get("log_level", "INFO")
            log_dir = "logs"
            
            ensure_directory_exists(log_dir)
            setup_logging(log_level, log_dir)
            
            logger.info(f"Logging configured - level: {log_level}, dir: {log_dir}")
            
        except Exception as e:
            logger.error("Failed to setup logging", error=str(e))
    
    async def _initialize_components(self) -> bool:
        """Initialize all system components."""
        try:
            # Initialize market data processor
            cache_size = self.config.data.get("cache_size", 10000)
            self.market_processor = MarketDataProcessor(cache_size=cache_size)
            
            # Set up callbacks
            self.market_processor.add_data_callback(self._on_live_data)
            self.market_processor.add_historical_callback(self._on_historical_data)
            self.market_processor.add_trade_callback(self._on_trade_completion)
            
            logger.debug("Components initialized", cache_size=cache_size)
            
            # AI components will be initialized in _initialize_ai_components
            
            return True
            
        except Exception as e:
            logger.error("Failed to initialize components", error=str(e))
            return False
    
    async def _initialize_ai_components(self) -> bool:
        """Initialize AI components (neural networks, RL agent, subsystems)."""
        try:
            logger.info("Initializing neural networks")
            
            # Initialize neural network manager
            self.network_manager = NetworkManager(self.config.neural)
            
            # Create adaptive network for RL agent
            adaptive_net = AdaptiveNetwork(self.config.neural)
            self.network_manager.register_network("main_dqn", adaptive_net)
            
            # Create specialized networks
            rl_network = ReinforcementLearningNetwork(
                state_dim=self.config.neural.get("input_dim", 100),
                action_dim=3,  # Hold, Buy, Sell
                network_type="dqn"
            )
            self.network_manager.register_network("rl_network", rl_network)
            
            logger.info("Configuring deep Q-learning agent")
            # Initialize RL agent
            self.rl_agent = RLAgent(
                config=self.config.agent,
                network_manager=self.network_manager
            )
            
            logger.info("Loading AI trading subsystems")
            # Initialize subsystem manager
            self.subsystem_manager = SubsystemManager(self.config.subsystems)
            
            # Initialize and register AI subsystems
            dna_config = self.config.subsystems["dna"].parameters if "dna" in self.config.subsystems else {}
            temporal_config = self.config.subsystems["temporal"].parameters if "temporal" in self.config.subsystems else {}
            immune_config = self.config.subsystems["immune"].parameters if "immune" in self.config.subsystems else {}
            microstructure_config = self.config.subsystems["microstructure"].parameters if "microstructure" in self.config.subsystems else {}
            dopamine_config = self.config.subsystems["dopamine"].parameters if "dopamine" in self.config.subsystems else {}
            
            dna_subsystem = DNASubsystem(dna_config)
            temporal_subsystem = TemporalSubsystem(temporal_config)
            immune_subsystem = ImmuneSubsystem(immune_config)
            microstructure_subsystem = MicrostructureSubsystem(microstructure_config)
            dopamine_subsystem = DopamineSubsystem(dopamine_config)
            
            # Register subsystems
            self.subsystem_manager.register_subsystem("dna", dna_subsystem)
            self.subsystem_manager.register_subsystem("temporal", temporal_subsystem)
            self.subsystem_manager.register_subsystem("immune", immune_subsystem)
            self.subsystem_manager.register_subsystem("microstructure", microstructure_subsystem)
            self.subsystem_manager.register_subsystem("dopamine", dopamine_subsystem)
            
            logger.info("AI subsystems ready: DNA, Temporal, Immune, Microstructure, Dopamine")
            
            logger.info(
                "AI components initialized",
                networks=len(self.network_manager.networks),
                subsystems=len(self.subsystem_manager.subsystems)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"AI system initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def _start_tcp_bridge(self) -> bool:
        """Start TCP bridge for NinjaTrader communication."""
        try:
            # Create TCP bridge
            self.tcp_bridge = TCPBridge(
                data_port=self.config.system.get("tcp_port_data", 5556),
                signals_port=self.config.system.get("tcp_port_signals", 5557),
                host=self.config.system.get("tcp_host", "localhost")
            )
            
            # Set callbacks
            self.tcp_bridge.set_data_callback(self._on_tcp_data)
            self.tcp_bridge.set_connection_callback(self._on_connection_change)
            
            # Start bridge
            self.tcp_bridge.start()
            
            # Wait a moment for startup
            await asyncio.sleep(1.0)
            
            logger.info("TCP bridge started - waiting for NinjaTrader connection")
            
            # Wait a bit longer to see if NinjaTrader connects
            await asyncio.sleep(2.0)
            
            if self.ninja_connected:
                logger.info("NinjaTrader connected during startup")
            else:
                logger.info("Still waiting for NinjaTrader connection - ensure ResearchStrategy is running")
            
            return True
            
        except Exception as e:
            logger.error("Failed to start TCP bridge", error=str(e))
            return False
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _health_check(self) -> None:
        """Perform periodic health check."""
        try:
            status = self.get_status()
            
            logger.debug(
                "Periodic health check",
                state=status["state"],
                uptime=status["statistics"]["uptime_seconds"],
                messages_processed=status["statistics"]["messages_processed"],
                ninja_connected=status["ninja_connected"]
            )
            
            # Check for issues
            if not self.ninja_connected:
                logger.warning("NinjaTrader not connected")
            
            if not self.has_historical_data:
                logger.warning("No historical data received yet")
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            self.stats["errors"] += 1
    
    def _on_tcp_data(self, message_data: Dict[str, Any]) -> None:
        """Handle incoming TCP data from NinjaTrader."""
        try:
            if self.market_processor:
                success = self.market_processor.process_message(message_data)
                if success:
                    self.stats["messages_processed"] += 1
                else:
                    self.stats["errors"] += 1
            
        except Exception as e:
            logger.error("Failed to process TCP data", error=str(e))
            self.stats["errors"] += 1
    
    def _on_connection_change(self, connected: bool) -> None:
        """Handle NinjaTrader connection status change."""
        self.ninja_connected = connected
        
        if connected:
            logger.info("NinjaTrader connected successfully")
            # Log current system status when connection is established
            logger.info(
                "System ready for trading",
                has_historical_data=self.has_historical_data,
                ai_components_ready=bool(self.rl_agent and self.subsystem_manager),
                tcp_connected=self.tcp_bridge.is_connected() if self.tcp_bridge else False
            )
        else:
            logger.warning("NinjaTrader disconnected - waiting for reconnection")
            # Reset historical data flag to trigger resend when reconnected
            if self.has_historical_data:
                logger.info("Will request historical data on next connection")
    
    def _on_live_data(self, data: LiveDataMessage) -> None:
        """Handle live market data."""
        try:
            logger.debug(
                "Live data received",
                current_price=data.market_conditions.current_price,
                account_balance=data.account_info.account_balance,
                position_size=data.account_info.total_position_size
            )
            
            # TODO: Process with AI components
            # if self.rl_agent and self.has_historical_data:
            #     current_state = self.market_processor.get_current_state()
            #     if current_state:
            #         action = self.rl_agent.select_action(current_state)
            #         if action != ActionType.HOLD:
            #             self.send_trading_signal(action, confidence=0.7)
            
        except Exception as e:
            logger.error("Failed to process live data", error=str(e))
            self.stats["errors"] += 1
    
    def _on_historical_data(self, data: HistoricalData) -> None:
        """Handle historical market data."""
        try:
            self.has_historical_data = True
            
            total_bars = (len(data.bars_1m) + len(data.bars_5m) + 
                         len(data.bars_15m) + len(data.bars_1h) + len(data.bars_4h))
            
            logger.info(
                "Historical data received",
                total_bars=total_bars,
                bars_1m=len(data.bars_1m),
                bars_5m=len(data.bars_5m),
                bars_15m=len(data.bars_15m),
                bars_1h=len(data.bars_1h),
                bars_4h=len(data.bars_4h)
            )
            
            # TODO: Initialize AI models with historical data
            # if self.rl_agent:
            #     self.rl_agent.initialize_with_history(data)
            
        except Exception as e:
            logger.error("Failed to process historical data", error=str(e))
            self.stats["errors"] += 1
    
    def _on_trade_completion(self, data: TradeCompletion) -> None:
        """Handle trade completion data."""
        try:
            logger.info(
                "Trade completed",
                pnl=data.pnl,
                exit_reason=data.exit_reason,
                duration_minutes=data.trade_duration_minutes,
                price_move_pct=data.price_move_pct
            )
            
            # TODO: Update AI models with trade results
            # if self.rl_agent:
            #     reward = self._calculate_reward(data)
            #     self.rl_agent.update_with_reward(reward, data)
            
        except Exception as e:
            logger.error("Failed to process trade completion", error=str(e))
            self.stats["errors"] += 1
    
    def _calculate_reward(self, trade_data: TradeCompletion) -> float:
        """Calculate reward signal from trade completion (placeholder).
        
        Args:
            trade_data: Trade completion data
            
        Returns:
            float: Calculated reward
        """
        # TODO: Implement sophisticated reward calculation
        # This is a simple placeholder implementation
        
        # Base reward from P&L
        pnl_reward = trade_data.pnl / 100.0  # Normalize
        
        # Penalty for excessive drawdown
        drawdown_penalty = 0.0
        if trade_data.daily_pnl < -500:  # More than $500 daily loss
            drawdown_penalty = -0.5
        
        # Bonus for quick profitable trades
        efficiency_bonus = 0.0
        if trade_data.pnl > 0 and trade_data.trade_duration_minutes < 30:
            efficiency_bonus = 0.2
        
        total_reward = pnl_reward + drawdown_penalty + efficiency_bonus
        
        return max(-1.0, min(1.0, total_reward))  # Clamp to [-1, 1]
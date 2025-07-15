#!/usr/bin/env python3
"""
Main entry point for the Dopamine Trading System.
Implements clean architecture with full AI integration.
"""

import asyncio
import signal
import sys
from pathlib import Path
import structlog
import numpy

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.trading_system_integrated import IntegratedTradingSystem
from src.shared.utils import setup_logging, ensure_directory_exists
from src.shared.constants import (
    SUCCESS_SYSTEM_START, ERROR_CONFIG_LOAD, LOG_DIR, DATA_DIR, MODELS_DIR
)

logger = structlog.get_logger(__name__)


class DopamineTrader:
    """Main application class for the Dopamine Trading System."""
    
    def __init__(self, config_file: str = "config/system_config.json"):
        """Initialize the Dopamine Trading System.
        
        Args:
            config_file: Path to system configuration file
        """
        self.config_file = config_file
        self.trading_system = None
        self.shutdown_event = asyncio.Event()
        
    async def start(self) -> int:
        """Start the trading system.
        
        Returns:
            int: Exit code (0 for success, 1 for error)
        """
        try:
            # Setup initial logging first
            setup_logging("INFO", LOG_DIR)
            
            logger = structlog.get_logger(__name__)
            logger.info("Initializing Dopamine Trading System")
            
            # Ensure required directories exist
            ensure_directory_exists(LOG_DIR)
            ensure_directory_exists(DATA_DIR)
            ensure_directory_exists(MODELS_DIR)
            
            logger.debug("Loading system configuration")
            logger.debug("=================================")
            
            # Initialize trading system
            self.trading_system = IntegratedTradingSystem(self.config_file)
            logger.debug("Initializing AI components and neural networks")
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Start the system
            if not await self.trading_system.start():
                logger.error("System startup failed")
                return ERROR_CONFIG_LOAD
            
            logger.info("Dopamine Trading System is now online")
            logger.info("Monitoring markets and waiting for NinjaTrader connection")
            logger.info("Press Ctrl+C to shutdown")
            
            # Run main loop
            await self._run_main_loop()
            
            return 0
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            return 0
        except Exception as e:
            logger = structlog.get_logger(__name__)
            logger.error(f"Fatal error in main application: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            return 1
        finally:
            await self._cleanup()
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _run_main_loop(self) -> None:
        """Run the main application loop."""
        try:
            # Health check interval
            health_check_interval = 30.0  # seconds
            performance_optimization_interval = 300.0  # 5 minutes
            
            last_health_check = 0
            last_optimization = 0
            
            while not self.shutdown_event.is_set():
                current_time = asyncio.get_event_loop().time()
                
                # Periodic health checks
                if current_time - last_health_check > health_check_interval:
                    await self._health_check()
                    last_health_check = current_time
                
                # Periodic performance optimization
                if current_time - last_optimization > performance_optimization_interval:
                    await self._optimize_performance()
                    last_optimization = current_time
                
                # Check if system is still running
                if not self.trading_system.is_running:
                    logger.warning("Trading system stopped unexpectedly")
                    break
                
                # Sleep before next iteration
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error("Error in main loop", error=str(e))
            self.shutdown_event.set()
    
    async def _health_check(self) -> None:
        """Perform system health check."""
        try:
            if not self.trading_system:
                return
            
            # Get comprehensive system status
            status = self.trading_system.get_ai_status()
            
            # Log key metrics at debug level to reduce noise
            logger.debug(
                "System health check",
                state=status.get("state"),
                uptime=status.get("statistics", {}).get("uptime_seconds", 0),
                messages_processed=status.get("statistics", {}).get("messages_processed", 0),
                signals_sent=status.get("statistics", {}).get("signals_sent", 0),
                errors=status.get("statistics", {}).get("errors", 0),
                ninja_connected=status.get("ninja_connected", False),
                ai_processing=status.get("ai", {}).get("ai_processing", False),
                total_trades=status.get("ai", {}).get("performance_metrics", {}).get("total_trades", 0),
                profitable_trades=status.get("ai", {}).get("performance_metrics", {}).get("profitable_trades", 0)
            )
            
            # Check for concerning conditions
            ai_status = status.get("ai", {})
            if ai_status.get("performance_metrics", {}).get("total_trades", 0) > 10:
                success_rate = (ai_status.get("performance_metrics", {}).get("profitable_trades", 0) / 
                              ai_status.get("performance_metrics", {}).get("total_trades", 1))
                
                if success_rate < 0.3:
                    logger.warning("Low trading success rate detected", success_rate=success_rate)
                elif success_rate > 0.7:
                    logger.info("High trading success rate", success_rate=success_rate)
            
            # Check error rate
            total_messages = status.get("statistics", {}).get("messages_processed", 0)
            total_errors = status.get("statistics", {}).get("errors", 0)
            
            if total_messages > 0:
                error_rate = total_errors / total_messages
                if error_rate > 0.05:  # 5% error rate threshold
                    logger.warning("High error rate detected", error_rate=error_rate)
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
    
    async def _optimize_performance(self) -> None:
        """Optimize system performance."""
        try:
            if not self.trading_system:
                return
            
            logger.debug("Starting performance optimization cycle")
            
            # Run system optimization
            await self.trading_system.optimize_performance()
            
            # Log optimization results
            status = self.trading_system.get_ai_status()
            ai_status = status.get("ai", {})
            
            logger.debug(
                "Performance optimization cycle completed",
                networks=len(ai_status.get("networks", {})),
                subsystems=len(ai_status.get("subsystems", {}).get("enabled_subsystems", [])),
                total_trades=ai_status.get("performance_metrics", {}).get("total_trades", 0),
                avg_latency=ai_status.get("performance_metrics", {}).get("avg_latency_ms", 0)
            )
            
        except Exception as e:
            logger.error("Performance optimization failed", error=str(e))
    
    async def _cleanup(self) -> None:
        """Cleanup resources."""
        try:
            logger.info("Cleaning up resources")
            
            if self.trading_system:
                await self.trading_system.stop()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error("Cleanup failed", error=str(e))


async def main():
    """Main entry point."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Dopamine Trading System")
    parser.add_argument(
        "--config", 
        default="config/system_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Initialize and start the application
    app = DopamineTrader(config_file=args.config)
    exit_code = await app.start()
    
    return exit_code


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
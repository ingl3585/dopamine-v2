"""
Integrated Trading System with full AI component integration.
This extends the base trading system with complete AI functionality.
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional
import structlog

from .trading_system import TradingSystem
from ..shared.types import LiveDataMessage, HistoricalData, TradeCompletion, ActionType, State

logger = structlog.get_logger(__name__)


class IntegratedTradingSystem(TradingSystem):
    """Trading system with full AI integration."""
    
    def __init__(self, config_file: str = "config/system_config.json"):
        """Initialize integrated trading system."""
        super().__init__(config_file)
        
        # AI processing state
        self.ai_processing = False
        self.last_ai_decision_time = 0
        self.ai_decision_cooldown = 1.0  # seconds
        
    async def _process_with_ai(self, data: LiveDataMessage) -> None:
        """Process live data with AI components."""
        if self.ai_processing:
            logger.debug("AI processing already in progress, skipping")
            return  # Skip if already processing
        
        # Check cooldown
        current_time = asyncio.get_event_loop().time()
        if current_time - self.last_ai_decision_time < self.ai_decision_cooldown:
            logger.debug(
                "AI decision cooldown active",
                time_remaining=self.ai_decision_cooldown - (current_time - self.last_ai_decision_time)
            )
            return
        
        logger.info("Starting AI processing pipeline")
        self.ai_processing = True
        self.last_ai_decision_time = current_time
        
        try:
            # Get current market state
            logger.debug("Building current market state")
            
            # Debug: Check cache sizes before building state
            cache_info = {}
            for tf in ['1m', '5m', '15m', '1h', '4h']:
                cache_info[tf] = len(self.market_processor._price_cache[tf])
            
            total_bars = sum(cache_info.values())
            logger.debug(f"Building state from {total_bars} cached bars")
            
            current_state = self.market_processor.get_current_state()
            if not current_state:
                logger.warning("No current market state available for AI processing - insufficient data in cache")
                return
            
            logger.info(f"Market state built - {len(current_state.prices)} price features, {len(current_state.volumes)} volume features")
            
            # Process with AI subsystems
            logger.debug("Processing state with AI subsystems")
            subsystem_signals = await self.subsystem_manager.process_state(current_state)
            
            logger.info(
                "AI subsystems processed",
                signals_generated=len(subsystem_signals),
                active_subsystems=list(subsystem_signals.keys())
            )
            
            # Aggregate subsystem signals
            logger.debug("Aggregating subsystem signals")
            consensus_signal = self.subsystem_manager.aggregate_signals(subsystem_signals)
            
            if consensus_signal:
                logger.info(
                    "Consensus signal generated",
                    action=consensus_signal.action.name,
                    confidence=consensus_signal.confidence,
                    strength=consensus_signal.strength
                )
            else:
                logger.info("No consensus signal generated from subsystems")
            
            # Use RL agent for final decision
            if consensus_signal:
                logger.debug("Enhancing state with subsystem signals")
                # Enhance state with subsystem signals
                enhanced_state = self._enhance_state_with_signals(current_state, subsystem_signals)
                
                # Get RL agent decision
                logger.debug("Getting RL agent decision")
                action, confidence = self.rl_agent.select_action(enhanced_state)
                
                logger.info(
                    "RL agent decision",
                    action=action.name,
                    confidence=confidence
                )
                
                # Apply dopamine subsystem learning enhancement
                if "dopamine" in subsystem_signals:
                    dopamine_signal = subsystem_signals["dopamine"]
                    learning_enhancement = dopamine_signal.metadata.get("learning_rate_multiplier", 1.0)
                    confidence *= learning_enhancement
                
                # Send trading signal if confidence is high enough
                min_confidence = self.config.agent.get("min_confidence", 0.6)
                
                action_name = action.name if hasattr(action, 'name') else str(action)
                will_trade = confidence >= min_confidence and action != ActionType.HOLD
                logger.info(f"Trading decision: {action_name} (confidence: {confidence:.2f}, min: {min_confidence}) -> {'TRADE' if will_trade else 'NO TRADE'}")
                
                if confidence >= min_confidence and action != ActionType.HOLD:
                    position_size = self._calculate_position_size(confidence)
                    
                    logger.info(f"Sending {action.name} signal - confidence: {confidence:.2f}, size: {position_size}")
                    
                    success = self.send_trading_signal(
                        action=action,
                        confidence=confidence,
                        position_size=position_size
                    )
                    
                    if success:
                        logger.info("Trading signal sent successfully")
                        # Update performance metrics
                        self.performance_metrics["total_trades"] += 1
                        
                        # Train RL agent with experience
                        await self._train_rl_agent(enhanced_state, action, consensus_signal)
                    else:
                        logger.warning("Failed to send trading signal")
                else:
                    logger.info(
                        "No trading signal sent",
                        reason="confidence_too_low" if confidence < min_confidence else "action_is_hold"
                    )
                
                # Update subsystem performance
                self._update_subsystem_performance(subsystem_signals, consensus_signal)
                
                logger.debug(
                    "AI processing completed",
                    action=action.name,
                    confidence=confidence,
                    subsystems=len(subsystem_signals)
                )
        
        except Exception as e:
            logger.error(
                "AI processing failed", 
                error=str(e),
                error_type=type(e).__name__
            )
            # Log full traceback for debugging
            import traceback
            logger.error("AI processing traceback", traceback=traceback.format_exc())
            self.stats["errors"] += 1
        
        finally:
            self.ai_processing = False
    
    def _enhance_state_with_signals(self, state: State, signals: Dict[str, Any]) -> State:
        """Enhance market state with AI subsystem signals."""
        # Create enhanced state with subsystem signal features
        enhanced_subsystem_signals = []
        
        for signal_name, signal in signals.items():
            if signal:
                # Convert signal to numerical features
                signal_features = [
                    signal.confidence,
                    signal.strength,
                    float(signal.action.value),
                    len(signal.metadata),
                    signal.timestamp / 1e10  # Normalized timestamp
                ]
                enhanced_subsystem_signals.extend(signal_features)
        
        # Pad to expected dimension
        target_dim = 25  # STATE_SUBSYSTEM_DIM from constants
        while len(enhanced_subsystem_signals) < target_dim:
            enhanced_subsystem_signals.append(0.0)
        
        # Truncate if too long
        enhanced_subsystem_signals = enhanced_subsystem_signals[:target_dim]
        
        # Create enhanced state with numpy array for subsystem_signals
        enhanced_state = State(
            prices=state.prices,
            volumes=state.volumes,
            account_metrics=state.account_metrics,
            market_conditions=state.market_conditions,
            technical_indicators=state.technical_indicators,
            subsystem_signals=np.array(enhanced_subsystem_signals),  # Convert to numpy array
            timestamp=state.timestamp
        )
        
        return enhanced_state
    
    def _calculate_position_size(self, confidence: float) -> int:
        """Calculate position size based on confidence."""
        agent_config = self.config.agent
        base_size = agent_config.get("base_position_size", 1)
        max_size = agent_config.get("max_position_size", 5)
        
        # Scale position size with confidence
        scaled_size = int(base_size * (1 + confidence))
        return min(scaled_size, max_size)
    
    async def _train_rl_agent(self, state: State, action: ActionType, reward_signal: Any) -> None:
        """Train RL agent with recent experience."""
        try:
            # Calculate reward from consensus signal
            reward = reward_signal.confidence * reward_signal.strength
            if reward_signal.action != action:
                reward *= 0.5  # Penalty for disagreement
            
            # Store experience and train
            await self.rl_agent.store_experience(state, action, reward, state, False)  # Next state = current, not done
            
            # Trigger training if enough experience
            if self.rl_agent.should_train():
                await self.rl_agent.train_step()
        
        except Exception as e:
            logger.error("RL training failed", error=str(e))
    
    def _update_subsystem_performance(self, signals: Dict[str, Any], consensus: Any) -> None:
        """Update subsystem performance based on consensus."""
        try:
            performance_feedback = {}
            
            for name, signal in signals.items():
                if signal and consensus:
                    # Calculate performance based on agreement with consensus
                    agreement = 1.0 if signal.action == consensus.action else 0.0
                    confidence_bonus = signal.confidence * 0.5
                    performance = agreement + confidence_bonus
                    
                    performance_feedback[name] = performance
            
            # Update subsystem weights
            if performance_feedback:
                self.subsystem_manager.update_weights(performance_feedback)
        
        except Exception as e:
            logger.error("Performance update failed", error=str(e))
    
    async def _on_live_data(self, data: LiveDataMessage) -> None:
        """Handle live market data with AI processing."""
        try:
            logger.info(f"Processing live market data - Price: ${data.market_conditions.current_price}, Balance: ${data.account_info.account_balance:.2f}, Position: {data.account_info.total_position_size}")
            
            # Process with AI components if available  
            logger.debug(f"AI components ready: RL={bool(self.rl_agent)}, Subsystems={bool(self.subsystem_manager)}, Historical={self.has_historical_data}")
            
            if self.rl_agent and self.subsystem_manager and self.has_historical_data:
                logger.info("Processing live data with AI components")
                await self._process_with_ai(data)
            elif not self.has_historical_data:
                logger.info("Waiting for historical data before AI processing")
            elif not self.rl_agent:
                logger.warning("RL agent not initialized for AI processing")
            elif not self.subsystem_manager:
                logger.warning("Subsystem manager not initialized for AI processing")
        
        except Exception as e:
            logger.error("Failed to process live data", error=str(e))
            self.stats["errors"] += 1
    
    async def _on_historical_data(self, data: HistoricalData) -> None:
        """Handle historical data with AI initialization."""
        try:
            self.has_historical_data = True
            
            total_bars = (len(data.bars_1m) + len(data.bars_5m) + 
                         len(data.bars_15m) + len(data.bars_1h) + len(data.bars_4h))
            
            logger.info(
                "AI processing historical data",
                total_bars=total_bars,
                bars_1m=len(data.bars_1m),
                bars_5m=len(data.bars_5m),
                bars_15m=len(data.bars_15m),
                bars_1h=len(data.bars_1h),
                bars_4h=len(data.bars_4h)
            )
            
            # Initialize AI models with historical data
            if self.rl_agent:
                logger.info("Initializing AI models with historical data")
                await self._initialize_ai_with_history(data)
            else:
                logger.warning("RL agent not available for historical data initialization")
        
        except Exception as e:
            logger.error("Failed to process historical data", error=str(e))
            self.stats["errors"] += 1
    
    async def _initialize_ai_with_history(self, data: HistoricalData) -> None:
        """Initialize AI components with historical data."""
        try:
            # Process historical data for initial training
            states = []
            
            # Create states from historical bars
            for timeframe in ["1m", "5m", "15m", "1h", "4h"]:
                bars = getattr(data, f"bars_{timeframe}", [])
                if bars:
                    for bar in bars[-100:]:  # Last 100 bars
                        # Create basic state from bar data
                        state = State(
                            prices=[bar.close],
                            volumes=[bar.volume],
                            account_metrics=[0.0] * 10,  # Default account metrics
                            market_conditions=[0.0] * 5,  # Default market conditions
                            technical_indicators=[0.0] * 20,  # Default technical indicators
                            subsystem_signals=[0.0] * 25,  # Default subsystem signals
                            timestamp=bar.timestamp
                        )
                        states.append(state)
            
            # Pre-train AI components with historical states
            if states:
                logger.info(
                    "Initializing AI subsystems with historical states",
                    total_states=len(states),
                    initialization_states=min(50, len(states))
                )
                
                # Initialize subsystems with historical data
                successful_inits = 0
                for i, state in enumerate(states[-50:]):  # Last 50 states
                    try:
                        await self.subsystem_manager.process_state(state)
                        successful_inits += 1
                        if (i + 1) % 10 == 0:  # Log progress every 10 states
                            logger.debug(
                                "AI initialization progress",
                                completed=i + 1,
                                total=min(50, len(states)),
                                success_rate=f"{successful_inits/(i+1)*100:.1f}%"
                            )
                    except Exception as e:
                        logger.debug("Subsystem initialization step failed", step=i, error=str(e))
                
                logger.info(
                    "AI initialization completed",
                    successful_states=successful_inits,
                    total_processed=min(50, len(states)),
                    ai_ready=True
                )
            else:
                logger.warning("No historical states available for AI initialization")
        
        except Exception as e:
            logger.error("AI initialization failed", error=str(e))
    
    async def _on_trade_completion(self, data: TradeCompletion) -> None:
        """Handle trade completion with AI learning."""
        try:
            logger.info(
                "Trade completed",
                pnl=data.pnl,
                exit_reason=data.exit_reason,
                duration_minutes=data.trade_duration_minutes,
                price_move_pct=data.price_move_pct
            )
            
            # Update performance metrics
            self.performance_metrics["total_pnl"] += data.pnl
            if data.pnl > 0:
                self.performance_metrics["profitable_trades"] += 1
            
            # Update AI components with trade results
            if self.rl_agent:
                # Use trade P&L as reward signal
                reward = data.pnl / 100.0  # Normalize P&L
                await self.rl_agent.update_with_trade_result(reward)
            
            # Update subsystem performance based on trade outcome
            if data.pnl > 0:
                # Reward subsystems that contributed to profitable trade
                self._reward_successful_subsystems()
        
        except Exception as e:
            logger.error("Failed to process trade completion", error=str(e))
            self.stats["errors"] += 1
    
    def _reward_successful_subsystems(self) -> None:
        """Reward subsystems that contributed to successful trades."""
        try:
            # Get recent subsystem performance
            performance_boost = {}
            
            for name in self.subsystem_manager.subsystems.keys():
                # Give small performance boost to all active subsystems
                performance_boost[name] = 0.1
            
            # Update weights with performance boost
            self.subsystem_manager.update_weights(performance_boost)
            
            logger.debug("Subsystems rewarded for successful trade")
        
        except Exception as e:
            logger.error("Subsystem reward failed", error=str(e))
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get comprehensive AI system status."""
        status = self.get_status()
        
        # Add AI-specific status
        ai_status = {
            "ai_processing": self.ai_processing,
            "last_ai_decision": self.last_ai_decision_time,
            "performance_metrics": self.performance_metrics.copy()
        }
        
        # Add component-specific status
        if self.network_manager:
            ai_status["networks"] = self.network_manager.get_all_networks_info()
        
        if self.subsystem_manager:
            ai_status["subsystems"] = self.subsystem_manager.get_system_metrics()
        
        if self.rl_agent:
            ai_status["rl_agent"] = self.rl_agent.get_performance_metrics()
        
        status["ai"] = ai_status
        return status
    
    async def optimize_performance(self) -> None:
        """Optimize system performance."""
        try:
            logger.debug("Starting performance optimization")
            
            # Optimize neural networks
            if self.network_manager:
                # Trigger network adaptation
                for network_name in self.network_manager.networks.keys():
                    network = self.network_manager.networks[network_name]
                    if hasattr(network, 'adapt_architecture'):
                        performance_metrics = {"loss": 0.1}  # Placeholder
                        network.adapt_architecture(performance_metrics)
            
            # Optimize subsystem weights
            if self.subsystem_manager:
                # Get performance metrics and adjust weights
                metrics = self.subsystem_manager.get_system_metrics()
                logger.debug("Subsystem metrics", metrics=metrics)
            
            # Optimize RL agent parameters
            if self.rl_agent:
                # Adjust exploration rate based on performance
                if self.performance_metrics["total_trades"] > 10:
                    success_rate = (self.performance_metrics["profitable_trades"] / 
                                  self.performance_metrics["total_trades"])
                    
                    # Reduce exploration if performing well
                    if success_rate > 0.6:
                        self.rl_agent.reduce_exploration()
                    elif success_rate < 0.4:
                        self.rl_agent.increase_exploration()
            
            logger.debug("Network and subsystem optimization completed")
        
        except Exception as e:
            logger.error("Performance optimization failed", error=str(e))
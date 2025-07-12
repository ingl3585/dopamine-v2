"""
Reward calculation engine for the Dopamine Trading System.
Implements sophisticated multi-objective reward calculation.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque
import structlog

from ..shared.types import (
    TradeCompletion, RewardComponents, AccountInfo, MarketConditions
)
from ..shared.constants import (
    REWARD_PNL_WEIGHT, REWARD_RISK_WEIGHT, REWARD_EFFICIENCY_WEIGHT,
    REWARD_SURPRISE_WEIGHT, REWARD_CONSISTENCY_WEIGHT, RISK_FREE_RATE,
    SHARPE_RATIO_PERIODS
)

logger = structlog.get_logger(__name__)


class RewardEngine:
    """Multi-objective reward calculation engine."""
    
    def __init__(self, config: dict):
        """Initialize reward engine.
        
        Args:
            config: Reward engine configuration
        """
        self.config = config
        
        # Reward weights
        self.pnl_weight = config.get("pnl_weight", REWARD_PNL_WEIGHT)
        self.risk_weight = config.get("risk_weight", REWARD_RISK_WEIGHT)
        self.efficiency_weight = config.get("efficiency_weight", REWARD_EFFICIENCY_WEIGHT)
        self.surprise_weight = config.get("surprise_weight", REWARD_SURPRISE_WEIGHT)
        self.consistency_weight = config.get("consistency_weight", REWARD_CONSISTENCY_WEIGHT)
        
        # Performance tracking
        self.trade_history = deque(maxlen=1000)
        self.pnl_history = deque(maxlen=1000)
        self.sharpe_window = deque(maxlen=252)  # 1 year of daily returns
        
        # Baseline metrics
        self.baseline_pnl = 0.0
        self.baseline_volatility = 0.02
        self.expected_return = 0.0
        
        # Adaptive learning
        self.learning_rate = 0.01
        self.surprise_threshold = 0.1
        
        logger.info("Reward engine initialized", weights={
            "pnl": self.pnl_weight,
            "risk": self.risk_weight,
            "efficiency": self.efficiency_weight,
            "surprise": self.surprise_weight,
            "consistency": self.consistency_weight
        })
    
    def calculate_reward(self, trade_completion: TradeCompletion,
                        account_info: AccountInfo,
                        market_conditions: MarketConditions) -> RewardComponents:
        """Calculate comprehensive reward from trade completion.
        
        Args:
            trade_completion: Completed trade information
            account_info: Current account state
            market_conditions: Current market conditions
            
        Returns:
            RewardComponents: Decomposed reward signal
        """
        try:
            # Store trade for history
            self.trade_history.append(trade_completion)
            self.pnl_history.append(trade_completion.pnl)
            
            # Calculate individual reward components
            pnl_reward = self._calculate_pnl_reward(trade_completion)
            risk_penalty = self._calculate_risk_penalty(trade_completion, account_info)
            efficiency_bonus = self._calculate_efficiency_bonus(trade_completion)
            surprise_bonus = self._calculate_surprise_bonus(trade_completion)
            consistency_bonus = self._calculate_consistency_bonus()
            
            # Combine weighted rewards
            total_reward = (
                self.pnl_weight * pnl_reward +
                self.risk_weight * risk_penalty +
                self.efficiency_weight * efficiency_bonus +
                self.surprise_weight * surprise_bonus +
                self.consistency_weight * consistency_bonus
            )
            
            # Clamp total reward
            total_reward = np.clip(total_reward, -1.0, 1.0)
            
            reward_components = RewardComponents(
                pnl_reward=pnl_reward,
                risk_penalty=risk_penalty,
                efficiency_bonus=efficiency_bonus,
                surprise_bonus=surprise_bonus,
                consistency_bonus=consistency_bonus,
                total_reward=total_reward
            )
            
            # Update baselines
            self._update_baselines(trade_completion)
            
            logger.debug(
                "Reward calculated",
                pnl=trade_completion.pnl,
                total_reward=total_reward,
                components={
                    "pnl": pnl_reward,
                    "risk": risk_penalty,
                    "efficiency": efficiency_bonus,
                    "surprise": surprise_bonus,
                    "consistency": consistency_bonus
                }
            )
            
            return reward_components
            
        except Exception as e:
            logger.error("Failed to calculate reward", error=str(e))
            return RewardComponents(
                pnl_reward=0.0, risk_penalty=0.0, efficiency_bonus=0.0,
                surprise_bonus=0.0, consistency_bonus=0.0, total_reward=0.0
            )
    
    def _calculate_pnl_reward(self, trade: TradeCompletion) -> float:
        """Calculate P&L-based reward component.
        
        Args:
            trade: Trade completion data
            
        Returns:
            float: P&L reward component
        """
        # Normalize P&L to reasonable range
        normalized_pnl = trade.pnl / 1000.0  # Assume $1000 is significant
        
        # Risk-adjusted return (account for volatility)
        if trade.volatility > 0:
            risk_adjusted_pnl = normalized_pnl / max(trade.volatility, 0.01)
        else:
            risk_adjusted_pnl = normalized_pnl
        
        # Apply diminishing returns to prevent extreme values
        if risk_adjusted_pnl > 0:
            reward = np.tanh(risk_adjusted_pnl)
        else:
            reward = -np.tanh(abs(risk_adjusted_pnl)) * 1.5  # Penalty amplification
        
        return np.clip(reward, -1.0, 1.0)
    
    def _calculate_risk_penalty(self, trade: TradeCompletion, 
                               account: AccountInfo) -> float:
        """Calculate risk-based penalty component.
        
        Args:
            trade: Trade completion data
            account: Account information
            
        Returns:
            float: Risk penalty (negative for high risk)
        """
        penalty = 0.0
        
        # Drawdown penalty
        if account.daily_pnl < 0:
            drawdown_pct = abs(account.daily_pnl) / max(account.account_balance, 1.0)
            if drawdown_pct > 0.02:  # More than 2% daily drawdown
                penalty -= drawdown_pct * 10.0
        
        # Position size penalty (for oversized positions)
        if account.total_position_size > 5:  # More than 5 contracts
            penalty -= (account.total_position_size - 5) * 0.1
        
        # Volatility penalty during high volatility periods
        if trade.volatility > 0.05:  # High volatility threshold
            penalty -= (trade.volatility - 0.05) * 5.0
        
        # Margin usage penalty
        if account.margin_used > 0:
            margin_ratio = account.margin_used / max(account.net_liquidation, 1.0)
            if margin_ratio > 0.5:  # Using more than 50% of capital
                penalty -= (margin_ratio - 0.5) * 2.0
        
        return np.clip(penalty, -1.0, 0.0)
    
    def _calculate_efficiency_bonus(self, trade: TradeCompletion) -> float:
        """Calculate efficiency-based bonus component.
        
        Args:
            trade: Trade completion data
            
        Returns:
            float: Efficiency bonus
        """
        bonus = 0.0
        
        # Time efficiency bonus (quick profitable trades)
        if trade.pnl > 0 and trade.trade_duration_minutes < 60:
            time_bonus = (60 - trade.trade_duration_minutes) / 60.0
            bonus += time_bonus * 0.2
        
        # Price move efficiency (capturing good moves)
        if trade.price_move_pct > 0.01:  # At least 1% move
            if (trade.pnl > 0 and trade.exit_reason != "stop_hit"):
                # Rewarded for capturing positive moves
                move_efficiency = min(trade.price_move_pct / 0.05, 1.0)  # Max at 5% move
                bonus += move_efficiency * 0.3
        
        # Exit reason bonus (clean exits are preferred)
        if trade.exit_reason == "target_hit":
            bonus += 0.2
        elif trade.exit_reason == "ai_exit":
            bonus += 0.1
        elif trade.exit_reason == "stop_hit":
            bonus -= 0.1  # Slight penalty for stop hits
        
        return np.clip(bonus, 0.0, 1.0)
    
    def _calculate_surprise_bonus(self, trade: TradeCompletion) -> float:
        """Calculate surprise-based bonus (dopamine-inspired).
        
        Args:
            trade: Trade completion data
            
        Returns:
            float: Surprise bonus
        """
        if len(self.pnl_history) < 10:
            return 0.0
        
        # Calculate expected P&L based on recent history
        recent_pnl = list(self.pnl_history)[-10:]
        expected_pnl = np.mean(recent_pnl)
        
        # Surprise is deviation from expectation
        surprise = trade.pnl - expected_pnl
        
        # Positive surprise gets bonus, negative gets penalty
        if abs(surprise) > self.surprise_threshold * 1000:  # $threshold
            if surprise > 0:
                bonus = min(surprise / 2000.0, 0.5)  # Cap at 0.5
            else:
                bonus = max(surprise / 1000.0, -0.3)  # Smaller penalty
            
            # Update expectation with learning
            self.expected_return += self.learning_rate * surprise
            
            return bonus
        
        return 0.0
    
    def _calculate_consistency_bonus(self) -> float:
        """Calculate consistency-based bonus.
        
        Returns:
            float: Consistency bonus
        """
        if len(self.pnl_history) < 20:
            return 0.0
        
        recent_pnl = list(self.pnl_history)[-20:]
        
        # Reward consistent positive performance
        positive_trades = sum(1 for pnl in recent_pnl if pnl > 0)
        consistency_ratio = positive_trades / len(recent_pnl)
        
        # Bonus for high win rate
        if consistency_ratio > 0.6:
            bonus = (consistency_ratio - 0.6) * 2.0  # Scale above 60% win rate
        else:
            bonus = 0.0
        
        # Penalty for high volatility in returns
        pnl_std = np.std(recent_pnl)
        if pnl_std > 500:  # High volatility in P&L
            bonus -= (pnl_std - 500) / 2000.0
        
        return np.clip(bonus, -0.3, 0.5)
    
    def _update_baselines(self, trade: TradeCompletion) -> None:
        """Update baseline metrics for adaptive learning.
        
        Args:
            trade: Trade completion data
        """
        # Update baseline P&L with exponential moving average
        if self.baseline_pnl == 0.0:
            self.baseline_pnl = trade.pnl
        else:
            alpha = 0.1  # Learning rate for baseline
            self.baseline_pnl = alpha * trade.pnl + (1 - alpha) * self.baseline_pnl
        
        # Update baseline volatility
        if trade.volatility > 0:
            if self.baseline_volatility == 0.02:
                self.baseline_volatility = trade.volatility
            else:
                alpha = 0.05  # Slower adaptation for volatility
                self.baseline_volatility = (alpha * trade.volatility + 
                                          (1 - alpha) * self.baseline_volatility)
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from recent performance.
        
        Returns:
            float: Sharpe ratio
        """
        if len(self.pnl_history) < 30:
            return 0.0
        
        returns = list(self.pnl_history)[-252:]  # Up to 1 year
        
        if len(returns) < 10:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize assuming 252 trading days
        sharpe = (mean_return * np.sqrt(252)) / (std_return * np.sqrt(252))
        
        return sharpe
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get reward engine performance metrics.
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        if not self.pnl_history:
            return {
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "baseline_pnl": self.baseline_pnl,
                "baseline_volatility": self.baseline_volatility
            }
        
        pnl_list = list(self.pnl_history)
        total_pnl = sum(pnl_list)
        avg_pnl = np.mean(pnl_list)
        win_rate = sum(1 for pnl in pnl_list if pnl > 0) / len(pnl_list)
        sharpe = self.calculate_sharpe_ratio()
        
        return {
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "baseline_pnl": self.baseline_pnl,
            "baseline_volatility": self.baseline_volatility,
            "trade_count": len(self.trade_history)
        }
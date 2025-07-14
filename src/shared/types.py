"""
Shared type definitions and data classes for the Dopamine Trading System.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import numpy as np


class MarketPosition(Enum):
    """Market position states."""
    FLAT = 0
    LONG = 1
    SHORT = 2


class ActionType(Enum):
    """Trading action types."""
    HOLD = 0
    BUY = 1
    SELL = 2


class MessageType(Enum):
    """TCP message types from NinjaTrader."""
    HISTORICAL_DATA = "historical_data"
    LIVE_DATA = "live_data"
    TRADE_COMPLETION = "trade_completion"


class SignalType(Enum):
    """AI subsystem signal types."""
    DNA = "dna"
    TEMPORAL = "temporal"
    IMMUNE = "immune"
    MICROSTRUCTURE = "microstructure"
    DOPAMINE = "dopamine"


@dataclass
class BarData:
    """Single bar/candle data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class MarketData:
    """Multi-timeframe market data structure."""
    price_1m: List[float]
    price_5m: List[float]
    price_15m: List[float]
    price_1h: List[float]
    price_4h: List[float]
    volume_1m: List[float]
    volume_5m: List[float]
    volume_15m: List[float]
    volume_1h: List[float]
    volume_4h: List[float]
    timestamp: int


@dataclass
class AccountInfo:
    """Account and portfolio information."""
    account_balance: float
    buying_power: float
    daily_pnl: float
    unrealized_pnl: float
    net_liquidation: float
    margin_used: float
    available_margin: float
    open_positions: int
    total_position_size: int


@dataclass
class MarketConditions:
    """Current market condition metrics."""
    current_price: float
    volatility: float
    drawdown_pct: float
    portfolio_heat: float
    regime: str
    trend_strength: float


@dataclass
class LiveDataMessage:
    """Complete live data message from NinjaTrader."""
    type: str
    market_data: MarketData
    account_info: AccountInfo
    market_conditions: MarketConditions


@dataclass
class HistoricalData:
    """Historical data from NinjaTrader."""
    type: str
    bars_1m: List[BarData]
    bars_5m: List[BarData]
    bars_15m: List[BarData]
    bars_1h: List[BarData]
    bars_4h: List[BarData]
    timestamp: int


@dataclass
class TradeCompletion:
    """Trade completion data from NinjaTrader."""
    type: str
    pnl: float
    exit_price: float
    entry_price: float
    size: int
    exit_reason: str
    entry_time: int
    exit_time: int
    account_balance: float
    net_liquidation: float
    margin_used: float
    daily_pnl: float
    trade_duration_minutes: float
    price_move_pct: float
    volatility: float
    regime: str
    trend_strength: float
    confidence: float
    consensus_strength: float
    primary_tool: str
    timestamp: int


@dataclass
class AISignal:
    """Signal from an AI subsystem."""
    signal_type: SignalType
    action: ActionType
    confidence: float
    strength: float
    metadata: Dict[str, Any]
    timestamp: int


@dataclass
class TradingSignal:
    """Final trading signal to send to NinjaTrader."""
    action: int  # 0=hold, 1=buy, 2=sell
    confidence: float
    position_size: int
    use_stop: bool
    stop_price: float
    use_target: bool
    target_price: float


@dataclass
class State:
    """RL agent state representation."""
    prices: np.ndarray
    volumes: np.ndarray
    account_metrics: np.ndarray
    market_conditions: np.ndarray
    technical_indicators: np.ndarray
    subsystem_signals: np.ndarray
    timestamp: int


@dataclass
class Experience:
    """RL experience tuple for replay buffer."""
    state: State
    action: ActionType
    reward: float
    next_state: State
    done: bool
    timestamp: int


@dataclass
class RewardComponents:
    """Decomposed reward signal components."""
    pnl_reward: float
    risk_penalty: float
    efficiency_bonus: float
    surprise_bonus: float
    consistency_bonus: float
    total_reward: float


@dataclass
class SubsystemConfig:
    """Configuration for an AI subsystem."""
    enabled: bool
    weight: float
    parameters: Dict[str, Any]


@dataclass
class SystemConfig:
    """Complete system configuration."""
    system: Dict[str, Any]
    agent: Dict[str, Any]
    subsystems: Dict[str, SubsystemConfig]
    neural: Dict[str, Any]
    risk: Dict[str, Any]
    data: Dict[str, Any]


# Type aliases for common data structures
PriceArray = List[float]
VolumeArray = List[float]
TimeframeData = Dict[str, Union[PriceArray, VolumeArray]]
SignalWeights = Dict[SignalType, float]
NetworkWeights = Dict[str, np.ndarray]
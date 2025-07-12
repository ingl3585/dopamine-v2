"""
Utility functions and helpers for the Dopamine Trading System.
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
import structlog
import os

from .types import BarData, MarketData, AccountInfo, MarketConditions
from .constants import (
    JSON_ENCODING, TIMEFRAMES, MIN_PRICE_VALUE, MAX_PRICE_VALUE,
    MIN_VOLUME_VALUE, MAX_VOLUME_VALUE, FEATURE_WINDOWS
)

# Configure structured logging
logger = structlog.get_logger(__name__)


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Set up structured logging for the system."""
    os.makedirs(log_dir, exist_ok=True)
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def validate_price_data(prices: List[float]) -> bool:
    """Validate price data for reasonable ranges and no NaN values."""
    if not prices:
        return False
    
    for price in prices:
        if not isinstance(price, (int, float)):
            return False
        if np.isnan(price) or np.isinf(price):
            return False
        if price < MIN_PRICE_VALUE or price > MAX_PRICE_VALUE:
            return False
    
    return True


def validate_volume_data(volumes: List[float]) -> bool:
    """Validate volume data for reasonable ranges and no NaN values."""
    if not volumes:
        return False
    
    for volume in volumes:
        if not isinstance(volume, (int, float)):
            return False
        if np.isnan(volume) or np.isinf(volume):
            return False
        if volume < MIN_VOLUME_VALUE or volume > MAX_VOLUME_VALUE:
            return False
    
    return True


def normalize_array(data: np.ndarray, method: str = "z_score") -> np.ndarray:
    """Normalize array using specified method."""
    if len(data) == 0:
        return data
    
    if method == "z_score":
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std
    
    elif method == "min_max":
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    
    elif method == "robust":
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return np.zeros_like(data)
        return (data - median) / mad
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_returns(prices: List[float]) -> List[float]:
    """Calculate simple returns from price series."""
    if len(prices) < 2:
        return []
    
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        else:
            returns.append(0.0)
    
    return returns


def calculate_log_returns(prices: List[float]) -> List[float]:
    """Calculate log returns from price series."""
    if len(prices) < 2:
        return []
    
    returns = []
    for i in range(1, len(prices)):
        if prices[i] > 0 and prices[i-1] > 0:
            ret = np.log(prices[i] / prices[i-1])
            returns.append(ret)
        else:
            returns.append(0.0)
    
    return returns


def calculate_volatility(returns: List[float], window: int = 20) -> float:
    """Calculate rolling volatility from returns."""
    if len(returns) < window:
        return 0.0
    
    recent_returns = returns[-window:]
    return np.std(recent_returns) * np.sqrt(252)  # Annualized


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from returns."""
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns) * 252  # Annualized
    volatility = np.std(returns) * np.sqrt(252)  # Annualized
    
    if volatility == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / volatility


def calculate_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    
    peak = equity_curve[0]
    max_drawdown = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        
        drawdown = (peak - value) / peak if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown


def extract_features(prices: List[float], volumes: List[float]) -> Dict[str, List[float]]:
    """Extract raw features from price and volume data without classic indicators."""
    features = {}
    
    if len(prices) < max(FEATURE_WINDOWS):
        return features
    
    # Price-based features
    features["returns"] = calculate_returns(prices)
    features["log_returns"] = calculate_log_returns(prices)
    
    # Volatility features
    returns = calculate_returns(prices)
    for window in FEATURE_WINDOWS:
        if len(returns) >= window:
            vol = pd.Series(returns).rolling(window=window).std().tolist()
            features[f"volatility_{window}"] = vol
    
    # Momentum features
    for window in FEATURE_WINDOWS:
        if len(prices) >= window + 1:
            momentum = []
            for i in range(window, len(prices)):
                mom = (prices[i] - prices[i-window]) / prices[i-window] if prices[i-window] != 0 else 0.0
                momentum.append(mom)
            features[f"momentum_{window}"] = [0.0] * window + momentum
    
    # Raw volume features (essential - kept as requested)
    if len(volumes) >= max(FEATURE_WINDOWS):
        # Volume returns instead of moving averages
        volume_returns = calculate_returns(volumes)
        features["volume_returns"] = volume_returns
        
        # Volume momentum over different windows
        for window in FEATURE_WINDOWS:
            if len(volume_returns) >= window:
                vol_momentum = []
                for i in range(window, len(volume_returns)):
                    momentum = sum(volume_returns[i-window:i]) / window
                    vol_momentum.append(momentum)
                features[f"volume_momentum_{window}"] = [0.0] * window + vol_momentum
    
    return features


def parse_json_message(message: str) -> Optional[Dict[str, Any]]:
    """Parse JSON message with error handling."""
    try:
        return json.loads(message)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON message", error=str(e), message=message[:100])
        return None


def serialize_to_json(data: Dict[str, Any]) -> str:
    """Serialize data to JSON string with error handling."""
    try:
        return json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.error("Failed to serialize to JSON", error=str(e))
        return "{}"


def timestamp_to_datetime(timestamp: int) -> datetime:
    """Convert timestamp to datetime object."""
    return datetime.fromtimestamp(timestamp / 1e7 - 62135596800, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> int:
    """Convert datetime to .NET ticks timestamp."""
    return int((dt.timestamp() + 62135596800) * 1e7)


def ensure_directory_exists(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Perform safe division with default value for zero denominator."""
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(value, max_val))
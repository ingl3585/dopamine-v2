"""
System constants and enums for the Dopamine Trading System.
"""

from typing import Dict, List

# TCP Communication
TCP_HOST = "localhost"
TCP_PORT_DATA = 5556
TCP_PORT_SIGNALS = 5557
TCP_BUFFER_SIZE = 8192
TCP_TIMEOUT = 30.0

# Message Protocol
MESSAGE_HEADER_SIZE = 4
MAX_MESSAGE_SIZE = 10000
JSON_ENCODING = "utf-8"

# Timeframes
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]
TIMEFRAME_MULTIPLIERS = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240
}

# Data Management
MAX_HISTORY_DAYS = 30
DEFAULT_HISTORY_DAYS = 10
MAX_CACHE_SIZE = 10000
MAX_PRICE_HISTORY = 1000

# RL Agent
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_EXPLORATION_RATE = 0.1
DEFAULT_BATCH_SIZE = 32
DEFAULT_MEMORY_SIZE = 10000
MIN_REPLAY_SIZE = 1000

# Neural Networks
DEFAULT_HIDDEN_LAYERS = [256, 128, 64]
DEFAULT_ACTIVATION = "relu"
DEFAULT_DROPOUT_RATE = 0.2
NETWORK_INPUT_DIM = 100
NETWORK_OUTPUT_DIM = 3  # Hold, Buy, Sell

# AI Subsystems
SUBSYSTEM_NAMES = ["dna", "temporal", "immune", "microstructure", "dopamine"]
DEFAULT_SUBSYSTEM_WEIGHT = 0.2
MIN_CONFIDENCE_THRESHOLD = 0.1
MAX_CONFIDENCE_THRESHOLD = 0.9

# DNA Subsystem
DNA_DEFAULT_PATTERN_LENGTH = 10
DNA_DEFAULT_MUTATION_RATE = 0.01
DNA_MAX_PATTERNS = 1000

# Temporal Subsystem
TEMPORAL_DEFAULT_CYCLES = [5, 15, 60, 240]
TEMPORAL_DEFAULT_LOOKBACK = 100
TEMPORAL_MIN_CYCLE_LENGTH = 3

# Immune Subsystem
IMMUNE_DEFAULT_THRESHOLD = 2.0
IMMUNE_DEFAULT_ADAPTATION_RATE = 0.05
IMMUNE_MEMORY_SIZE = 500

# Microstructure Subsystem
MICROSTRUCTURE_DEFAULT_WINDOW = 50
MICROSTRUCTURE_DEFAULT_VOL_THRESHOLD = 0.02
MICROSTRUCTURE_REGIME_TYPES = ["trending", "ranging", "volatile", "calm"]

# Dopamine Subsystem
DOPAMINE_DEFAULT_BASELINE = 0.0
DOPAMINE_DEFAULT_SURPRISE_THRESHOLD = 0.1
DOPAMINE_DECAY_FACTOR = 0.95

# Risk Management
DEFAULT_MAX_POSITION_SIZE = 10
DEFAULT_MAX_DAILY_LOSS = 1000.0
DEFAULT_MAX_DRAWDOWN_PCT = 5.0
DEFAULT_VOLATILITY_LIMIT = 0.05
RISK_FREE_RATE = 0.02  # 2% annual

# Reward Calculation
REWARD_PNL_WEIGHT = 1.0
REWARD_RISK_WEIGHT = 0.5
REWARD_EFFICIENCY_WEIGHT = 0.3
REWARD_SURPRISE_WEIGHT = 0.2
REWARD_CONSISTENCY_WEIGHT = 0.1

# Performance Metrics
SHARPE_RATIO_PERIODS = 252  # Trading days per year
MAX_ACCEPTABLE_LATENCY_MS = 100
TARGET_STARTUP_TIME_S = 10
TARGET_MEMORY_USAGE_GB = 2
TARGET_CPU_USAGE_PCT = 50

# Logging
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# File Paths
CONFIG_FILE = "config/system_config.json"
LOG_DIR = "logs"
DATA_DIR = "data"
MODELS_DIR = "models"

# Market Conditions
MARKET_REGIMES = ["bull", "bear", "sideways", "volatile"]
TREND_STRENGTHS = ["weak", "moderate", "strong"]
VOLATILITY_REGIMES = ["low", "normal", "high", "extreme"]

# Technical Indicators
INDICATOR_PERIODS = {
    "sma_fast": 10,
    "sma_slow": 20,
    "ema_fast": 12,
    "ema_slow": 26,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0
}

# State Space Dimensions
STATE_PRICE_DIM = 50
STATE_VOLUME_DIM = 50
STATE_ACCOUNT_DIM = 10
STATE_MARKET_DIM = 5
STATE_TECHNICAL_DIM = 20
STATE_SUBSYSTEM_DIM = 25

# Action Space
ACTION_SPACE_SIZE = 3  # Hold, Buy, Sell
POSITION_SIZE_RANGE = (1, 10)
CONFIDENCE_RANGE = (0.0, 1.0)

# Error Codes
ERROR_TCP_CONNECTION = 1001
ERROR_DATA_VALIDATION = 1002
ERROR_CONFIG_LOAD = 1003
ERROR_MODEL_LOAD = 1004
ERROR_INSUFFICIENT_DATA = 1005
ERROR_RISK_VIOLATION = 1006

# Success Codes
SUCCESS_SYSTEM_START = 2001
SUCCESS_TCP_CONNECTED = 2002
SUCCESS_DATA_PROCESSED = 2003
SUCCESS_SIGNAL_SENT = 2004
SUCCESS_MODEL_UPDATED = 2005

# System States
SYSTEM_STATES = ["initializing", "connecting", "loading", "running", "stopping", "error"]

# Data Validation
MIN_PRICE_VALUE = 0.01
MAX_PRICE_VALUE = 1000000.0
MIN_VOLUME_VALUE = 0
MAX_VOLUME_VALUE = 1000000000
MIN_ACCOUNT_BALANCE = 1000.0

# Feature Engineering
FEATURE_WINDOWS = [5, 10, 20, 50]
FEATURE_TYPES = ["returns", "volatility", "momentum", "mean_reversion"]
NORMALIZATION_METHODS = ["z_score", "min_max", "robust"]

# Model Persistence
MODEL_SAVE_FREQUENCY = 1000  # steps
MODEL_CHECKPOINT_KEEP = 5
MODEL_FILE_EXTENSION = ".pth"

# Performance Monitoring
METRICS_UPDATE_FREQUENCY = 100  # steps
PERFORMANCE_WINDOW = 1000  # for rolling metrics
ALERT_THRESHOLDS = {
    "latency_ms": 500,
    "memory_gb": 3.0,
    "cpu_pct": 80.0,
    "error_rate": 0.05
}
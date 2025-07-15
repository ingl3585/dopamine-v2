"""
Configuration manager for the Dopamine Trading System.
Handles loading, validation, and management of system configuration.
"""

import json
import os
from typing import Dict, Any, Optional
import structlog

from ..shared.types import SystemConfig, SubsystemConfig
from ..shared.constants import CONFIG_FILE, DEFAULT_LOG_LEVEL
from ..shared.utils import ensure_directory_exists

logger = structlog.get_logger(__name__)


class ConfigManager:
    """Manages system configuration with validation and defaults."""
    
    def __init__(self, config_file: str = CONFIG_FILE):
        """Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self._config: Optional[SystemConfig] = None
        self._defaults = self._get_default_config()
    
    def load_config(self) -> SystemConfig:
        """Load configuration from file with validation and defaults.
        
        Returns:
            SystemConfig: Validated system configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        try:
            if not os.path.exists(self.config_file):
                logger.warning(
                    "Config file not found, creating default config",
                    config_file=self.config_file
                )
                self._create_default_config()
            
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Merge with defaults and validate
            merged_config = self._merge_with_defaults(config_data)
            validated_config = self._validate_config(merged_config)
            
            self._config = self._dict_to_system_config(validated_config)
            
            logger.debug(
                "Config file parsed and validated",
                config_file=self.config_file,
                environment=self._config.system.get("environment")
            )
            
            return self._config
            
        except Exception as e:
            logger.error(
                "Failed to load configuration",
                config_file=self.config_file,
                error=str(e)
            )
            raise
    
    def get_config(self) -> SystemConfig:
        """Get current configuration, loading if necessary.
        
        Returns:
            SystemConfig: Current system configuration
        """
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def get_subsystem_config(self, subsystem_name: str) -> SubsystemConfig:
        """Get configuration for a specific subsystem.
        
        Args:
            subsystem_name: Name of the subsystem
            
        Returns:
            SubsystemConfig: Subsystem configuration
            
        Raises:
            KeyError: If subsystem not found in configuration
        """
        config = self.get_config()
        if subsystem_name not in config.subsystems:
            raise KeyError(f"Subsystem '{subsystem_name}' not found in configuration")
        
        return config.subsystems[subsystem_name]
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        if self._config is None:
            self.load_config()
        
        # Apply updates to current config
        config_dict = self._system_config_to_dict(self._config)
        config_dict = self._deep_update(config_dict, updates)
        
        # Validate and save
        validated_config = self._validate_config(config_dict)
        self._config = self._dict_to_system_config(validated_config)
        self.save_config()
        
        logger.info("Configuration updated", updates=updates)
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        if self._config is None:
            raise ValueError("No configuration loaded to save")
        
        config_dict = self._system_config_to_dict(self._config)
        
        # Ensure config directory exists
        config_dir = os.path.dirname(self.config_file)
        if config_dir:
            ensure_directory_exists(config_dir)
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info("Configuration saved", config_file=self.config_file)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "system": {
                "environment": "development",
                "log_level": DEFAULT_LOG_LEVEL,
                "tcp_host": "localhost",
                "tcp_port_data": 5556,
                "tcp_port_signals": 5557
            },
            "agent": {
                "learning_rate": 0.001,
                "discount_factor": 0.99,
                "exploration_rate": 0.1,
                "batch_size": 32,
                "memory_size": 10000,
                "update_frequency": 100
            },
            "subsystems": {
                "dna": {
                    "enabled": True,
                    "weight": 0.2,
                    "pattern_length": 10,
                    "mutation_rate": 0.01
                },
                "temporal": {
                    "enabled": True,
                    "weight": 0.2,
                    "cycle_lengths": [5, 15, 60, 240],
                    "lookback_periods": 100
                },
                "immune": {
                    "enabled": True,
                    "weight": 0.2,
                    "anomaly_threshold": 2.0,
                    "adaptation_rate": 0.05
                },
                "microstructure": {
                    "enabled": True,
                    "weight": 0.2,
                    "regime_detection_window": 50,
                    "volatility_threshold": 0.02
                },
                "dopamine": {
                    "enabled": True,
                    "weight": 0.2,
                    "baseline_reward": 0.0,
                    "surprise_threshold": 0.1
                }
            },
            "neural": {
                "hidden_layers": [256, 128, 64],
                "activation": "relu",
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "batch_size": 64
            },
            "risk": {
                "max_position_size": 10,
                "max_daily_loss": 1000.0,
                "max_drawdown_pct": 5.0,
                "volatility_limit": 0.05
            },
            "data": {
                "timeframes": ["1m", "5m", "15m", "1h", "4h"],
                "history_days": 10,
                "cache_size": 10000
            }
        }
    
    def _create_default_config(self) -> None:
        """Create default configuration file."""
        config_dir = os.path.dirname(self.config_file)
        if config_dir:
            ensure_directory_exists(config_dir)
        
        with open(self.config_file, 'w') as f:
            json.dump(self._defaults, f, indent=2)
        
        logger.info("Default configuration file created", config_file=self.config_file)
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with defaults."""
        merged = self._defaults.copy()
        merged = self._deep_update(merged, config)
        return merged
    
    def _deep_update(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Deep update dictionary with another dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration values."""
        # Validate system section
        if "system" not in config:
            raise ValueError("Missing 'system' section in configuration")
        
        system = config["system"]
        if not isinstance(system.get("tcp_port_data"), int):
            raise ValueError("tcp_port_data must be an integer")
        if not isinstance(system.get("tcp_port_signals"), int):
            raise ValueError("tcp_port_signals must be an integer")
        
        # Validate agent section
        if "agent" not in config:
            raise ValueError("Missing 'agent' section in configuration")
        
        agent = config["agent"]
        if not 0 < agent.get("learning_rate", 0) <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 < agent.get("discount_factor", 0) <= 1:
            raise ValueError("discount_factor must be between 0 and 1")
        if not 0 <= agent.get("exploration_rate", 0) <= 1:
            raise ValueError("exploration_rate must be between 0 and 1")
        
        # Validate subsystems section
        if "subsystems" not in config:
            raise ValueError("Missing 'subsystems' section in configuration")
        
        subsystems = config["subsystems"]
        total_weight = 0
        for name, subsystem in subsystems.items():
            if not isinstance(subsystem.get("enabled"), bool):
                raise ValueError(f"Subsystem '{name}' enabled must be boolean")
            if not 0 <= subsystem.get("weight", 0) <= 1:
                raise ValueError(f"Subsystem '{name}' weight must be between 0 and 1")
            if subsystem.get("enabled", False):
                total_weight += subsystem.get("weight", 0)
        
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            logger.warning(
                "Subsystem weights do not sum to 1.0",
                total_weight=total_weight
            )
        
        # Validate neural section
        if "neural" not in config:
            raise ValueError("Missing 'neural' section in configuration")
        
        neural = config["neural"]
        if not isinstance(neural.get("hidden_layers"), list):
            raise ValueError("hidden_layers must be a list")
        
        # Validate risk section
        if "risk" not in config:
            raise ValueError("Missing 'risk' section in configuration")
        
        risk = config["risk"]
        if not risk.get("max_position_size", 0) > 0:
            raise ValueError("max_position_size must be positive")
        
        return config
    
    def _dict_to_system_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig object."""
        # Convert subsystems to SubsystemConfig objects
        subsystems = {}
        for name, subsystem_dict in config_dict["subsystems"].items():
            subsystems[name] = SubsystemConfig(
                enabled=subsystem_dict["enabled"],
                weight=subsystem_dict["weight"],
                parameters={k: v for k, v in subsystem_dict.items() 
                           if k not in ["enabled", "weight"]}
            )
        
        return SystemConfig(
            system=config_dict["system"],
            agent=config_dict["agent"],
            subsystems=subsystems,
            neural=config_dict["neural"],
            risk=config_dict["risk"],
            data=config_dict["data"]
        )
    
    def _system_config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """Convert SystemConfig object to dictionary."""
        # Convert subsystems back to dictionaries
        subsystems_dict = {}
        for name, subsystem in config.subsystems.items():
            subsystems_dict[name] = {
                "enabled": subsystem.enabled,
                "weight": subsystem.weight,
                **subsystem.parameters
            }
        
        return {
            "system": config.system,
            "agent": config.agent,
            "subsystems": subsystems_dict,
            "neural": config.neural,
            "risk": config.risk,
            "data": config.data
        }
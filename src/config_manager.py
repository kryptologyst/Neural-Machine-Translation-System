"""
Configuration Management for Neural Machine Translation System

This module provides centralized configuration management using YAML files
with support for environment variables and validation.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration settings."""
    default_language_pair: str = "en-fr"
    max_length: int = 512
    num_beams: int = 4
    early_stopping: bool = True
    temperature: float = 1.0
    do_sample: bool = False
    device: str = "auto"
    batch_size: int = 1
    use_cache: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration settings."""
    metrics: List[str] = field(default_factory=lambda: ["bleu", "chrf", "rouge_l", "bert_score"])
    bert_score_lang: str = "en"
    save_predictions: bool = True
    output_dir: str = "models/evaluations"
    compute_confidence: bool = False


@dataclass
class DatasetConfig:
    """Dataset configuration settings."""
    output_dir: str = "data"
    domains: List[str] = field(default_factory=lambda: ["greetings", "business", "technology", "travel"])
    samples_per_domain: int = 50
    test_size: float = 0.2
    validation_size: float = 0.1
    max_samples: int = 10000


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/translation.log"
    max_size: str = "10MB"
    backup_count: int = 5
    console_output: bool = True


@dataclass
class WebAppConfig:
    """Web application configuration settings."""
    host: str = "localhost"
    port: int = 8501
    debug: bool = False
    theme: str = "light"
    page_title: str = "Neural Machine Translation"
    page_icon: str = "🌐"
    max_file_size: int = 10 * 1024 * 1024  # 10MB


@dataclass
class VisualizationConfig:
    """Visualization configuration settings."""
    output_dir: str = "models/visualizations"
    style: str = "seaborn-v0_8"
    palette: str = "husl"
    figure_size: List[int] = field(default_factory=lambda: [12, 8])
    dpi: int = 300
    save_format: str = "png"


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    web_app: WebAppConfig = field(default_factory=WebAppConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    language_pairs: Dict[str, str] = field(default_factory=dict)


class ConfigManager:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or "config/config.yaml"
        self._config: Optional[AppConfig] = None
    
    @lru_cache(maxsize=1)
    def load_config(self) -> AppConfig:
        """
        Load configuration from file with environment variable overrides.
        
        Returns:
            Loaded configuration object
        """
        try:
            config_data = self._load_yaml_config()
            config_data = self._apply_env_overrides(config_data)
            
            # Create configuration objects
            self._config = AppConfig(
                model=ModelConfig(**config_data.get("model", {})),
                evaluation=EvaluationConfig(**config_data.get("evaluation", {})),
                dataset=DatasetConfig(**config_data.get("dataset", {})),
                logging=LoggingConfig(**config_data.get("logging", {})),
                web_app=WebAppConfig(**config_data.get("web_app", {})),
                visualization=VisualizationConfig(**config_data.get("visualization", {})),
                language_pairs=config_data.get("language_pairs", {})
            )
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Return default configuration
            return AppConfig()
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            logger.warning(f"Configuration file {self.config_path} not found, using defaults")
            return {}
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            "NMT_MODEL_DEVICE": ("model", "device"),
            "NMT_MODEL_MAX_LENGTH": ("model", "max_length"),
            "NMT_MODEL_NUM_BEAMS": ("model", "num_beams"),
            "NMT_LOG_LEVEL": ("logging", "level"),
            "NMT_WEB_PORT": ("web_app", "port"),
            "NMT_WEB_HOST": ("web_app", "host"),
            "NMT_DATA_DIR": ("dataset", "output_dir"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if key in ["max_length", "num_beams", "port"]:
                    env_value = int(env_value)
                elif key in ["test_size", "validation_size", "temperature"]:
                    env_value = float(env_value)
                elif key in ["early_stopping", "do_sample", "debug", "save_predictions"]:
                    env_value = env_value.lower() in ("true", "1", "yes", "on")
                
                if section not in config_data:
                    config_data[section] = {}
                config_data[section][key] = env_value
                logger.info(f"Applied environment override: {env_var}={env_value}")
        
        return config_data
    
    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self) -> AppConfig:
        """Reload configuration from file."""
        self.load_config.cache_clear()
        return self.load_config()
    
    def validate_config(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        config = self.get_config()
        
        # Validate model configuration
        if config.model.max_length <= 0:
            errors.append("Model max_length must be positive")
        
        if config.model.num_beams <= 0:
            errors.append("Model num_beams must be positive")
        
        if config.model.temperature <= 0:
            errors.append("Model temperature must be positive")
        
        # Validate dataset configuration
        if config.dataset.test_size < 0 or config.dataset.test_size > 1:
            errors.append("Dataset test_size must be between 0 and 1")
        
        if config.dataset.validation_size < 0 or config.dataset.validation_size > 1:
            errors.append("Dataset validation_size must be between 0 and 1")
        
        if config.dataset.test_size + config.dataset.validation_size > 1:
            errors.append("Dataset test_size + validation_size must not exceed 1")
        
        # Validate web app configuration
        if config.web_app.port <= 0 or config.web_app.port > 65535:
            errors.append("Web app port must be between 1 and 65535")
        
        return errors


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> AppConfig:
    """Get the current application configuration."""
    return get_config_manager().get_config()


def reload_config() -> AppConfig:
    """Reload the application configuration."""
    return get_config_manager().reload_config()


def validate_config() -> List[str]:
    """Validate the current configuration."""
    return get_config_manager().validate_config()


# Convenience functions for common configuration access
def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return get_config().model


def get_evaluation_config() -> EvaluationConfig:
    """Get evaluation configuration."""
    return get_config().evaluation


def get_dataset_config() -> DatasetConfig:
    """Get dataset configuration."""
    return get_config().dataset


def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return get_config().logging


def get_web_app_config() -> WebAppConfig:
    """Get web application configuration."""
    return get_config().web_app


def get_visualization_config() -> VisualizationConfig:
    """Get visualization configuration."""
    return get_config().visualization


def get_language_pairs() -> Dict[str, str]:
    """Get supported language pairs."""
    return get_config().language_pairs


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Model device: {config.model.device}")
    print(f"Supported language pairs: {len(config.language_pairs)}")
    
    # Validate configuration
    errors = validate_config()
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")

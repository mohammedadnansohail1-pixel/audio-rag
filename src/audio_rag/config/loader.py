"""Configuration loader with YAML merging and environment overrides."""

import os
from pathlib import Path
from typing import Any

import yaml

from audio_rag.config.schema import AudioRAGConfig
from audio_rag.core.exceptions import ConfigError
from audio_rag.utils.logging import get_logger

logger = get_logger(__name__)


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, override takes precedence.
    
    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Parsed YAML as dictionary
        
    Raises:
        ConfigError: If file not found or invalid YAML
    """
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")


def apply_env_overrides(config: dict[str, Any], prefix: str = "AUDIO_RAG") -> dict[str, Any]:
    """Apply environment variable overrides to config.
    
    Environment variables follow pattern: {PREFIX}__{SECTION}__{KEY}
    Example: AUDIO_RAG__ASR__MODEL_SIZE=medium
    
    Args:
        config: Configuration dictionary
        prefix: Environment variable prefix
        
    Returns:
        Config with environment overrides applied
    """
    result = config.copy()
    
    for key, value in os.environ.items():
        if not key.startswith(f"{prefix}__"):
            continue
        
        # Parse key: AUDIO_RAG__ASR__MODEL_SIZE -> ['asr', 'model_size']
        parts = key[len(prefix) + 2:].lower().split("__")
        
        # Navigate to nested location
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        
        # Set value (attempt type conversion)
        final_key = parts[-1]
        target[final_key] = _convert_value(value)
        logger.debug(f"Env override: {'.'.join(parts)} = {value}")
    
    return result


def _convert_value(value: str) -> Any:
    """Convert string value to appropriate type."""
    # Boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    
    # None
    if value.lower() in ("null", "none"):
        return None
    
    # Number
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    return value


def load_config(
    config_path: Path | str | None = None,
    env: str | None = None,
    config_dir: Path | str = "configs",
) -> AudioRAGConfig:
    """Load configuration from YAML files with environment overrides.
    
    Loading order (each overrides previous):
    1. Default values from schema
    2. base.yaml (if exists)
    3. {env}.yaml (if env specified and exists)
    4. config_path (if specified)
    5. Environment variables
    
    Args:
        config_path: Optional specific config file to load
        env: Environment name (development, production, etc.)
        config_dir: Directory containing config files
        
    Returns:
        Validated AudioRAGConfig instance
        
    Raises:
        ConfigError: If configuration is invalid
    """
    config_dir = Path(config_dir)
    config: dict[str, Any] = {}
    
    # Load base.yaml if exists
    base_path = config_dir / "base.yaml"
    if base_path.exists():
        logger.debug(f"Loading base config: {base_path}")
        config = deep_merge(config, load_yaml(base_path))
    
    # Load environment-specific config if specified
    if env:
        env_path = config_dir / f"{env}.yaml"
        if env_path.exists():
            logger.debug(f"Loading {env} config: {env_path}")
            config = deep_merge(config, load_yaml(env_path))
    
    # Load specific config file if provided
    if config_path:
        config_path = Path(config_path)
        logger.debug(f"Loading config: {config_path}")
        config = deep_merge(config, load_yaml(config_path))
    
    # Apply environment variable overrides
    config = apply_env_overrides(config)
    
    # Validate and return
    try:
        return AudioRAGConfig(**config)
    except Exception as e:
        raise ConfigError(f"Configuration validation failed: {e}")

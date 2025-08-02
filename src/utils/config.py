"""Configuration management utilities."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging


class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # Resolve relative paths
        self._resolve_paths(config)
        
        return config
    
    def _resolve_paths(self, config: Dict[str, Any]) -> None:
        """Resolve relative paths in configuration."""
        if 'paths' in config:
            base_dir = self.config_path.parent.parent
            for key, path in config['paths'].items():
                if isinstance(path, str) and not os.path.isabs(path):
                    config['paths'][key] = str(base_dir / path)
    
    def _validate_config(self) -> None:
        """Validate configuration structure and values."""
        required_sections = ['model', 'training', 'data', 'classes', 'paths']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate model configuration
        model_config = self.config['model']
        if 'num_classes' not in model_config:
            raise ValueError("Model configuration must specify num_classes")
            
        # Validate class configuration
        classes_config = self.config['classes']
        if len(classes_config['class_names']) != model_config['num_classes']:
            raise ValueError("Number of class names must match num_classes")
            
        if len(classes_config['class_colors']) != model_config['num_classes']:
            raise ValueError("Number of class colors must match num_classes")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.name')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Output path (default: original config path)
        """
        output_path = Path(path) if path else self.config_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, indent=2)
    
    def create_directories(self) -> None:
        """Create all directories specified in paths configuration."""
        paths_config = self.config.get('paths', {})
        
        for key, path in paths_config.items():
            if isinstance(path, str):
                Path(path).mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory: {path}")


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)


def setup_logging(config: Config) -> None:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration object
    """
    log_level = getattr(logging, config.get('logging.log_level', 'INFO').upper())
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup basic logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)
    
    # Add file handler if enabled
    if config.get('logging.log_to_file', False):
        log_file = config.get('logging.log_file', 'logs/training.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
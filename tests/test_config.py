"""Test configuration utilities."""

import pytest
import tempfile
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import Config, load_config


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config_data = {
        'model': {
            'name': 'unet',
            'backbone': 'resnet34',
            'num_classes': 8,
            'input_channels': 3,
            'pretrained': True
        },
        'training': {
            'batch_size': 4,
            'epochs': 2,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        },
        'data': {
            'image_size': [256, 256],
            'train_split': 0.7,
            'val_split': 0.2,
            'test_split': 0.1,
            'num_workers': 2
        },
        'classes': {
            'num_classes': 8,
            'class_names': [
                'Bareland', 'Rangeland', 'Developed space', 'Road',
                'Tree', 'Water', 'Agriculture land', 'Building'
            ],
            'class_colors': [
                '#800000', '#00FF24', '#949494', '#FFFFFF',
                '#226126', '#0045FF', '#4BB549', '#DE1F07'
            ]
        },
        'paths': {
            'data_dir': 'test_data',
            'models_dir': 'test_models',
            'logs_dir': 'test_logs'
        },
        'hardware': {
            'device': 'cpu'
        },
        'seed': 42
    }
    
    return config_data


@pytest.fixture
def temp_config_file(sample_config):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestConfig:
    """Test configuration management."""
    
    def test_load_config(self, temp_config_file):
        """Test loading configuration from file."""
        config = load_config(temp_config_file)
        
        assert isinstance(config, Config)
        assert config.get('model.name') == 'unet'
        assert config.get('model.num_classes') == 8
        assert config.get('training.batch_size') == 4
    
    def test_config_get_with_dot_notation(self, temp_config_file):
        """Test getting values with dot notation."""
        config = load_config(temp_config_file)
        
        assert config.get('model.name') == 'unet'
        assert config.get('training.learning_rate') == 0.001
        assert config.get('nonexistent.key', 'default') == 'default'
    
    def test_config_set(self, temp_config_file):
        """Test setting configuration values."""
        config = load_config(temp_config_file)
        
        config.set('model.name', 'deeplabv3')
        assert config.get('model.name') == 'deeplabv3'
        
        config.set('new.nested.key', 'value')
        assert config.get('new.nested.key') == 'value'
    
    def test_config_validation(self, sample_config):
        """Test configuration validation."""
        # Valid config should work
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            temp_path = f.name
        
        config = load_config(temp_path)
        assert config.get('model.num_classes') == 8
        
        Path(temp_path).unlink()
        
        # Invalid config should fail
        invalid_config = sample_config.copy()
        del invalid_config['model']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name
        
        with pytest.raises(ValueError):
            load_config(temp_path)
        
        Path(temp_path).unlink()
    
    def test_missing_config_file(self):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_config.yaml')
"""Model factory for creating different architectures."""

import torch.nn as nn
from typing import Dict, Any
from .unet import UNet
from .deeplabv3 import DeepLabV3, DeepLabV3Plus


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        PyTorch model
    """
    model_name = config.get('name', '').lower()
    num_classes = config['num_classes']
    input_channels = config.get('input_channels', 3)
    backbone = config.get('backbone', 'resnet34')
    pretrained = config.get('pretrained', True)
    
    if model_name == 'unet':
        model = UNet(
            num_classes=num_classes,
            input_channels=input_channels,
            backbone=backbone,
            pretrained=pretrained,
            bilinear=config.get('bilinear', False)
        )
    elif model_name == 'deeplabv3':
        model = DeepLabV3(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            dilated=config.get('dilated', True),
            aux_loss=config.get('aux_loss', False)
        )
    elif model_name == 'deeplabv3plus':
        model = DeepLabV3Plus(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get model information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': model.__class__.__name__
    }


def print_model_summary(model: nn.Module) -> None:
    """
    Print model summary.
    
    Args:
        model: PyTorch model
    """
    info = get_model_info(model)
    
    print("=== Model Summary ===")
    print(f"Architecture: {info['architecture']}")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")
    print("====================")


# Registry of available models
MODEL_REGISTRY = {
    'unet': UNet,
    'deeplabv3': DeepLabV3,
    'deeplabv3plus': DeepLabV3Plus
}


def list_available_models():
    """List all available model architectures."""
    return list(MODEL_REGISTRY.keys())
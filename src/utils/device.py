"""Device management utilities."""

import torch
import logging
from typing import Union, Optional


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Desired device ('auto', 'cpu', 'cuda', 'mps', or torch.device)
               If 'auto', automatically selects the best available device
               
    Returns:
        torch.device object
    """
    if device is None or device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logging.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device('cpu')
            logging.info("Using CPU device")
    elif isinstance(device, str):
        device = torch.device(device)
        logging.info(f"Using specified device: {device}")
    
    return device


def set_device(device: Union[str, torch.device]) -> torch.device:
    """
    Set the default device for PyTorch operations.
    
    Args:
        device: Device to set as default
        
    Returns:
        torch.device object
    """
    device = get_device(device)
    
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        
    return device


def get_device_info() -> dict:
    """
    Get detailed information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cpu_count': torch.get_num_threads(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_devices': [],
        'mps_available': False
    }
    
    if torch.cuda.is_available():
        info['cuda_device_count'] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            info['cuda_devices'].append({
                'index': i,
                'name': device_props.name,
                'total_memory': device_props.total_memory,
                'major': device_props.major,
                'minor': device_props.minor
            })
    
    if hasattr(torch.backends, 'mps'):
        info['mps_available'] = torch.backends.mps.is_available()
    
    return info


def print_device_info() -> None:
    """Print detailed device information."""
    info = get_device_info()
    
    print("=== Device Information ===")
    print(f"CPU threads: {info['cpu_count']}")
    print(f"CUDA available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA devices: {info['cuda_device_count']}")
        for device in info['cuda_devices']:
            memory_gb = device['total_memory'] / (1024**3)
            print(f"  Device {device['index']}: {device['name']} "
                  f"({memory_gb:.1f} GB, Compute {device['major']}.{device['minor']})")
    
    print(f"MPS available: {info['mps_available']}")
    print("==========================")
"""Reproducibility utilities."""

import random
import numpy as np
import torch
import logging
from typing import Optional


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    logging.info(f"Set random seed to {seed}")


def make_deterministic(deterministic: bool = True) -> None:
    """
    Configure PyTorch for deterministic behavior.
    
    Args:
        deterministic: Whether to enable deterministic mode
    """
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        logging.info("Enabled deterministic mode")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        logging.info("Disabled deterministic mode")


def setup_reproducibility(seed: Optional[int] = None, deterministic: bool = True) -> None:
    """
    Setup reproducibility configuration.
    
    Args:
        seed: Random seed (if None, uses default value 42)
        deterministic: Whether to enable deterministic mode
    """
    if seed is None:
        seed = 42
    
    set_seed(seed)
    make_deterministic(deterministic)
    
    logging.info(f"Reproducibility setup complete (seed={seed}, deterministic={deterministic})")
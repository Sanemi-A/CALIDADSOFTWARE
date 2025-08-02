"""Optimizers and learning rate schedulers."""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, CyclicLR, OneCycleLR
)
from typing import Dict, Any, Union
import math


def create_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Optimizer configuration
        
    Returns:
        Optimizer instance
    """
    optimizer_type = config.get('optimizer', 'adam').lower()
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    
    if optimizer_type == 'adam':
        betas = config.get('betas', (0.9, 0.999))
        eps = config.get('eps', 1e-8)
        return optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'adamw':
        betas = config.get('betas', (0.9, 0.999))
        eps = config.get('eps', 1e-8)
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        nesterov = config.get('nesterov', True)
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    
    elif optimizer_type == 'rmsprop':
        alpha = config.get('alpha', 0.99)
        eps = config.get('eps', 1e-8)
        momentum = config.get('momentum', 0)
        return optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum
        )
    
    elif optimizer_type == 'adagrad':
        eps = config.get('eps', 1e-10)
        return optim.Adagrad(
            model.parameters(),
            lr=lr,
            eps=eps,
            weight_decay=weight_decay
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any]
) -> Union[torch.optim.lr_scheduler._LRScheduler, None]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Scheduler configuration
        
    Returns:
        Scheduler instance or None
    """
    if 'lr_scheduler' not in config:
        return None
    
    scheduler_config = config['lr_scheduler']
    scheduler_type = scheduler_config.get('type', 'none').lower()
    
    if scheduler_type == 'none' or scheduler_type is None:
        return None
    
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'multistep':
        milestones = scheduler_config.get('milestones', [50, 80])
        gamma = scheduler_config.get('gamma', 0.1)
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_type == 'exponential':
        gamma = scheduler_config.get('gamma', 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == 'cosine':
        max_epochs = scheduler_config.get('max_epochs', 100)
        min_lr = scheduler_config.get('min_lr', 0)
        return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)
    
    elif scheduler_type == 'plateau':
        mode = scheduler_config.get('mode', 'min')
        factor = scheduler_config.get('factor', 0.1)
        patience = scheduler_config.get('patience', 10)
        threshold = scheduler_config.get('threshold', 1e-4)
        min_lr = scheduler_config.get('min_lr', 0)
        return ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr
        )
    
    elif scheduler_type == 'cyclic':
        base_lr = scheduler_config.get('base_lr', 1e-7)
        max_lr = scheduler_config.get('max_lr', 1e-3)
        step_size_up = scheduler_config.get('step_size_up', 2000)
        mode = scheduler_config.get('mode', 'triangular')
        return CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode=mode
        )
    
    elif scheduler_type == 'onecycle':
        max_lr = scheduler_config.get('max_lr', 1e-3)
        total_steps = scheduler_config.get('total_steps', 1000)
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps
        )
    
    elif scheduler_type == 'polynomial':
        # Custom polynomial scheduler
        max_epochs = scheduler_config.get('max_epochs', 100)
        power = scheduler_config.get('power', 0.9)
        min_lr = scheduler_config.get('min_lr', 0)
        return PolynomialLR(optimizer, max_epochs, power, min_lr)
    
    elif scheduler_type == 'warmup_cosine':
        # Custom warmup + cosine scheduler
        warmup_epochs = scheduler_config.get('warmup_epochs', 5)
        max_epochs = scheduler_config.get('max_epochs', 100)
        min_lr = scheduler_config.get('min_lr', 0)
        return WarmupCosineAnnealingLR(optimizer, warmup_epochs, max_epochs, min_lr)
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial learning rate decay scheduler."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        power: float = 0.9,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        super(PolynomialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.max_epochs:
            coeff = (1 - self.last_epoch / self.max_epochs) ** self.power
            return [
                (base_lr - self.min_lr) * coeff + self.min_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [self.min_lr for _ in self.base_lrs]


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup + Cosine Annealing learning rate scheduler."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup scheduler."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        target_lr: float,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                self.target_lr * (self.last_epoch + 1) / self.warmup_epochs
                for _ in self.base_lrs
            ]
        else:
            return [self.target_lr for _ in self.base_lrs]


def get_lr_scheduler_info(scheduler) -> Dict[str, Any]:
    """
    Get information about learning rate scheduler.
    
    Args:
        scheduler: Learning rate scheduler
        
    Returns:
        Dictionary with scheduler information
    """
    if scheduler is None:
        return {'type': 'none'}
    
    info = {
        'type': scheduler.__class__.__name__,
        'current_lr': scheduler.get_last_lr()
    }
    
    # Add scheduler-specific information
    if hasattr(scheduler, 'step_size'):
        info['step_size'] = scheduler.step_size
    if hasattr(scheduler, 'gamma'):
        info['gamma'] = scheduler.gamma
    if hasattr(scheduler, 'T_max'):
        info['T_max'] = scheduler.T_max
    if hasattr(scheduler, 'patience'):
        info['patience'] = scheduler.patience
    
    return info
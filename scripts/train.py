#!/usr/bin/env python3
"""
Training script for land cover segmentation models.

Usage:
    python scripts/train.py --config config/unet_config.yaml
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm

from src.utils import Config, load_config, setup_logging, get_device, setup_reproducibility
from src.models import create_model
from src.data import create_dataloaders
from src.training import create_loss_function, create_optimizer, create_scheduler
from src.evaluation import SegmentationMetrics


class Trainer:
    """Main trainer class for semantic segmentation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = get_device(config.get('hardware.device'))
        
        # Setup reproducibility
        setup_reproducibility(
            seed=config.get('seed'),
            deterministic=config.get('deterministic', True)
        )
        
        # Create directories
        config.create_directories()
        
        # Initialize components
        self.model = self._setup_model()
        self.train_loader, self.val_loader, self.test_loader = self._setup_data()
        self.criterion = self._setup_loss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.metrics = SegmentationMetrics(
            config.get('classes.num_classes'),
            config.get('classes.class_names')
        )
        
        # Training state
        self.current_epoch = 0
        self.best_miou = 0.0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_miou': [], 'val_miou': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }
        
        # Early stopping
        self.early_stopping_patience = config.get('training.early_stopping.patience', 15)
        self.early_stopping_counter = 0
        self.best_loss = float('inf')
        
        logging.info(f"Trainer initialized with device: {self.device}")
    
    def _setup_model(self) -> nn.Module:
        """Setup model."""
        model = create_model(self.config.get('model'))
        model.to(self.device)
        
        # Enable mixed precision if requested
        if self.config.get('hardware.mixed_precision', False):
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        logging.info(f"Model created: {model.__class__.__name__}")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _setup_data(self) -> tuple:
        """Setup data loaders."""
        train_loader, val_loader, test_loader = create_dataloaders(
            self.config.config, self.config.get('data')
        )
        
        logging.info(f"Data loaders created:")
        logging.info(f"  Train: {len(train_loader.dataset)} samples")
        logging.info(f"  Val: {len(val_loader.dataset)} samples")
        if test_loader:
            logging.info(f"  Test: {len(test_loader.dataset)} samples")
        
        return train_loader, val_loader, test_loader
    
    def _setup_loss(self) -> nn.Module:
        """Setup loss function."""
        # Calculate class weights if needed
        class_weights = None
        if hasattr(self.train_loader.dataset, 'calculate_class_weights'):
            class_weights = self.train_loader.dataset.calculate_class_weights()
            class_weights = class_weights.to(self.device)
            logging.info(f"Using class weights: {class_weights}")
        
        loss_config = self.config.get('training.loss', {'type': 'cross_entropy'})
        criterion = create_loss_function(loss_config, class_weights)
        
        logging.info(f"Loss function: {criterion.__class__.__name__}")
        return criterion
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        optimizer = create_optimizer(self.model, self.config.get('training'))
        logging.info(f"Optimizer: {optimizer.__class__.__name__}")
        return optimizer
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        scheduler = create_scheduler(self.optimizer, self.config.get('training'))
        if scheduler:
            logging.info(f"Scheduler: {scheduler.__class__.__name__}")
        return scheduler
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move to device
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            predictions = torch.argmax(outputs, dim=1)
            self.metrics.update(predictions, targets)
            
            # Update running loss
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_metrics = self.metrics.compute_all_metrics()
        
        return {
            'loss': epoch_loss,
            'miou': epoch_metrics['mIoU'],
            'pixel_acc': epoch_metrics['PixelAcc']
        }
    
    def validate_epoch(self) -> dict:
        """Validate for one epoch."""
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                # Move to device
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # Update metrics
                predictions = torch.argmax(outputs, dim=1)
                self.metrics.update(predictions, targets)
                
                # Update running loss
                total_loss += loss.item()
        
        # Calculate epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_metrics = self.metrics.compute_all_metrics()
        
        return {
            'loss': epoch_loss,
            'miou': epoch_metrics['mIoU'],
            'pixel_acc': epoch_metrics['PixelAcc']
        }
    
    def train(self) -> dict:
        """Main training loop."""
        num_epochs = self.config.get('training.epochs', 100)
        
        logging.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_miou'].append(train_metrics['miou'])
            self.history['val_miou'].append(val_metrics['miou'])
            self.history['train_acc'].append(train_metrics['pixel_acc'])
            self.history['val_acc'].append(val_metrics['pixel_acc'])
            self.history['lr'].append(current_lr)
            
            # Log metrics
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Train mIoU: {train_metrics['miou']:.4f}, "
                f"Val mIoU: {val_metrics['miou']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Save best model
            if val_metrics['miou'] > self.best_miou:
                self.best_miou = val_metrics['miou']
                self.save_checkpoint('best_model.pth')
                logging.info(f"New best mIoU: {self.best_miou:.4f}")
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                logging.info(f"Early stopping after {epoch+1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        logging.info("Training completed!")
        return self.history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        models_dir = Path(self.config.get('paths.models_dir'))
        models_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
            'history': self.history,
            'config': self.config.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, models_dir / filename)
        logging.info(f"Checkpoint saved: {models_dir / filename}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train semantic segmentation model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logging.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_miou = checkpoint.get('best_miou', 0.0)
        trainer.history = checkpoint.get('history', {})
    
    # Train model
    history = trainer.train()
    
    # Save final model
    trainer.save_checkpoint('final_model.pth')
    
    logging.info("Training script completed successfully!")


if __name__ == "__main__":
    main()
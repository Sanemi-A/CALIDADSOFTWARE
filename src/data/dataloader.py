"""DataLoader utilities."""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, Any, Optional
import logging
from .dataset import LandCoverDataset
from .transforms import get_transforms


def create_dataloaders(
    config: Dict[str, Any],
    data_config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: General configuration
        data_config: Data-specific configuration
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get paths
    paths = config.get('paths', {})
    
    # Check if separate train/val/test directories exist
    train_images = paths.get('train_images')
    train_masks = paths.get('train_masks')
    val_images = paths.get('val_images')
    val_masks = paths.get('val_masks')
    test_images = paths.get('test_images')
    test_masks = paths.get('test_masks')
    
    if all([train_images, train_masks, val_images, val_masks]):
        # Use separate directories
        train_loader = _create_single_dataloader(
            train_images, train_masks, 'train', config, data_config
        )
        val_loader = _create_single_dataloader(
            val_images, val_masks, 'val', config, data_config
        )
        
        if all([test_images, test_masks]):
            test_loader = _create_single_dataloader(
                test_images, test_masks, 'test', config, data_config
            )
        else:
            test_loader = None
            logging.info("No test data directories specified")
    
    else:
        # Use single directory with splits
        images_dir = paths.get('data_dir', 'data') + '/images'
        masks_dir = paths.get('data_dir', 'data') + '/masks'
        
        train_loader, val_loader, test_loader = _create_split_dataloaders(
            images_dir, masks_dir, config, data_config
        )
    
    return train_loader, val_loader, test_loader


def _create_single_dataloader(
    images_dir: str,
    masks_dir: str,
    mode: str,
    config: Dict[str, Any],
    data_config: Dict[str, Any]
) -> DataLoader:
    """Create a single dataloader for given directories."""
    # Get transforms
    transforms = get_transforms(mode, data_config)
    
    # Create dataset
    dataset = LandCoverDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=transforms
    )
    
    # DataLoader parameters
    batch_size = data_config.get('batch_size', 16)
    if mode != 'train':
        batch_size = min(batch_size, 8)  # Smaller batch size for val/test
    
    num_workers = data_config.get('num_workers', 4)
    pin_memory = data_config.get('pin_memory', True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(mode == 'train')
    )
    
    logging.info(f"Created {mode} dataloader: {len(dataset)} samples, "
                f"batch_size={batch_size}")
    
    return dataloader


def _create_split_dataloaders(
    images_dir: str,
    masks_dir: str,
    config: Dict[str, Any],
    data_config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create dataloaders with dataset splitting."""
    # Create full dataset with validation transforms
    transforms = get_transforms('val', data_config)
    full_dataset = LandCoverDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=transforms
    )
    
    # Get split ratios
    train_split = data_config.get('train_split', 0.7)
    val_split = data_config.get('val_split', 0.2)
    test_split = data_config.get('test_split', 0.1)
    
    # Validate splits
    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_split}")
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )
    
    # Update transforms for training dataset
    train_transforms = get_transforms('train', data_config)
    _update_dataset_transforms(train_dataset, train_transforms)
    
    # Create dataloaders
    batch_size = data_config.get('batch_size', 16)
    num_workers = data_config.get('num_workers', 4)
    pin_memory = data_config.get('pin_memory', True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_batch_size = min(batch_size, 8)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = None
    if test_size > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    logging.info(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return train_loader, val_loader, test_loader


def _update_dataset_transforms(subset_dataset, new_transforms):
    """Update transforms for a subset dataset."""
    # Access the underlying dataset and update transforms
    if hasattr(subset_dataset, 'dataset'):
        subset_dataset.dataset.transform = new_transforms


def calculate_dataset_statistics(dataloader: DataLoader) -> Dict[str, torch.Tensor]:
    """
    Calculate mean and std statistics for the dataset.
    
    Args:
        dataloader: DataLoader to analyze
        
    Returns:
        Dictionary with mean and std statistics
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    logging.info("Calculating dataset statistics...")
    
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return {'mean': mean, 'std': std}


def get_class_weights(dataloader: DataLoader, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights from dataloader.
    
    Args:
        dataloader: DataLoader to analyze
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    class_counts = torch.zeros(num_classes)
    
    logging.info("Calculating class weights...")
    
    for _, masks in dataloader:
        for class_id in range(num_classes):
            class_counts[class_id] += (masks == class_id).sum().item()
    
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts)
    
    # Handle zero counts
    class_weights[class_counts == 0] = 1.0
    
    return class_weights


def create_weighted_sampler(dataset: LandCoverDataset) -> torch.utils.data.WeightedRandomSampler:
    """
    Create weighted random sampler for imbalanced classes.
    
    Args:
        dataset: Dataset to create sampler for
        
    Returns:
        WeightedRandomSampler
    """
    # Calculate class distribution
    distribution = dataset.get_class_distribution()
    
    # Calculate sample weights
    sample_weights = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        # Get most common class in this sample
        unique, counts = torch.unique(mask, return_counts=True)
        dominant_class = unique[torch.argmax(counts)].item()
        
        # Weight inversely proportional to class frequency
        class_freq = distribution[dominant_class]['percentage'] / 100.0
        weight = 1.0 / (class_freq + 1e-6)
        sample_weights.append(weight)
    
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
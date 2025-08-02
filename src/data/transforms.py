"""Data transformations and augmentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Any, Optional, Tuple


def get_train_transforms(config: Dict[str, Any]) -> A.Compose:
    """
    Get training data transforms with augmentations.
    
    Args:
        config: Configuration dictionary containing augmentation parameters
        
    Returns:
        Albumentations Compose object
    """
    image_size = config.get('image_size', [512, 512])
    aug_config = config.get('augmentation', {})
    
    transforms = [
        # Resize to target size
        A.Resize(height=image_size[0], width=image_size[1], p=1.0),
    ]
    
    # Geometric augmentations
    if aug_config.get('horizontal_flip', 0) > 0:
        transforms.append(A.HorizontalFlip(p=aug_config['horizontal_flip']))
    
    if aug_config.get('vertical_flip', 0) > 0:
        transforms.append(A.VerticalFlip(p=aug_config['vertical_flip']))
    
    if aug_config.get('rotation', 0) > 0:
        transforms.append(A.Rotate(
            limit=aug_config['rotation'], 
            p=0.7,
            border_mode=0,  # Constant
            value=0
        ))
    
    # Scale and crop augmentations
    if 'scale' in aug_config and 'crop_size' in aug_config:
        scale_range = aug_config['scale']
        crop_size = aug_config['crop_size']
        transforms.append(A.RandomResizedCrop(
            height=crop_size[0],
            width=crop_size[1],
            scale=scale_range,
            ratio=(0.8, 1.2),
            p=0.5
        ))
    
    # Color augmentations
    color_transforms = []
    
    if aug_config.get('brightness', 0) > 0:
        brightness_limit = aug_config['brightness']
        color_transforms.append(A.RandomBrightness(limit=brightness_limit, p=0.5))
    
    if aug_config.get('contrast', 0) > 0:
        contrast_limit = aug_config['contrast']
        color_transforms.append(A.RandomContrast(limit=contrast_limit, p=0.5))
    
    if aug_config.get('saturation', 0) > 0:
        saturation_limit = aug_config['saturation']
        color_transforms.append(A.HueSaturationValue(
            sat_shift_limit=int(saturation_limit * 255),
            hue_shift_limit=0,
            val_shift_limit=0,
            p=0.5
        ))
    
    if aug_config.get('hue', 0) > 0:
        hue_limit = aug_config['hue']
        color_transforms.append(A.HueSaturationValue(
            hue_shift_limit=int(hue_limit * 255),
            sat_shift_limit=0,
            val_shift_limit=0,
            p=0.5
        ))
    
    if color_transforms:
        transforms.append(A.OneOf(color_transforms, p=0.5))
    
    # Additional augmentations
    transforms.extend([
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.2),
    ])
    
    # Normalization (if needed)
    # transforms.append(A.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    #     max_pixel_value=1.0,
    #     p=1.0
    # ))
    
    return A.Compose(transforms)


def get_val_transforms(config: Dict[str, Any]) -> A.Compose:
    """
    Get validation data transforms (no augmentations).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Albumentations Compose object
    """
    image_size = config.get('image_size', [512, 512])
    
    transforms = [
        A.Resize(height=image_size[0], width=image_size[1], p=1.0),
    ]
    
    # Normalization (if needed)
    # transforms.append(A.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    #     max_pixel_value=1.0,
    #     p=1.0
    # ))
    
    return A.Compose(transforms)


def get_test_transforms(config: Dict[str, Any]) -> A.Compose:
    """
    Get test data transforms.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Albumentations Compose object
    """
    return get_val_transforms(config)


def get_transforms(
    mode: str,
    config: Dict[str, Any]
) -> A.Compose:
    """
    Get appropriate transforms based on mode.
    
    Args:
        mode: Mode ('train', 'val', 'test')
        config: Configuration dictionary
        
    Returns:
        Albumentations Compose object
    """
    if mode == 'train':
        return get_train_transforms(config)
    elif mode == 'val':
        return get_val_transforms(config)
    elif mode == 'test':
        return get_test_transforms(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


class TTA:
    """Test Time Augmentation."""
    
    def __init__(self, transforms: list):
        """
        Initialize TTA with list of transforms.
        
        Args:
            transforms: List of augmentation transforms
        """
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray) -> list:
        """
        Apply test time augmentation.
        
        Args:
            image: Input image
            
        Returns:
            List of augmented images
        """
        augmented_images = [image]  # Original image
        
        for transform in self.transforms:
            augmented = transform(image=image)['image']
            augmented_images.append(augmented)
        
        return augmented_images


def get_tta_transforms() -> TTA:
    """
    Get Test Time Augmentation transforms.
    
    Returns:
        TTA object with appropriate transforms
    """
    transforms = [
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Rotate(limit=90, p=1.0),
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0)
        ])
    ]
    
    return TTA(transforms)


def create_mixup_data(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Create MixUp augmented data.
    
    Args:
        x: Input data
        y: Target data
        alpha: MixUp parameter
        
    Returns:
        Mixed input, target1, target2, lambda value
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def create_cutmix_data(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Create CutMix augmented data.
    
    Args:
        x: Input data
        y: Target data
        alpha: CutMix parameter
        
    Returns:
        Mixed input, target1, target2, lambda value
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    
    _, _, H, W = x.shape
    
    # Generate random bounding box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Adjust lambda to actual cut ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    # Apply cutmix to batch
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    y[:, bby1:bby2, bbx1:bbx2] = y[index, bby1:bby2, bbx1:bbx2]
    
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam
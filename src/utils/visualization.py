"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from typing import List, Optional, Tuple, Union
from pathlib import Path
import cv2


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to RGB.
    
    Args:
        hex_color: Hex color string (e.g., '#FF0000')
        
    Returns:
        RGB tuple (0-255)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_color_map(class_colors: List[str]) -> np.ndarray:
    """
    Create color map for visualization.
    
    Args:
        class_colors: List of hex color strings
        
    Returns:
        Color map array
    """
    color_map = np.zeros((len(class_colors), 3), dtype=np.uint8)
    
    for i, color in enumerate(class_colors):
        color_map[i] = hex_to_rgb(color)
    
    return color_map


def mask_to_rgb(mask: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    """
    Convert segmentation mask to RGB image.
    
    Args:
        mask: Segmentation mask (H, W)
        color_map: Color map array (num_classes, 3)
        
    Returns:
        RGB image (H, W, 3)
    """
    h, w = mask.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(color_map)):
        rgb_image[mask == class_id] = color_map[class_id]
    
    return rgb_image


def visualize_predictions(
    images: Union[torch.Tensor, np.ndarray],
    true_masks: Union[torch.Tensor, np.ndarray],
    pred_masks: Union[torch.Tensor, np.ndarray],
    class_colors: List[str],
    class_names: List[str],
    save_path: Optional[str] = None,
    max_samples: int = 4
) -> None:
    """
    Visualize predictions alongside ground truth.
    
    Args:
        images: Input images
        true_masks: Ground truth masks
        pred_masks: Predicted masks
        class_colors: List of class colors (hex)
        class_names: List of class names
        save_path: Path to save visualization
        max_samples: Maximum number of samples to visualize
    """
    # Convert tensors to numpy if needed
    if torch.is_tensor(images):
        images = images.cpu().numpy()
    if torch.is_tensor(true_masks):
        true_masks = true_masks.cpu().numpy()
    if torch.is_tensor(pred_masks):
        pred_masks = pred_masks.cpu().numpy()
    
    # Create color map
    color_map = create_color_map(class_colors)
    
    # Limit number of samples
    n_samples = min(len(images), max_samples)
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Original image
        img = images[i]
        if img.shape[0] == 3:  # CHW to HWC
            img = np.transpose(img, (1, 2, 0))
        
        # Normalize image for display
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        true_rgb = mask_to_rgb(true_masks[i], color_map)
        axes[i, 1].imshow(true_rgb)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted mask
        pred_rgb = mask_to_rgb(pred_masks[i], color_map)
        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Training Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # mIoU
    if 'train_miou' in history and 'val_miou' in history:
        axes[0, 1].plot(history['train_miou'], label='Training mIoU')
        axes[0, 1].plot(history['val_miou'], label='Validation mIoU')
        axes[0, 1].set_title('Mean IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mIoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Accuracy
    if 'train_acc' in history and 'val_acc' in history:
        axes[1, 0].plot(history['train_acc'], label='Training Accuracy')
        axes[1, 0].plot(history['val_acc'], label='Validation Accuracy')
        axes[1, 0].set_title('Pixel Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning Rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()


def create_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True
) -> None:
    """
    Create and visualize confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save confusion matrix
        normalize: Whether to normalize the matrix
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
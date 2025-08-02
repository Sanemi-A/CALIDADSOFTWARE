"""Loss functions for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class DiceLoss(nn.Module):
    """Dice Loss for semantic segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply softmax to predictions
        pred_soft = F.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=pred.size(1))
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate intersection and union
        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        pred_sum = pred_soft.sum(dim=(2, 3))
        target_sum = target_one_hot.sum(dim=(2, 3))
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class IoULoss(nn.Module):
    """IoU (Jaccard) Loss for semantic segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply softmax to predictions
        pred_soft = F.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=pred.size(1))
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate intersection and union
        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return IoU loss (1 - IoU)
        return 1.0 - iou.mean()


class CombinedLoss(nn.Module):
    """Combined loss function using multiple loss components."""
    
    def __init__(
        self,
        losses: Dict[str, float],
        class_weights: Optional[torch.Tensor] = None
    ):
        super(CombinedLoss, self).__init__()
        self.loss_weights = losses
        self.class_weights = class_weights
        
        # Initialize loss functions
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.iou_loss = IoULoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        if 'cross_entropy' in self.loss_weights:
            ce = self.ce_loss(pred, target)
            total_loss += self.loss_weights['cross_entropy'] * ce
        
        if 'dice' in self.loss_weights:
            dice = self.dice_loss(pred, target)
            total_loss += self.loss_weights['dice'] * dice
        
        if 'focal' in self.loss_weights:
            focal = self.focal_loss(pred, target)
            total_loss += self.loss_weights['focal'] * focal
        
        if 'iou' in self.loss_weights:
            iou = self.iou_loss(pred, target)
            total_loss += self.loss_weights['iou'] * iou
        
        return total_loss


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice Loss."""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply softmax to predictions
        pred_soft = F.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=pred.size(1))
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate True Positives, False Positives, False Negatives
        tp = (pred_soft * target_one_hot).sum(dim=(2, 3))
        fp = (pred_soft * (1 - target_one_hot)).sum(dim=(2, 3))
        fn = ((1 - pred_soft) * target_one_hot).sum(dim=(2, 3))
        
        # Calculate Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Return Tversky loss
        return 1.0 - tversky.mean()


def create_loss_function(
    config: Dict[str, Any],
    class_weights: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        config: Loss configuration
        class_weights: Optional class weights for handling imbalance
        
    Returns:
        Loss function
    """
    loss_type = config.get('type', 'cross_entropy').lower()
    
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'dice':
        smooth = config.get('smooth', 1e-6)
        return DiceLoss(smooth=smooth)
    
    elif loss_type == 'focal':
        alpha = config.get('alpha', 1.0)
        gamma = config.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'iou':
        smooth = config.get('smooth', 1e-6)
        return IoULoss(smooth=smooth)
    
    elif loss_type == 'tversky':
        alpha = config.get('alpha', 0.3)
        beta = config.get('beta', 0.7)
        smooth = config.get('smooth', 1e-6)
        return TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
    
    elif loss_type == 'combined':
        losses = config.get('losses', {
            'cross_entropy': 1.0,
            'dice': 1.0
        })
        return CombinedLoss(losses=losses, class_weights=class_weights)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Default loss configurations
DEFAULT_LOSS_CONFIGS = {
    'cross_entropy': {
        'type': 'cross_entropy'
    },
    'dice': {
        'type': 'dice',
        'smooth': 1e-6
    },
    'focal': {
        'type': 'focal',
        'alpha': 1.0,
        'gamma': 2.0
    },
    'combined_dice_ce': {
        'type': 'combined',
        'losses': {
            'cross_entropy': 1.0,
            'dice': 1.0
        }
    },
    'combined_focal_dice': {
        'type': 'combined',
        'losses': {
            'focal': 1.0,
            'dice': 1.0
        }
    }
}
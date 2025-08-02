"""Evaluation metrics for semantic segmentation."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class SegmentationMetrics:
    """Comprehensive metrics for semantic segmentation evaluation."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_pixels = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update metrics with new predictions and targets.
        
        Args:
            pred: Predicted segmentation mask (B, H, W) or (B, C, H, W)
            target: Ground truth segmentation mask (B, H, W)
        """
        # Handle logits input
        if pred.dim() == 4:  # (B, C, H, W)
            pred = torch.argmax(pred, dim=1)
        
        # Flatten tensors
        pred_flat = pred.flatten().cpu().numpy()
        target_flat = target.flatten().cpu().numpy()
        
        # Update confusion matrix
        mask = (target_flat >= 0) & (target_flat < self.num_classes)
        hist = np.bincount(
            self.num_classes * target_flat[mask] + pred_flat[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist
        self.total_pixels += len(pred_flat)
    
    def compute_iou(self) -> Dict[str, float]:
        """Compute IoU for each class and mean IoU."""
        # Calculate IoU for each class
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0) - 
            intersection
        )
        
        # Avoid division by zero
        iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
        
        # Create results dictionary
        results = {}
        for i, class_name in enumerate(self.class_names):
            results[f"IoU_{class_name}"] = float(iou[i])
        
        results["mIoU"] = float(np.nanmean(iou))
        
        return results
    
    def compute_pixel_accuracy(self) -> float:
        """Compute overall pixel accuracy."""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return float(correct / total) if total > 0 else 0.0
    
    def compute_class_accuracy(self) -> Dict[str, float]:
        """Compute accuracy for each class."""
        class_correct = np.diag(self.confusion_matrix)
        class_total = self.confusion_matrix.sum(axis=1)
        
        results = {}
        for i, class_name in enumerate(self.class_names):
            if class_total[i] > 0:
                acc = float(class_correct[i] / class_total[i])
            else:
                acc = 0.0
            results[f"Acc_{class_name}"] = acc
        
        # Mean class accuracy
        accuracies = [results[f"Acc_{name}"] for name in self.class_names]
        results["mAcc"] = float(np.mean(accuracies))
        
        return results
    
    def compute_precision_recall_f1(self) -> Dict[str, float]:
        """Compute precision, recall, and F1-score for each class."""
        results = {}
        
        for i, class_name in enumerate(self.class_names):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            results[f"Precision_{class_name}"] = float(precision)
            
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            results[f"Recall_{class_name}"] = float(recall)
            
            # F1-score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            results[f"F1_{class_name}"] = float(f1)
        
        # Mean metrics
        precisions = [results[f"Precision_{name}"] for name in self.class_names]
        recalls = [results[f"Recall_{name}"] for name in self.class_names]
        f1_scores = [results[f"F1_{name}"] for name in self.class_names]
        
        results["mPrecision"] = float(np.mean(precisions))
        results["mRecall"] = float(np.mean(recalls))
        results["mF1"] = float(np.mean(f1_scores))
        
        return results
    
    def compute_dice_coefficient(self) -> Dict[str, float]:
        """Compute Dice coefficient for each class."""
        results = {}
        
        for i, class_name in enumerate(self.class_names):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            results[f"Dice_{class_name}"] = float(dice)
        
        # Mean Dice coefficient
        dice_scores = [results[f"Dice_{name}"] for name in self.class_names]
        results["mDice"] = float(np.mean(dice_scores))
        
        return results
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all available metrics."""
        metrics = {}
        
        # IoU metrics
        metrics.update(self.compute_iou())
        
        # Accuracy metrics
        metrics["PixelAcc"] = self.compute_pixel_accuracy()
        metrics.update(self.compute_class_accuracy())
        
        # Precision, Recall, F1
        metrics.update(self.compute_precision_recall_f1())
        
        # Dice coefficient
        metrics.update(self.compute_dice_coefficient())
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get the confusion matrix."""
        return self.confusion_matrix.copy()
    
    def print_metrics(self) -> None:
        """Print formatted metrics."""
        metrics = self.compute_all_metrics()
        
        print("=== Segmentation Metrics ===")
        print(f"Pixel Accuracy: {metrics['PixelAcc']:.4f}")
        print(f"Mean IoU: {metrics['mIoU']:.4f}")
        print(f"Mean Accuracy: {metrics['mAcc']:.4f}")
        print(f"Mean Precision: {metrics['mPrecision']:.4f}")
        print(f"Mean Recall: {metrics['mRecall']:.4f}")
        print(f"Mean F1-Score: {metrics['mF1']:.4f}")
        print(f"Mean Dice: {metrics['mDice']:.4f}")
        
        print("\n=== Per-Class Metrics ===")
        for class_name in self.class_names:
            print(f"{class_name}:")
            print(f"  IoU: {metrics[f'IoU_{class_name}']:.4f}")
            print(f"  Acc: {metrics[f'Acc_{class_name}']:.4f}")
            print(f"  Precision: {metrics[f'Precision_{class_name}']:.4f}")
            print(f"  Recall: {metrics[f'Recall_{class_name}']:.4f}")
            print(f"  F1: {metrics[f'F1_{class_name}']:.4f}")
            print(f"  Dice: {metrics[f'Dice_{class_name}']:.4f}")


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Calculate IoU for each class.
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth segmentation mask
        num_classes: Number of classes
        
    Returns:
        IoU for each class
    """
    ious = []
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return torch.tensor(ious)


def calculate_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """
    Calculate mean IoU.
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth segmentation mask
        num_classes: Number of classes
        
    Returns:
        Mean IoU
    """
    ious = calculate_iou(pred, target, num_classes)
    return ious.mean().item()


def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate pixel accuracy.
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth segmentation mask
        
    Returns:
        Pixel accuracy
    """
    correct = (pred == target).sum().float()
    total = target.numel()
    return (correct / total).item()


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Calculate Dice score for each class.
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth segmentation mask
        num_classes: Number of classes
        
    Returns:
        Dice score for each class
    """
    dice_scores = []
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        total = pred_mask.sum() + target_mask.sum()
        
        if total == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = 2.0 * intersection / total
        
        dice_scores.append(dice)
    
    return torch.tensor(dice_scores)
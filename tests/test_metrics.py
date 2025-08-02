"""Test evaluation metrics."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    SegmentationMetrics, calculate_iou, calculate_miou,
    calculate_pixel_accuracy, calculate_dice_score
)


class TestSegmentationMetrics:
    """Test segmentation metrics calculation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample prediction and target data."""
        # Create simple 4x4 predictions and targets for 3 classes
        pred = torch.tensor([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 2, 2]
        ])
        
        target = torch.tensor([
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 2, 0]
        ])
        
        return pred, target
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = SegmentationMetrics(num_classes=8)
        
        assert metrics.num_classes == 8
        assert len(metrics.class_names) == 8
        assert metrics.confusion_matrix.shape == (8, 8)
    
    def test_metrics_update(self, sample_data):
        """Test metrics update with sample data."""
        pred, target = sample_data
        metrics = SegmentationMetrics(num_classes=3)
        
        # Add batch dimension
        pred_batch = pred.unsqueeze(0)
        target_batch = target.unsqueeze(0)
        
        metrics.update(pred_batch, target_batch)
        
        # Check that confusion matrix was updated
        assert metrics.confusion_matrix.sum() == pred.numel()
    
    def test_iou_calculation(self, sample_data):
        """Test IoU calculation."""
        pred, target = sample_data
        metrics = SegmentationMetrics(num_classes=3, class_names=['A', 'B', 'C'])
        
        metrics.update(pred.unsqueeze(0), target.unsqueeze(0))
        iou_results = metrics.compute_iou()
        
        assert 'mIoU' in iou_results
        assert 'IoU_A' in iou_results
        assert 'IoU_B' in iou_results
        assert 'IoU_C' in iou_results
        
        # Check that IoU values are between 0 and 1
        for key, value in iou_results.items():
            assert 0 <= value <= 1
    
    def test_pixel_accuracy(self, sample_data):
        """Test pixel accuracy calculation."""
        pred, target = sample_data
        metrics = SegmentationMetrics(num_classes=3)
        
        metrics.update(pred.unsqueeze(0), target.unsqueeze(0))
        pixel_acc = metrics.compute_pixel_accuracy()
        
        assert 0 <= pixel_acc <= 1
        
        # Manual calculation for verification
        correct = (pred == target).sum().item()
        total = pred.numel()
        expected_acc = correct / total
        
        assert abs(pixel_acc - expected_acc) < 1e-6
    
    def test_precision_recall_f1(self, sample_data):
        """Test precision, recall, and F1 calculation."""
        pred, target = sample_data
        metrics = SegmentationMetrics(num_classes=3, class_names=['A', 'B', 'C'])
        
        metrics.update(pred.unsqueeze(0), target.unsqueeze(0))
        pr_results = metrics.compute_precision_recall_f1()
        
        # Check that all expected keys are present
        for class_name in ['A', 'B', 'C']:
            assert f'Precision_{class_name}' in pr_results
            assert f'Recall_{class_name}' in pr_results
            assert f'F1_{class_name}' in pr_results
        
        assert 'mPrecision' in pr_results
        assert 'mRecall' in pr_results
        assert 'mF1' in pr_results
        
        # Check that values are between 0 and 1
        for key, value in pr_results.items():
            assert 0 <= value <= 1
    
    def test_dice_coefficient(self, sample_data):
        """Test Dice coefficient calculation."""
        pred, target = sample_data
        metrics = SegmentationMetrics(num_classes=3, class_names=['A', 'B', 'C'])
        
        metrics.update(pred.unsqueeze(0), target.unsqueeze(0))
        dice_results = metrics.compute_dice_coefficient()
        
        assert 'mDice' in dice_results
        for class_name in ['A', 'B', 'C']:
            assert f'Dice_{class_name}' in dice_results
        
        # Check that values are between 0 and 1
        for key, value in dice_results.items():
            assert 0 <= value <= 1
    
    def test_all_metrics(self, sample_data):
        """Test computing all metrics at once."""
        pred, target = sample_data
        metrics = SegmentationMetrics(num_classes=3, class_names=['A', 'B', 'C'])
        
        metrics.update(pred.unsqueeze(0), target.unsqueeze(0))
        all_metrics = metrics.compute_all_metrics()
        
        # Check that all expected metric types are present
        expected_keys = ['mIoU', 'PixelAcc', 'mAcc', 'mPrecision', 'mRecall', 'mF1', 'mDice']
        for key in expected_keys:
            assert key in all_metrics
    
    def test_perfect_prediction(self):
        """Test metrics with perfect prediction."""
        # Create identical prediction and target
        size = (1, 4, 4)
        pred = torch.randint(0, 3, size)
        target = pred.clone()
        
        metrics = SegmentationMetrics(num_classes=3)
        metrics.update(pred, target)
        
        all_metrics = metrics.compute_all_metrics()
        
        # Perfect prediction should give IoU = 1, accuracy = 1, etc.
        assert abs(all_metrics['mIoU'] - 1.0) < 1e-6
        assert abs(all_metrics['PixelAcc'] - 1.0) < 1e-6
        assert abs(all_metrics['mDice'] - 1.0) < 1e-6


class TestUtilityFunctions:
    """Test utility functions for metrics."""
    
    def test_calculate_iou(self):
        """Test IoU calculation function."""
        pred = torch.tensor([[0, 0, 1], [1, 1, 0], [2, 2, 2]])
        target = torch.tensor([[0, 0, 1], [1, 0, 0], [2, 2, 1]])
        
        iou = calculate_iou(pred, target, num_classes=3)
        
        assert iou.shape == (3,)
        assert torch.all(iou >= 0)
        assert torch.all(iou <= 1)
    
    def test_calculate_miou(self):
        """Test mIoU calculation function."""
        pred = torch.tensor([[0, 0, 1], [1, 1, 0], [2, 2, 2]])
        target = torch.tensor([[0, 0, 1], [1, 0, 0], [2, 2, 1]])
        
        miou = calculate_miou(pred, target, num_classes=3)
        
        assert isinstance(miou, float)
        assert 0 <= miou <= 1
    
    def test_calculate_pixel_accuracy(self):
        """Test pixel accuracy calculation function."""
        pred = torch.tensor([[0, 0, 1], [1, 1, 0]])
        target = torch.tensor([[0, 1, 1], [1, 0, 0]])
        
        acc = calculate_pixel_accuracy(pred, target)
        
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
        
        # Manual verification
        correct = (pred == target).sum().item()
        total = pred.numel()
        expected_acc = correct / total
        
        assert abs(acc - expected_acc) < 1e-6
    
    def test_calculate_dice_score(self):
        """Test Dice score calculation function."""
        pred = torch.tensor([[0, 0, 1], [1, 1, 0], [2, 2, 2]])
        target = torch.tensor([[0, 0, 1], [1, 0, 0], [2, 2, 1]])
        
        dice = calculate_dice_score(pred, target, num_classes=3)
        
        assert dice.shape == (3,)
        assert torch.all(dice >= 0)
        assert torch.all(dice <= 1)
    
    @pytest.mark.parametrize("num_classes", [2, 8, 21])
    def test_metrics_with_different_classes(self, num_classes):
        """Test metrics with different number of classes."""
        pred = torch.randint(0, num_classes, (1, 16, 16))
        target = torch.randint(0, num_classes, (1, 16, 16))
        
        # Test all utility functions
        iou = calculate_iou(pred.squeeze(), target.squeeze(), num_classes)
        miou = calculate_miou(pred.squeeze(), target.squeeze(), num_classes)
        acc = calculate_pixel_accuracy(pred.squeeze(), target.squeeze())
        dice = calculate_dice_score(pred.squeeze(), target.squeeze(), num_classes)
        
        assert iou.shape == (num_classes,)
        assert isinstance(miou, float)
        assert isinstance(acc, float)
        assert dice.shape == (num_classes,)
    
    def test_edge_cases(self):
        """Test edge cases like empty classes."""
        # Create prediction with missing classes
        pred = torch.zeros((1, 4, 4), dtype=torch.long)  # Only class 0
        target = torch.tensor([[0, 0, 1, 1],
                              [0, 0, 1, 1],
                              [2, 2, 1, 1],
                              [2, 2, 2, 2]]).unsqueeze(0)
        
        metrics = SegmentationMetrics(num_classes=3)
        metrics.update(pred, target)
        
        # Should handle missing classes gracefully
        all_metrics = metrics.compute_all_metrics()
        
        # Check that metrics are computed without errors
        assert 'mIoU' in all_metrics
        assert not np.isnan(all_metrics['mIoU'])
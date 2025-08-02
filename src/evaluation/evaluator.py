"""Model evaluator for comprehensive evaluation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json

from .metrics import SegmentationMetrics
from ..utils.visualization import visualize_predictions, create_confusion_matrix


class Evaluator:
    """Comprehensive model evaluator for semantic segmentation."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        class_names: List[str],
        class_colors: List[str]
    ):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        self.class_colors = class_colors
        self.metrics = SegmentationMetrics(num_classes, class_names)
        
        # Move model to device
        self.model.to(self.device)
    
    def evaluate(
        self,
        dataloader: DataLoader,
        save_results: bool = True,
        save_dir: Optional[str] = None,
        visualize: bool = True,
        max_vis_samples: int = 8
    ) -> Dict[str, Any]:
        """
        Evaluate model on given dataloader.
        
        Args:
            dataloader: DataLoader for evaluation
            save_results: Whether to save evaluation results
            save_dir: Directory to save results
            visualize: Whether to create visualizations
            max_vis_samples: Maximum samples for visualization
            
        Returns:
            Dictionary with evaluation results
        """
        self.model.eval()
        self.metrics.reset()
        
        all_predictions = []
        all_targets = []
        all_images = []
        inference_times = []
        
        logging.info("Starting model evaluation...")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                # Move to device
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                
                # Handle model outputs (some models return dict)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Get predictions
                predictions = torch.argmax(outputs, dim=1)
                
                # Update metrics
                self.metrics.update(predictions, targets)
                
                # Store for visualization (limited samples)
                if len(all_predictions) < max_vis_samples:
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
                    all_images.append(images.cpu())
                
                if (batch_idx + 1) % 10 == 0:
                    logging.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Compute all metrics
        metrics_dict = self.metrics.compute_all_metrics()
        
        # Add timing metrics
        metrics_dict['avg_inference_time'] = np.mean(inference_times)
        metrics_dict['std_inference_time'] = np.std(inference_times)
        metrics_dict['total_inference_time'] = np.sum(inference_times)
        
        # Performance metrics
        metrics_dict['fps'] = len(dataloader.dataset) / np.sum(inference_times)
        
        # Create evaluation results
        results = {
            'metrics': metrics_dict,
            'confusion_matrix': self.metrics.get_confusion_matrix().tolist(),
            'class_names': self.class_names,
            'num_samples': len(dataloader.dataset),
            'inference_times': inference_times
        }
        
        # Print metrics
        self.metrics.print_metrics()
        print(f"\nTiming Metrics:")
        print(f"Average inference time: {metrics_dict['avg_inference_time']:.4f}s")
        print(f"FPS: {metrics_dict['fps']:.2f}")
        
        # Save results if requested
        if save_results and save_dir:
            self._save_results(results, save_dir)
        
        # Create visualizations if requested
        if visualize and save_dir and len(all_predictions) > 0:
            self._create_visualizations(
                all_images, all_targets, all_predictions, save_dir
            )
        
        return results
    
    def _save_results(self, results: Dict[str, Any], save_dir: str) -> None:
        """Save evaluation results."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = save_path / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'metrics': results['metrics'],
                'class_names': results['class_names'],
                'num_samples': results['num_samples']
            }
            json.dump(json_results, f, indent=2)
        
        logging.info(f"Saved evaluation metrics to {metrics_path}")
        
        # Save confusion matrix as numpy array
        cm_path = save_path / 'confusion_matrix.npy'
        np.save(cm_path, np.array(results['confusion_matrix']))
        
        logging.info(f"Saved confusion matrix to {cm_path}")
    
    def _create_visualizations(
        self,
        images: List[torch.Tensor],
        targets: List[torch.Tensor],
        predictions: List[torch.Tensor],
        save_dir: str
    ) -> None:
        """Create and save visualizations."""
        save_path = Path(save_dir)
        
        # Concatenate all samples
        all_images = torch.cat(images, dim=0)
        all_targets = torch.cat(targets, dim=0)
        all_predictions = torch.cat(predictions, dim=0)
        
        # Create prediction visualizations
        vis_path = save_path / 'predictions_visualization.png'
        visualize_predictions(
            all_images,
            all_targets,
            all_predictions,
            self.class_colors,
            self.class_names,
            save_path=str(vis_path),
            max_samples=8
        )
        
        # Create confusion matrix visualization
        cm_path = save_path / 'confusion_matrix.png'
        cm = self.metrics.get_confusion_matrix()
        create_confusion_matrix(
            all_targets.numpy(),
            all_predictions.numpy(),
            self.class_names,
            save_path=str(cm_path),
            normalize=True
        )
        
        logging.info(f"Saved visualizations to {save_path}")
    
    def benchmark_inference(
        self,
        dataloader: DataLoader,
        num_warmup: int = 10,
        num_benchmark: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark model inference performance.
        
        Args:
            dataloader: DataLoader for benchmarking
            num_warmup: Number of warmup iterations
            num_benchmark: Number of benchmark iterations
            
        Returns:
            Benchmark results
        """
        self.model.eval()
        
        # Get a single batch for benchmarking
        images, _ = next(iter(dataloader))
        images = images.to(self.device)
        
        # Warmup
        logging.info(f"Warming up for {num_warmup} iterations...")
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(images)
        
        # Benchmark
        logging.info(f"Benchmarking for {num_benchmark} iterations...")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        times = []
        with torch.no_grad():
            for _ in range(num_benchmark):
                start_time = time.time()
                _ = self.model(images)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        results = {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'median_time': float(np.median(times)),
            'fps': float(images.size(0) / np.mean(times)),
            'batch_size': images.size(0)
        }
        
        logging.info("Benchmark Results:")
        logging.info(f"Mean inference time: {results['mean_time']:.4f}s")
        logging.info(f"FPS: {results['fps']:.2f}")
        logging.info(f"Batch size: {results['batch_size']}")
        
        return results
    
    def evaluate_class_distribution(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Analyze class distribution in predictions vs ground truth.
        
        Args:
            dataloader: DataLoader for analysis
            
        Returns:
            Class distribution analysis
        """
        self.model.eval()
        
        gt_class_counts = np.zeros(self.num_classes)
        pred_class_counts = np.zeros(self.num_classes)
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                predictions = torch.argmax(outputs, dim=1)
                
                # Count pixels for each class
                for cls in range(self.num_classes):
                    gt_class_counts[cls] += (targets == cls).sum().item()
                    pred_class_counts[cls] += (predictions.cpu() == cls).sum().item()
        
        # Calculate percentages
        total_gt = gt_class_counts.sum()
        total_pred = pred_class_counts.sum()
        
        results = {}
        for i, class_name in enumerate(self.class_names):
            results[class_name] = {
                'gt_count': int(gt_class_counts[i]),
                'pred_count': int(pred_class_counts[i]),
                'gt_percentage': float(gt_class_counts[i] / total_gt * 100),
                'pred_percentage': float(pred_class_counts[i] / total_pred * 100)
            }
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'num_classes': self.num_classes
        }
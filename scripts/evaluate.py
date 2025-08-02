#!/usr/bin/env python3
"""
Evaluation script for land cover segmentation models.

Usage:
    python scripts/evaluate.py --config config/unet_config.yaml --model-path models/best_model.pth
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch

from src.utils import Config, load_config, setup_logging, get_device
from src.models import create_model
from src.data import create_dataloaders
from src.evaluation import Evaluator


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--save-dir", type=str, default="outputs/evaluation", help="Directory to save results")
    parser.add_argument("--dataset", type=str, choices=['val', 'test'], default='val', help="Dataset to evaluate")
    parser.add_argument("--visualize", action='store_true', help="Create visualizations")
    parser.add_argument("--benchmark", action='store_true', help="Run performance benchmark")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Get device
    device = get_device(config.get('hardware.device'))
    
    # Create model
    model = create_model(config.get('model'))
    
    # Load model checkpoint
    logging.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        logging.info("Loaded model weights")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config.config, config.get('data')
    )
    
    # Select dataset
    if args.dataset == 'val':
        dataloader = val_loader
    elif args.dataset == 'test':
        if test_loader is None:
            logging.error("Test dataset not available")
            return
        dataloader = test_loader
    else:
        logging.error(f"Unknown dataset: {args.dataset}")
        return
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        device=device,
        num_classes=config.get('classes.num_classes'),
        class_names=config.get('classes.class_names'),
        class_colors=config.get('classes.class_colors')
    )
    
    # Run evaluation
    logging.info(f"Evaluating on {args.dataset} dataset...")
    results = evaluator.evaluate(
        dataloader=dataloader,
        save_results=True,
        save_dir=args.save_dir,
        visualize=args.visualize
    )
    
    # Run benchmark if requested
    if args.benchmark:
        logging.info("Running performance benchmark...")
        benchmark_results = evaluator.benchmark_inference(dataloader)
        results['benchmark'] = benchmark_results
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Mean IoU: {results['metrics']['mIoU']:.4f}")
    print(f"Pixel Accuracy: {results['metrics']['PixelAcc']:.4f}")
    print(f"Average inference time: {results['metrics']['avg_inference_time']:.4f}s")
    print(f"FPS: {results['metrics']['fps']:.2f}")
    
    if 'benchmark' in results:
        print(f"\nBenchmark Results:")
        print(f"Mean inference time: {results['benchmark']['mean_time']:.4f}s")
        print(f"FPS: {results['benchmark']['fps']:.2f}")
    
    # Check performance targets
    miou_target = 0.75
    inference_time_target = 2.0
    
    print(f"\nPerformance Targets:")
    print(f"mIoU > {miou_target}: {'✓' if results['metrics']['mIoU'] > miou_target else '✗'}")
    print(f"Inference time < {inference_time_target}s: {'✓' if results['metrics']['avg_inference_time'] < inference_time_target else '✗'}")
    
    logging.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
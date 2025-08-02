#!/usr/bin/env python3
"""
Prediction script for land cover segmentation models.

Usage:
    python scripts/predict.py --image-path path/to/image.jpg --model-path models/best_model.pth --config config/unet_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.utils import Config, load_config, setup_logging, get_device
from src.models import create_model
from src.data.transforms import get_transforms
from src.utils.visualization import mask_to_rgb, hex_to_rgb


def load_and_preprocess_image(image_path: str, config: Config) -> torch.Tensor:
    """Load and preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Get transforms
    transforms = get_transforms('test', config.get('data'))
    
    # Convert to numpy
    image_np = np.array(image, dtype=np.float32) / 255.0
    
    # Apply transforms
    transformed = transforms(image=image_np)
    image_tensor = transformed['image']
    
    # Add batch dimension
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_size


def predict_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Run inference on image."""
    model.eval()
    
    with torch.no_grad():
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(image_tensor)
        
        # Handle model outputs
        if isinstance(outputs, dict):
            outputs = outputs['out']
        
        # Get predictions
        predictions = torch.argmax(outputs, dim=1)
        
    return predictions


def save_prediction(
    prediction: torch.Tensor,
    class_colors: list,
    output_path: str,
    original_size: tuple = None
) -> None:
    """Save prediction as colored image."""
    # Convert to numpy
    pred_np = prediction.squeeze().cpu().numpy()
    
    # Create color map
    color_map = np.array([hex_to_rgb(color) for color in class_colors])
    
    # Convert to RGB
    rgb_image = mask_to_rgb(pred_np, color_map)
    
    # Resize to original size if provided
    if original_size:
        rgb_pil = Image.fromarray(rgb_image)
        rgb_pil = rgb_pil.resize(original_size, Image.NEAREST)
        rgb_image = np.array(rgb_pil)
    
    # Save image
    Image.fromarray(rgb_image).save(output_path)
    logging.info(f"Prediction saved to {output_path}")


def create_visualization(
    original_image: np.ndarray,
    prediction: torch.Tensor,
    class_colors: list,
    class_names: list,
    output_path: str
) -> None:
    """Create side-by-side visualization."""
    # Convert prediction to RGB
    pred_np = prediction.squeeze().cpu().numpy()
    color_map = np.array([hex_to_rgb(color) for color in class_colors])
    pred_rgb = mask_to_rgb(pred_np, color_map)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(pred_rgb)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(hex_to_rgb(color))/255.0, label=name)
        for color, name in zip(class_colors, class_names)
    ]
    fig.legend(
        handles=legend_elements,
        loc='center right',
        bbox_to_anchor=(1.15, 0.5),
        fontsize=10
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Visualization saved to {output_path}")


def analyze_prediction(
    prediction: torch.Tensor,
    class_names: list
) -> dict:
    """Analyze prediction statistics."""
    pred_np = prediction.squeeze().cpu().numpy()
    
    # Count pixels for each class
    unique, counts = np.unique(pred_np, return_counts=True)
    total_pixels = pred_np.size
    
    results = {}
    for class_id, count in zip(unique, counts):
        if class_id < len(class_names):
            class_name = class_names[class_id]
            percentage = (count / total_pixels) * 100
            results[class_name] = {
                'pixels': int(count),
                'percentage': float(percentage)
            }
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Predict land cover segmentation")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Output directory")
    parser.add_argument("--visualize", action='store_true', help="Create visualization")
    parser.add_argument("--analyze", action='store_true', help="Analyze prediction statistics")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    # Load and preprocess image
    logging.info(f"Loading image from {args.image_path}")
    image_tensor, original_size = load_and_preprocess_image(args.image_path, config)
    
    # Keep original image for visualization
    original_image = np.array(Image.open(args.image_path).convert('RGB'))
    
    # Run prediction
    logging.info("Running inference...")
    import time
    start_time = time.time()
    prediction = predict_image(model, image_tensor, device)
    inference_time = time.time() - start_time
    
    logging.info(f"Inference completed in {inference_time:.4f} seconds")
    
    # Generate output filename
    input_name = Path(args.image_path).stem
    
    # Save prediction
    pred_output_path = output_dir / f"{input_name}_prediction.png"
    save_prediction(
        prediction,
        config.get('classes.class_colors'),
        str(pred_output_path),
        original_size
    )
    
    # Create visualization if requested
    if args.visualize:
        vis_output_path = output_dir / f"{input_name}_visualization.png"
        create_visualization(
            original_image,
            prediction,
            config.get('classes.class_colors'),
            config.get('classes.class_names'),
            str(vis_output_path)
        )
    
    # Analyze prediction if requested
    if args.analyze:
        analysis = analyze_prediction(prediction, config.get('classes.class_names'))
        
        print("\n" + "="*40)
        print("PREDICTION ANALYSIS")
        print("="*40)
        print(f"Image: {args.image_path}")
        print(f"Inference time: {inference_time:.4f}s")
        print(f"Image size: {original_size}")
        
        print("\nClass Distribution:")
        for class_name, stats in analysis.items():
            print(f"  {class_name}: {stats['pixels']:,} pixels ({stats['percentage']:.2f}%)")
    
    logging.info("Prediction completed successfully!")


if __name__ == "__main__":
    main()
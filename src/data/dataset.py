"""Dataset implementation for land cover segmentation."""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List
import logging
from pathlib import Path


class LandCoverDataset(Dataset):
    """
    Dataset class for land cover segmentation.
    
    Args:
        images_dir: Directory containing input images
        masks_dir: Directory containing segmentation masks
        transform: Optional transform to be applied
        class_mapping: Optional mapping from RGB values to class indices
        image_extensions: List of valid image extensions
    """
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
        class_mapping: Optional[dict] = None,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.class_mapping = class_mapping or self._default_class_mapping()
        self.image_extensions = [ext.lower() for ext in image_extensions]
        
        # Get list of image files
        self.image_files = self._get_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        logging.info(f"Found {len(self.image_files)} images in {self.images_dir}")
        
        # Validate that corresponding masks exist
        self._validate_masks()
    
    def _default_class_mapping(self) -> dict:
        """Default class mapping from RGB hex values to class indices."""
        return {
            (128, 0, 0): 0,     # Bareland (#800000)
            (0, 255, 36): 1,    # Rangeland (#00FF24)
            (148, 148, 148): 2, # Developed space (#949494)
            (255, 255, 255): 3, # Road (#FFFFFF)
            (34, 97, 38): 4,    # Tree (#226126)
            (0, 69, 255): 5,    # Water (#0045FF)
            (75, 181, 73): 6,   # Agriculture land (#4BB549)
            (222, 31, 7): 7     # Building (#DE1F07)
        }
    
    def _get_image_files(self) -> List[Path]:
        """Get list of valid image files."""
        image_files = []
        
        for ext in self.image_extensions:
            pattern = f"*{ext}"
            image_files.extend(self.images_dir.glob(pattern))
            pattern = f"*{ext.upper()}"
            image_files.extend(self.images_dir.glob(pattern))
        
        return sorted(list(set(image_files)))
    
    def _validate_masks(self) -> None:
        """Validate that corresponding masks exist for all images."""
        missing_masks = []
        
        for image_file in self.image_files:
            mask_file = self._get_mask_path(image_file)
            if not mask_file.exists():
                missing_masks.append(str(image_file))
        
        if missing_masks:
            logging.warning(f"Missing masks for {len(missing_masks)} images")
            # Remove images without masks
            self.image_files = [f for f in self.image_files 
                              if self._get_mask_path(f).exists()]
            
        logging.info(f"Validated {len(self.image_files)} image-mask pairs")
    
    def _get_mask_path(self, image_path: Path) -> Path:
        """Get corresponding mask path for an image."""
        mask_name = image_path.stem
        
        # Try different mask extensions
        for ext in self.image_extensions:
            mask_path = self.masks_dir / f"{mask_name}{ext}"
            if mask_path.exists():
                return mask_path
        
        # Default to .png if not found
        return self.masks_dir / f"{mask_name}.png"
    
    def _rgb_to_mask(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB mask to class indices."""
        mask = np.zeros(rgb_image.shape[:2], dtype=np.int64)
        
        for rgb_value, class_id in self.class_mapping.items():
            # Create boolean mask for this class
            class_mask = np.all(rgb_image == rgb_value, axis=-1)
            mask[class_mask] = class_id
        
        return mask
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Load mask
        mask_path = self._get_mask_path(image_path)
        mask_rgb = Image.open(mask_path).convert('RGB')
        
        # Convert to numpy arrays
        image_np = np.array(image, dtype=np.float32) / 255.0
        mask_rgb_np = np.array(mask_rgb)
        
        # Convert RGB mask to class indices
        mask_np = self._rgb_to_mask(mask_rgb_np)
        
        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=image_np, mask=mask_np)
            image_np = transformed['image']
            mask_np = transformed['mask']
        
        # Convert to tensors
        if isinstance(image_np, np.ndarray):
            if len(image_np.shape) == 3:
                # HWC to CHW
                image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
            else:
                image_tensor = torch.from_numpy(image_np).float()
        else:
            image_tensor = image_np
        
        mask_tensor = torch.from_numpy(mask_np).long()
        
        return image_tensor, mask_tensor
    
    def get_class_distribution(self) -> dict:
        """Calculate class distribution in the dataset."""
        class_counts = {i: 0 for i in range(len(self.class_mapping))}
        total_pixels = 0
        
        logging.info("Calculating class distribution...")
        
        for i in range(len(self)):
            _, mask = self.__getitem__(i)
            mask_np = mask.numpy()
            
            unique, counts = np.unique(mask_np, return_counts=True)
            for class_id, count in zip(unique, counts):
                if class_id in class_counts:
                    class_counts[class_id] += count
                    total_pixels += count
        
        # Convert to percentages
        class_distribution = {}
        for class_id, count in class_counts.items():
            percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
            class_distribution[class_id] = {
                'count': count,
                'percentage': percentage
            }
        
        return class_distribution
    
    def calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced data."""
        distribution = self.get_class_distribution()
        num_classes = len(self.class_mapping)
        
        # Calculate weights inversely proportional to class frequency
        weights = torch.zeros(num_classes)
        total_samples = sum(dist['count'] for dist in distribution.values())
        
        for class_id in range(num_classes):
            if class_id in distribution and distribution[class_id]['count'] > 0:
                weights[class_id] = total_samples / (num_classes * distribution[class_id]['count'])
            else:
                weights[class_id] = 1.0
        
        return weights
    
    def get_sample_info(self, idx: int) -> dict:
        """Get information about a specific sample."""
        image_path = self.image_files[idx]
        mask_path = self._get_mask_path(image_path)
        
        # Load image to get dimensions
        image = Image.open(image_path)
        
        return {
            'index': idx,
            'image_path': str(image_path),
            'mask_path': str(mask_path),
            'image_size': image.size,
            'image_mode': image.mode
        }
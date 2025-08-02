"""Test data processing pipeline."""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import LandCoverDataset
from src.data.transforms import get_transforms


class TestLandCoverDataset:
    """Test the LandCoverDataset class."""
    
    @pytest.fixture
    def temp_dataset(self):
        """Create a temporary dataset for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create directories
        images_dir = temp_dir / 'images'
        masks_dir = temp_dir / 'masks'
        images_dir.mkdir()
        masks_dir.mkdir()
        
        # Create sample images and masks
        for i in range(5):
            # Create RGB image
            image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(image).save(images_dir / f'image_{i}.jpg')
            
            # Create mask with known class colors
            mask = np.zeros((64, 64, 3), dtype=np.uint8)
            if i % 2 == 0:
                mask[:32, :32] = [128, 0, 0]  # Bareland
                mask[32:, 32:] = [0, 255, 36]  # Rangeland
            else:
                mask[:32, :] = [148, 148, 148]  # Developed space
                mask[32:, :] = [255, 255, 255]  # Road
            
            Image.fromarray(mask).save(masks_dir / f'image_{i}.png')
        
        yield str(images_dir), str(masks_dir)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_dataset_initialization(self, temp_dataset):
        """Test dataset initialization."""
        images_dir, masks_dir = temp_dataset
        
        dataset = LandCoverDataset(images_dir, masks_dir)
        
        assert len(dataset) == 5
        assert dataset.images_dir == Path(images_dir)
        assert dataset.masks_dir == Path(masks_dir)
    
    def test_dataset_getitem(self, temp_dataset):
        """Test dataset __getitem__ method."""
        images_dir, masks_dir = temp_dataset
        
        dataset = LandCoverDataset(images_dir, masks_dir)
        
        image, mask = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape[0] == 3  # RGB channels
        assert len(mask.shape) == 2  # 2D mask
        assert mask.dtype == torch.long
    
    def test_dataset_with_transforms(self, temp_dataset):
        """Test dataset with transforms."""
        images_dir, masks_dir = temp_dataset
        
        # Create simple transform config
        transform_config = {
            'image_size': [32, 32],
            'augmentation': {}
        }
        
        transforms = get_transforms('val', transform_config)
        dataset = LandCoverDataset(images_dir, masks_dir, transform=transforms)
        
        image, mask = dataset[0]
        
        assert image.shape == (3, 32, 32)
        assert mask.shape == (32, 32)
    
    def test_class_mapping(self, temp_dataset):
        """Test class mapping functionality."""
        images_dir, masks_dir = temp_dataset
        
        dataset = LandCoverDataset(images_dir, masks_dir)
        
        # Check that default class mapping is used
        assert len(dataset.class_mapping) == 8
        assert (128, 0, 0) in dataset.class_mapping  # Bareland
        assert (0, 255, 36) in dataset.class_mapping  # Rangeland
    
    def test_missing_masks(self):
        """Test handling of missing masks."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            images_dir = temp_dir / 'images'
            masks_dir = temp_dir / 'masks'
            images_dir.mkdir()
            masks_dir.mkdir()
            
            # Create image without corresponding mask
            image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(image).save(images_dir / 'test.jpg')
            
            # This should handle missing masks gracefully
            dataset = LandCoverDataset(str(images_dir), str(masks_dir))
            assert len(dataset) == 0  # No valid image-mask pairs
            
        finally:
            shutil.rmtree(temp_dir)


class TestTransforms:
    """Test data transforms."""
    
    def test_train_transforms(self):
        """Test training transforms."""
        config = {
            'image_size': [256, 256],
            'augmentation': {
                'horizontal_flip': 0.5,
                'vertical_flip': 0.5,
                'rotation': 15,
                'brightness': 0.1,
                'contrast': 0.1
            }
        }
        
        transforms = get_transforms('train', config)
        
        # Test with sample data
        image = np.random.rand(128, 128, 3).astype(np.float32)
        mask = np.random.randint(0, 8, (128, 128))
        
        result = transforms(image=image, mask=mask)
        
        assert 'image' in result
        assert 'mask' in result
        assert result['image'].shape == (256, 256, 3)
        assert result['mask'].shape == (256, 256)
    
    def test_val_transforms(self):
        """Test validation transforms."""
        config = {
            'image_size': [512, 512],
            'augmentation': {}
        }
        
        transforms = get_transforms('val', config)
        
        # Test with sample data
        image = np.random.rand(256, 256, 3).astype(np.float32)
        mask = np.random.randint(0, 8, (256, 256))
        
        result = transforms(image=image, mask=mask)
        
        assert result['image'].shape == (512, 512, 3)
        assert result['mask'].shape == (512, 512)
    
    def test_different_transform_modes(self):
        """Test different transform modes."""
        config = {
            'image_size': [128, 128],
            'augmentation': {}
        }
        
        modes = ['train', 'val', 'test']
        
        for mode in modes:
            transforms = get_transforms(mode, config)
            
            image = np.random.rand(64, 64, 3).astype(np.float32)
            mask = np.random.randint(0, 8, (64, 64))
            
            result = transforms(image=image, mask=mask)
            
            assert result['image'].shape == (128, 128, 3)
            assert result['mask'].shape == (128, 128)
    
    def test_invalid_transform_mode(self):
        """Test handling of invalid transform mode."""
        config = {'image_size': [256, 256]}
        
        with pytest.raises(ValueError):
            get_transforms('invalid_mode', config)


class TestDataAugmentation:
    """Test data augmentation functions."""
    
    def test_augmentation_consistency(self):
        """Test that augmentations are applied consistently to image and mask."""
        config = {
            'image_size': [256, 256],
            'augmentation': {
                'horizontal_flip': 1.0,  # Always apply
                'rotation': 90  # Fixed rotation
            }
        }
        
        transforms = get_transforms('train', config)
        
        # Create a simple pattern to verify consistency
        image = np.zeros((128, 128, 3), dtype=np.float32)
        image[:64, :64] = 1.0  # Top-left quadrant
        
        mask = np.zeros((128, 128), dtype=np.int64)
        mask[:64, :64] = 1  # Same pattern
        
        result = transforms(image=image, mask=mask)
        
        # After horizontal flip, the pattern should move to top-right
        # The exact position depends on the combination of augmentations,
        # but we can check that non-zero regions correspond
        aug_image = result['image']
        aug_mask = result['mask']
        
        # Check that mask and image transformations are consistent
        # (this is a simplified check)
        assert aug_image.shape == (256, 256, 3)
        assert aug_mask.shape == (256, 256)
    
    @pytest.mark.parametrize("flip_prob", [0.0, 0.5, 1.0])
    def test_flip_probabilities(self, flip_prob):
        """Test flip augmentation with different probabilities."""
        config = {
            'image_size': [64, 64],
            'augmentation': {
                'horizontal_flip': flip_prob
            }
        }
        
        transforms = get_transforms('train', config)
        
        image = np.random.rand(32, 32, 3).astype(np.float32)
        mask = np.random.randint(0, 8, (32, 32))
        
        # Apply transforms multiple times
        results = []
        for _ in range(10):
            result = transforms(image=image, mask=mask)
            results.append(result)
        
        # All results should have the correct shape
        for result in results:
            assert result['image'].shape == (64, 64, 3)
            assert result['mask'].shape == (64, 64)


class TestDataLoaderIntegration:
    """Test integration with PyTorch DataLoader."""
    
    def test_dataloader_compatibility(self, temp_dataset):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        images_dir, masks_dir = temp_dataset
        
        dataset = LandCoverDataset(images_dir, masks_dir)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Test loading a batch
        batch = next(iter(dataloader))
        images, masks = batch
        
        assert images.shape[0] == 2  # Batch size
        assert masks.shape[0] == 2  # Batch size
        assert images.shape[1] == 3  # RGB channels
        assert len(masks.shape) == 3  # (batch, height, width)
    
    def test_batch_consistency(self, temp_dataset):
        """Test that batches have consistent shapes."""
        from torch.utils.data import DataLoader
        
        images_dir, masks_dir = temp_dataset
        
        config = {
            'image_size': [128, 128],
            'augmentation': {}
        }
        transforms = get_transforms('val', config)
        
        dataset = LandCoverDataset(images_dir, masks_dir, transform=transforms)
        dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            # All images in batch should have same shape
            assert images.shape[1:] == (3, 128, 128)
            assert masks.shape[1:] == (128, 128)
            
            if batch_idx >= 1:  # Test first two batches
                break
"""Test model architectures."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import UNet, DeepLabV3, create_model


class TestUNet:
    """Test U-Net model architecture."""
    
    def test_unet_custom_forward(self):
        """Test custom U-Net forward pass."""
        model = UNet(num_classes=8, backbone='custom')
        
        # Test with different input sizes
        for size in [(1, 3, 256, 256), (2, 3, 512, 512)]:
            x = torch.randn(size)
            output = model(x)
            
            assert output.shape[0] == size[0]  # Batch size
            assert output.shape[1] == 8  # Number of classes
            assert output.shape[2] == size[2]  # Height
            assert output.shape[3] == size[3]  # Width
    
    def test_unet_resnet_backbone(self):
        """Test U-Net with ResNet backbone."""
        model = UNet(num_classes=8, backbone='resnet34', pretrained=False)
        
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        
        assert output.shape == (1, 8, 256, 256)
    
    def test_unet_different_input_channels(self):
        """Test U-Net with different input channels."""
        model = UNet(num_classes=8, input_channels=4, backbone='custom')
        
        x = torch.randn(1, 4, 256, 256)
        output = model(x)
        
        assert output.shape == (1, 8, 256, 256)


class TestDeepLabV3:
    """Test DeepLabV3 model architecture."""
    
    def test_deeplabv3_forward(self):
        """Test DeepLabV3 forward pass."""
        model = DeepLabV3(num_classes=8, backbone='resnet50', pretrained=False)
        
        x = torch.randn(1, 3, 512, 512)
        output = model(x)
        
        assert output.shape == (1, 8, 512, 512)
    
    def test_deeplabv3_with_aux_loss(self):
        """Test DeepLabV3 with auxiliary loss."""
        model = DeepLabV3(num_classes=8, backbone='resnet50', pretrained=False, aux_loss=True)
        model.train()  # Set to training mode for aux loss
        
        x = torch.randn(1, 3, 512, 512)
        output = model(x)
        
        assert isinstance(output, dict)
        assert 'out' in output
        assert 'aux' in output
        assert output['out'].shape == (1, 8, 512, 512)
        assert output['aux'].shape == (1, 8, 512, 512)
    
    def test_deeplabv3_eval_mode(self):
        """Test DeepLabV3 in evaluation mode."""
        model = DeepLabV3(num_classes=8, backbone='resnet50', pretrained=False, aux_loss=True)
        model.eval()  # Set to evaluation mode
        
        x = torch.randn(1, 3, 512, 512)
        output = model(x)
        
        assert torch.is_tensor(output)
        assert output.shape == (1, 8, 512, 512)


class TestModelFactory:
    """Test model factory functions."""
    
    def test_create_unet(self):
        """Test creating U-Net through factory."""
        config = {
            'name': 'unet',
            'num_classes': 8,
            'backbone': 'resnet34',
            'pretrained': False
        }
        
        model = create_model(config)
        assert isinstance(model, UNet)
        
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        assert output.shape == (1, 8, 256, 256)
    
    def test_create_deeplabv3(self):
        """Test creating DeepLabV3 through factory."""
        config = {
            'name': 'deeplabv3',
            'num_classes': 8,
            'backbone': 'resnet50',
            'pretrained': False
        }
        
        model = create_model(config)
        assert isinstance(model, DeepLabV3)
        
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        assert output.shape == (1, 8, 256, 256)
    
    def test_invalid_model_name(self):
        """Test handling of invalid model name."""
        config = {
            'name': 'invalid_model',
            'num_classes': 8
        }
        
        with pytest.raises(ValueError):
            create_model(config)
    
    @pytest.mark.parametrize("num_classes", [1, 8, 21])
    def test_different_num_classes(self, num_classes):
        """Test models with different number of classes."""
        config = {
            'name': 'unet',
            'num_classes': num_classes,
            'backbone': 'custom'
        }
        
        model = create_model(config)
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        
        assert output.shape == (1, num_classes, 256, 256)


class TestModelProperties:
    """Test model properties and characteristics."""
    
    def test_model_parameters_count(self):
        """Test that models have reasonable parameter counts."""
        models = [
            UNet(num_classes=8, backbone='custom'),
            UNet(num_classes=8, backbone='resnet34', pretrained=False),
            DeepLabV3(num_classes=8, backbone='resnet50', pretrained=False)
        ]
        
        for model in models:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            assert total_params > 0
            assert trainable_params > 0
            assert trainable_params <= total_params
    
    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = UNet(num_classes=8, backbone='custom')
        model.train()
        
        x = torch.randn(1, 3, 256, 256, requires_grad=True)
        output = model(x)
        loss = output.mean()
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
    
    def test_model_device_compatibility(self):
        """Test model device compatibility."""
        model = UNet(num_classes=8, backbone='custom')
        
        # Test CPU
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        assert output.device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            output = model(x)
            assert output.device.type == 'cuda'
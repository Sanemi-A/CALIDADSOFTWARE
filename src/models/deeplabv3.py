"""DeepLabV3 model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import torchvision.models as models


class ASPPConv(nn.Sequential):
    """ASPP Convolution Block."""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """ASPP Pooling Block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""
    
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256):
        super(ASPP, self).__init__()
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
            
        # Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Fusion
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabV3Head(nn.Sequential):
    """DeepLabV3 Classification Head."""
    
    def __init__(self, in_channels: int, num_classes: int):
        super(DeepLabV3Head, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )


class DeepLabV3(nn.Module):
    """
    DeepLabV3 model for semantic segmentation.
    
    Args:
        num_classes: Number of segmentation classes
        backbone: Backbone architecture ('resnet50', 'resnet101')
        pretrained: Whether to use pretrained backbone weights
        dilated: Whether to use dilated convolutions
        aux_loss: Whether to use auxiliary loss
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dilated: bool = True,
        aux_loss: bool = False
    ):
        super(DeepLabV3, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.aux_loss = aux_loss
        
        # Load backbone
        if backbone == 'resnet50':
            backbone_model = models.resnet50(pretrained=pretrained)
            inplanes = 2048
        elif backbone == 'resnet101':
            backbone_model = models.resnet101(pretrained=pretrained)
            inplanes = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Build backbone with dilated convolutions
        self.backbone = self._make_backbone(backbone_model, dilated)
        
        # Classification head
        self.classifier = DeepLabV3Head(inplanes, num_classes)
        
        # Auxiliary classifier (optional)
        if aux_loss:
            self.aux_classifier = nn.Sequential(
                nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, 1)
            )

    def _make_backbone(self, backbone_model: nn.Module, dilated: bool) -> nn.Module:
        """Create backbone with optional dilated convolutions."""
        # Extract layers from backbone
        layers = [
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
            backbone_model.layer1,
            backbone_model.layer2,
            backbone_model.layer3,
            backbone_model.layer4
        ]
        
        if dilated:
            # Modify stride and dilation for layer3 and layer4
            self._modify_layer_for_dilation(layers[6], stride=1, dilation=2)  # layer3
            self._modify_layer_for_dilation(layers[7], stride=1, dilation=4)  # layer4
        
        return nn.Sequential(*layers)

    def _modify_layer_for_dilation(self, layer: nn.Module, stride: int, dilation: int):
        """Modify layer for dilated convolutions."""
        for block in layer:
            if hasattr(block, 'downsample') and block.downsample is not None:
                # Modify downsample layer
                block.downsample[0].stride = (stride, stride)
            
            # Modify convolution layers
            for conv_layer in [block.conv1, block.conv2, getattr(block, 'conv3', None)]:
                if conv_layer is not None:
                    if conv_layer.stride == (2, 2):
                        conv_layer.stride = (stride, stride)
                    if conv_layer.kernel_size == (3, 3):
                        conv_layer.dilation = (dilation, dilation)
                        conv_layer.padding = (dilation, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        
        # Extract features through backbone
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 6:  # After layer3 (for aux loss)
                features.append(x)
        
        features.append(x)  # After layer4
        
        # Main classifier
        x = self.classifier(features[-1])
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        if self.training and self.aux_loss:
            # Auxiliary classifier
            aux_x = self.aux_classifier(features[-2])
            aux_x = F.interpolate(aux_x, size=input_shape, mode='bilinear', align_corners=False)
            return {'out': x, 'aux': aux_x}
        
        return x


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ model with encoder-decoder architecture.
    
    Args:
        num_classes: Number of segmentation classes
        backbone: Backbone architecture ('resnet50', 'resnet101')
        pretrained: Whether to use pretrained backbone weights
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet50',
        pretrained: bool = True
    ):
        super(DeepLabV3Plus, self).__init__()
        self.num_classes = num_classes
        
        # Load backbone
        if backbone == 'resnet50':
            backbone_model = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            backbone_model = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Encoder
        self.backbone = DeepLabV3(num_classes, backbone, pretrained, dilated=True)
        
        # Low-level feature processing
        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),  # 256 (ASPP) + 48 (low-level)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        
        # Extract low-level features (after layer1)
        low_level_features = None
        for i, layer in enumerate(self.backbone.backbone[:5]):  # Through layer1
            x = layer(x)
            if i == 4:  # After layer1
                low_level_features = x
        
        # Continue through backbone to get high-level features
        for layer in self.backbone.backbone[5:]:
            x = layer(x)
        
        # Apply ASPP
        x = self.backbone.classifier[0](x)  # ASPP layer
        
        # Upsample high-level features
        x = F.interpolate(x, size=low_level_features.shape[-2:], 
                         mode='bilinear', align_corners=False)
        
        # Project low-level features
        low_level_features = self.project(low_level_features)
        
        # Concatenate features
        x = torch.cat([x, low_level_features], dim=1)
        
        # Decode
        x = self.decoder(x)
        
        # Final upsampling
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x
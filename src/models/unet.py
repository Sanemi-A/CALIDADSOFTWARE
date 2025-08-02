"""U-Net model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torchvision.models as models


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net model for semantic segmentation.
    
    Args:
        num_classes: Number of segmentation classes
        input_channels: Number of input channels (default: 3 for RGB)
        backbone: Backbone architecture ('custom', 'resnet34', 'resnet50')
        pretrained: Whether to use pretrained backbone weights
        bilinear: Whether to use bilinear upsampling
    """
    
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        backbone: str = 'custom',
        pretrained: bool = True,
        bilinear: bool = False
    ):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.backbone = backbone
        self.bilinear = bilinear

        if backbone == 'custom':
            self._build_custom_unet()
        elif backbone in ['resnet34', 'resnet50']:
            self._build_resnet_unet(backbone, pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def _build_custom_unet(self):
        """Build custom U-Net architecture."""
        self.inc = DoubleConv(self.input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.num_classes)

    def _build_resnet_unet(self, backbone: str, pretrained: bool):
        """Build U-Net with ResNet backbone."""
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone}")

        # Encoder
        self.encoder_conv1 = nn.Conv2d(self.input_channels, 64, 7, 2, 3, bias=False)
        if pretrained and self.input_channels == 3:
            self.encoder_conv1.weight = resnet.conv1.weight
        
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4

        # Decoder
        self.decoder4 = self._make_decoder_layer(channels[4], channels[3])
        self.decoder3 = self._make_decoder_layer(channels[3], channels[2])
        self.decoder2 = self._make_decoder_layer(channels[2], channels[1])
        self.decoder1 = self._make_decoder_layer(channels[1], channels[0])
        
        # Final layer
        self.final_conv = nn.Conv2d(channels[0], self.num_classes, 1)

    def _make_decoder_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create decoder layer."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone == 'custom':
            return self._forward_custom(x)
        else:
            return self._forward_resnet(x)

    def _forward_custom(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for custom U-Net."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def _forward_resnet(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ResNet-based U-Net."""
        # Encoder
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu(x)
        e1 = x
        
        x = self.encoder_maxpool(x)
        e2 = self.encoder_layer1(x)
        e3 = self.encoder_layer2(e2)
        e4 = self.encoder_layer3(e3)
        e5 = self.encoder_layer4(e4)

        # Decoder
        d4 = self.decoder4(e5)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        
        d3 = self.decoder3(d4 + e4)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        
        d2 = self.decoder2(d3 + e3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        
        d1 = self.decoder1(d2 + e2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)

        # Final prediction
        logits = self.final_conv(d1 + e1)
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return logits
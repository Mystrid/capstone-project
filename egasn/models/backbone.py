import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PrunedASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module with pruned depthwise separable convolutions.
    Extracts multi-scale spatial context.
    """
    def __init__(self, in_channels: int, out_channels: int = 256, dilations: list = [6, 12, 18]):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = self._build_aspp_conv(in_channels, out_channels, dilations[0])
        self.conv3 = self._build_aspp_conv(in_channels, out_channels, dilations[1])
        self.conv4 = self._build_aspp_conv(in_channels, out_channels, dilations[2])
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def _build_aspp_conv(self, in_channels: int, out_channels: int, dilation: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=dilation, dilation=dilation, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PrunedASPP.
        Args:
            x (torch.Tensor): Input feature map.
        Returns:
            torch.Tensor: ASPP output feature map.
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(out)


class MobileNetV2Encoder(nn.Module):
    """
    MobileNetV2 feature extractor backbone.
    Extracts low-level and high-level features for segmentation.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # low-level features: features[:4] -> stride 4, 24 channels
        self.low_level_features = nn.Sequential(*mobilenet.features[:4])
        
        # high-level features: features[4:] -> stride 32, 1280 channels
        self.high_level_features = nn.Sequential(*mobilenet.features[4:])
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for MobileNetV2Encoder.
        Args:
            x (torch.Tensor): Input image batch.
        Returns:
            tuple: (low_level_features, high_level_features)
        """
        low_level = self.low_level_features(x)
        high_level = self.high_level_features(low_level)
        return low_level, high_level

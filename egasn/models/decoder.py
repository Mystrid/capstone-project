import torch
import torch.nn as nn
import torch.nn.functional as F

class EGASNDecoder(nn.Module):
    """
    Decoder module for the EGASN framework.
    Combines high-level ASPP features with low-level spatial features from the backbone.
    """
    def __init__(self, high_channels: int = 256, low_channels: int = 24, num_classes: int = 19):
        super().__init__()
        
        # 1x1 conv to project low level features
        self.project_low = nn.Sequential(
            nn.Conv2d(low_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block
        in_channels = high_channels + 48
        self.decoder_block = nn.Sequential(
            # First depthwise separable conv
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Second depthwise separable conv
            nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Final classification head
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, high_level: torch.Tensor, low_level: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for EGASNDecoder.
        Args:
            high_level (torch.Tensor): High-level features (e.g., from ASPP) at H/32.
            low_level (torch.Tensor): Low-level features from backbone at H/4.
        Returns:
            torch.Tensor: Segmentation logits upsampled to original resolution.
        """
        # Upsample high_level by 8x to match low_level size (H/4)
        high_level_upsampled = F.interpolate(high_level, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        
        # Project low level features
        low_level_proj = self.project_low(low_level)
        
        # Concatenate features
        concat_features = torch.cat([high_level_upsampled, low_level_proj], dim=1)
        
        # Apply decoder block
        decoded = self.decoder_block(concat_features)
        
        # Get logits
        logits = self.classifier(decoded)
        
        # Upsample by 4x to match original resolution
        logits_upsampled = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        
        return logits_upsampled


class DirichletHead(nn.Module):
    """
    Evidential deep learning head utilizing Dirichlet distribution 
    for uncertainty estimation.
    """
    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor) -> dict:
        """
        Forward pass for DirichletHead.
        Args:
            logits (torch.Tensor): Raw model outputs (B, num_classes, H, W).
        Returns:
            dict: Containing logits, evidence, alpha, probs, and uncertainty.
        """
        evidence = F.softplus(logits)
        alpha = evidence + 1
        alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
        
        probs = alpha / alpha_sum
        uncertainty = self.num_classes / alpha_sum
        
        return {
            'logits': logits,
            'evidence': evidence,
            'alpha': alpha,
            'probs': probs,
            'uncertainty': uncertainty
        }

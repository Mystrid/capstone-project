import torch
import torch.nn as nn

from egasn.models.backbone import MobileNetV2Encoder, PrunedASPP
from egasn.models.decoder import EGASNDecoder, DirichletHead
from egasn.models.cross_attention import CrossAttentionGating
from egasn.graph.feature_warper import warp_features

class EGASN(nn.Module):
    """
    Epipolar Graph-Attention Segmentation Network (EGASN).
    Hybrid segmentation framework for autonomous vehicles combining MobileNetV2+ASPP,
    cross-attention gating, and evidential deep learning for uncertainty estimation.
    """
    def __init__(self, 
                 num_classes: int = 19, 
                 aspp_dilations: list = [6, 12, 18], 
                 feature_channels: int = 256, 
                 attention_heads: int = 4, 
                 lambda_init: float = 0.1):
        super().__init__()
        
        self.encoder = MobileNetV2Encoder()
        
        # High level output from MobileNetV2Encoder is 1280 channels
        self.aspp = PrunedASPP(in_channels=1280, out_channels=feature_channels, dilations=aspp_dilations)
        
        # Low level output is 24 channels
        self.decoder = EGASNDecoder(high_channels=feature_channels, low_channels=24, num_classes=num_classes)
        
        self.dirichlet_head = DirichletHead(num_classes=num_classes)
        
        self.cross_attention = CrossAttentionGating(
            channels=feature_channels, 
            num_heads=attention_heads, 
            lambda_init=lambda_init
        )

    def forward(self, x: torch.Tensor, x_prev_features: torch.Tensor = None, flow_field: torch.Tensor = None) -> dict:
        """
        Forward pass for EGASN.
        Args:
            x (torch.Tensor): Current input frame (B, 3, H, W).
            x_prev_features (torch.Tensor, optional): Cached ASPP features from the previous frame.
            flow_field (torch.Tensor, optional): Optical flow or epipolar correspondence field for warping.
        Returns:
            dict: Model outputs including segmentation logits, probabilities, uncertainty, and features.
        """
        # Encode features
        low_level, high_level = self.encoder(x)
        
        # Spatial feature extraction
        spatial_features = self.aspp(high_level)
        
        # Temporal feature fusion
        if x_prev_features is not None and flow_field is not None:
            warped = warp_features(x_prev_features, flow_field)
            fused = self.cross_attention(spatial_features, warped)
        else:
            fused = spatial_features
            
        # Decode and compute output probabilities and uncertainty
        logits = self.decoder(fused, low_level)
        output = self.dirichlet_head(logits)
        
        # Cache fused features for the next frame
        output['features'] = fused
        
        return output

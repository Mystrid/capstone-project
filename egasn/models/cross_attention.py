import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionGating(nn.Module):
    """
    Cross-Attention Gating module to adaptively fuse current features with 
    warped features from the previous frame.
    """
    def __init__(self, channels: int = 256, num_heads: int = 4, lambda_init: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.lambda_scale = nn.Parameter(torch.tensor(lambda_init))
        
        self.W_q = nn.Linear(channels, channels)
        self.W_k = nn.Linear(channels, channels)
        self.W_v = nn.Linear(channels, channels)

    def forward(self, x_current: torch.Tensor, x_warped: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for CrossAttentionGating.
        Args:
            x_current (torch.Tensor): Features of the current frame (B, C, H, W).
            x_warped (torch.Tensor, optional): Warped features from previous frame (B, C, H, W).
        Returns:
            torch.Tensor: Fused features (B, C, H, W).
        """
        if x_warped is None:
            return x_current
            
        B, C, H, W = x_current.shape
        
        # Reshape to (B, H*W, C)
        q_in = x_current.view(B, C, -1).permute(0, 2, 1)
        k_in = x_warped.view(B, C, -1).permute(0, 2, 1)
        v_in = x_warped.view(B, C, -1).permute(0, 2, 1)
        
        # Linear projections
        Q = self.W_q(q_in)
        K = self.W_k(k_in)
        V = self.W_v(v_in)
        
        # Split into heads (B, N, num_heads, head_dim) and permute to (B, num_heads, N, head_dim)
        N = H * W
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention: Q @ K^T / sqrt(d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn_scores, dim=-1)
        
        # Context = A @ V
        context = torch.matmul(attn, V)
        
        # Merge heads: (B, num_heads, N, head_dim) -> (B, N, C)
        context = context.transpose(1, 2).contiguous().view(B, N, C)
        
        # Reshape back to (B, C, H, W)
        context = context.permute(0, 2, 1).view(B, C, H, W)
        
        # Adaptive gating
        output = x_current + self.lambda_scale * context
        return output

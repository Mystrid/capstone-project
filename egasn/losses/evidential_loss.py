import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EvidentialLoss(nn.Module):
    """
    Evidential deep learning loss using Dirichlet distribution.
    Computes MSE and KL divergence for evidential uncertainty estimation.
    """
    def __init__(self, num_classes: int = 19, beta_max: float = 1.0, annealing_epochs: int = 10, gamma_tc: float = 0.5):
        super(EvidentialLoss, self).__init__()
        self.num_classes = num_classes
        self.beta_max = beta_max
        self.annealing_epochs = annealing_epochs
        self.gamma_tc = gamma_tc
        self.current_epoch = 0

    def _kl_divergence(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the KL Divergence KL[Dir(alpha_tilde) || Dir(1)]
        """
        alpha_tilde = target + (1 - target) * alpha
        S_alpha = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        # KL = lgamma(sum(alpha_tilde)) - sum(lgamma(alpha_tilde)) - lgamma(K) + sum((alpha_tilde - 1) * (digamma(alpha_tilde) - digamma(sum(alpha_tilde))))
        kl = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) - math.lgamma(self.num_classes)
        kl = kl + torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_alpha)), dim=1, keepdim=True)
        return kl.squeeze(1)

    def _mse_loss(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the expected MSE loss.
        """
        S = torch.sum(alpha, dim=1, keepdim=True)
        p = alpha / S
        
        # Expected MSE = sum((target - p)^2 + p*(1-p)/(S+1), dim=1)
        mse = torch.sum((target - p) ** 2 + p * (1 - p) / (S + 1), dim=1)
        return mse

    def forward(self, output_dict: dict, target: torch.Tensor, tc_loss: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for computing the evidential loss.
        
        Args:
            output_dict: Dictionary containing 'alpha' tensor of shape (B, K, H, W)
            target: Ground truth tensor of shape (B, H, W) with ignore index 255
            tc_loss: Optional temporal consistency loss
        """
        alpha = output_dict['alpha']  # (B, K, H, W)
        B, K, H, W = alpha.shape
        
        valid_mask = target != 255
        target_valid = target.clone()
        target_valid[~valid_mask] = 0
        
        target_onehot = F.one_hot(target_valid, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Apply valid mask implicitly by zeroing out the loss later
        L_mse = self._mse_loss(alpha, target_onehot)
        L_kl = self._kl_divergence(alpha, target_onehot)
        
        # Zero out loss for invalid pixels
        L_mse = L_mse * valid_mask.float()
        L_kl = L_kl * valid_mask.float()
        
        # Compute mean over valid pixels
        num_valid = valid_mask.float().sum()
        if num_valid > 0:
            L_mse = L_mse.sum() / num_valid
            L_kl = L_kl.sum() / num_valid
        else:
            L_mse = L_mse.sum() * 0.0
            L_kl = L_kl.sum() * 0.0
            
        beta = min(self.current_epoch / self.annealing_epochs, 1.0) * self.beta_max
        loss = L_mse + beta * L_kl
        
        if tc_loss is not None:
            loss += self.gamma_tc * tc_loss
            
        return loss

    def set_epoch(self, epoch: int):
        """Sets the current epoch for beta annealing."""
        self.current_epoch = epoch

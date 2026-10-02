"""Evidential and temporal consistency loss interface for EGASN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from egasn.losses.temporal_consistency import TemporalConsistencyLoss


class EvidentialLoss(nn.Module):
    """Evidential deep learning loss using Dirichlet distribution."""

    def __init__(
        self,
        num_classes: int = 19,
        beta_max: float = 1.0,
        annealing_epochs: int = 10,
        gamma_tc: float = 0.5,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.beta_max = beta_max
        self.annealing_epochs = annealing_epochs
        self.gamma_tc = gamma_tc
        self.ignore_index = ignore_index
        self.current_epoch = 0

    @property
    def beta(self) -> float:
        """Computes current annealing factor beta."""
        if self.annealing_epochs <= 0:
            return self.beta_max
        return min(self.current_epoch / self.annealing_epochs, 1.0) * self.beta_max

    def set_epoch(self, epoch: int):
        """Sets the current epoch for beta annealing."""
        self.current_epoch = epoch

    def kl_divergence(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes KL Divergence KL[Dir(alpha_tilde) || Dir(1)]."""
        if target.dim() == 3:
            valid_mask = target != self.ignore_index
            target_clamped = target.clone()
            target_clamped[~valid_mask] = 0
            target_onehot = F.one_hot(target_clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        else:
            target_onehot = target

        alpha_tilde = target_onehot + (1.0 - target_onehot) * alpha
        S_alpha = torch.sum(alpha_tilde, dim=1, keepdim=True)

        kl = (
            torch.lgamma(S_alpha)
            - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
            - math.lgamma(self.num_classes)
        )
        kl = kl + torch.sum(
            (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_alpha)),
            dim=1,
            keepdim=True,
        )
        return kl.squeeze(1)

    def _mse_loss(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes expected MSE loss."""
        S = torch.sum(alpha, dim=1, keepdim=True)
        p = alpha / S
        return torch.sum((target - p) ** 2 + p * (1.0 - p) / (S + 1.0), dim=1)

    def forward(
        self,
        output_dict: dict,
        target: torch.Tensor,
        tc_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes combined evidential loss."""
        alpha = output_dict["alpha"]
        valid_mask = target != self.ignore_index

        if not valid_mask.any():
            return torch.tensor(0.0, device=alpha.device, requires_grad=True)

        target_valid = target.clone()
        target_valid[~valid_mask] = 0
        target_onehot = F.one_hot(target_valid, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        L_mse = self._mse_loss(alpha, target_onehot) * valid_mask.float()
        L_kl = self.kl_divergence(alpha, target) * valid_mask.float()

        num_valid = valid_mask.float().sum()
        L_mse = L_mse.sum() / num_valid
        L_kl = L_kl.sum() / num_valid

        loss = L_mse + self.beta * L_kl
        if tc_loss is not None:
            loss = loss + self.gamma_tc * tc_loss

        return loss


__all__ = ["EvidentialLoss", "TemporalConsistencyLoss"]

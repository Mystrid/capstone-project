import torch
import torch.nn as nn

class TemporalConsistencyLoss(nn.Module):
    """
    Computes temporal consistency loss between current frame predictions
    and warped previous frame predictions.
    """
    def __init__(self):
        super(TemporalConsistencyLoss, self).__init__()
        
    def forward(self, pred_curr: torch.Tensor, pred_prev_warped: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred_curr: (B, K, H, W) current frame probabilities
            pred_prev_warped: (B, K, H, W) previous frame predictions warped by flow
            valid_mask: (B, 1, H, W) optional mask (1=valid flow, 0=invalid/no flow)
        Returns:
            Scalar loss tensor
        """
        diff_sq = (pred_curr - pred_prev_warped) ** 2
        L_TC = torch.sum(diff_sq, dim=1, keepdim=True)
        
        if valid_mask is not None:
            L_TC = L_TC * valid_mask
            num_valid = valid_mask.sum()
            if num_valid > 0:
                loss = L_TC.sum() / num_valid
            else:
                loss = L_TC.sum() * 0.0
        else:
            loss = L_TC.mean()
            
        return loss

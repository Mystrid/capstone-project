import torch

def dirichlet_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    """
    Computes uncertainty from Dirichlet concentration parameters.
    
    Args:
        alpha: (B, K, H, W) Dirichlet concentration parameters
        
    Returns:
        Uncertainty map (B, 1, H, W)
    """
    K = alpha.size(1)
    S = alpha.sum(dim=1, keepdim=True)
    uncertainty = K / S
    return uncertainty

def flag_high_uncertainty(uncertainty_map: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Flags pixels with uncertainty exceeding the threshold.
    
    Args:
        uncertainty_map: (B, 1, H, W) uncertainty values
        threshold: Threshold above which to flag uncertainty
        
    Returns:
        Binary mask (B, 1, H, W) of high uncertainty pixels
    """
    return (uncertainty_map > threshold).float()

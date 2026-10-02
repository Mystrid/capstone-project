import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any

from egasn.graph.keypoint_extractor import ORBKeypointExtractor
from egasn.graph.epipolar_filter import EpipolarFilter
from egasn.graph.flow_interpolator import DelaunayFlowInterpolator


def warp_features(features: torch.Tensor, flow_field: torch.Tensor) -> torch.Tensor:
    """
    Warps historical features using a pseudo-dense flow field.

    Args:
        features (torch.Tensor): Tensor of shape (B, C, H, W) representing historical features X_{t-1}.
        flow_field (torch.Tensor): Tensor of shape (B, 2, H, W) or (2, H, W) representing displacement.
                                   Channel 0 is flow_x, Channel 1 is flow_y.

    Returns:
        torch.Tensor: Warped features of shape (B, C, H, W).
    """
    if flow_field.dim() == 3:
        flow_field = flow_field.unsqueeze(0)  # Make it (1, 2, H, W)
        
    B, C, H, W = features.shape
    
    # Create normalized base grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=features.device, dtype=features.dtype),
        torch.arange(W, device=features.device, dtype=features.dtype),
        indexing='ij'
    )
    
    # Expand base grid to batch size
    base_grid_x = x_grid.unsqueeze(0).expand(B, -1, -1)
    base_grid_y = y_grid.unsqueeze(0).expand(B, -1, -1)
    
    # Add flow displacement to base grid
    flow_x = flow_field[:, 0, :, :].to(features.device)
    flow_y = flow_field[:, 1, :, :].to(features.device)
    
    new_x = base_grid_x + flow_x
    new_y = base_grid_y + flow_y
    
    # Normalize coordinates to [-1, 1] range for grid_sample
    norm_x = 2.0 * new_x / (W - 1) - 1.0
    norm_y = 2.0 * new_y / (H - 1) - 1.0
    
    # grid_sample expects grid of shape (B, H, W, 2) with (x, y) coordinates
    grid = torch.stack((norm_x, norm_y), dim=3)
    
    # Apply bilinear interpolation to warp features
    warped_features = F.grid_sample(
        features, 
        grid, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    
    return warped_features


class EpipolarGraphConstructor:
    """
    Integrates keypoint extraction, epipolar filtering, and flow interpolation 
    to construct the epipolar graph for cross-attention gating.
    """
    def __init__(self, 
                 output_size: Tuple[int, int],
                 n_features: int = 2000,
                 ransac_threshold: float = 3.0,
                 min_inliers: int = 8):
        """
        Initializes the entire graph pipeline.

        Args:
            output_size (Tuple[int, int]): Spatial dimensions (H', W') of the target feature map.
            n_features (int): Number of ORB features to extract.
            ransac_threshold (float): Reprojection threshold for RANSAC fundamental matrix estimation.
            min_inliers (int): Minimum inlier matches required to compute valid flow.
        """
        self.output_size = output_size
        self.extractor = ORBKeypointExtractor(n_features=n_features)
        self.filter = EpipolarFilter(ransac_threshold=ransac_threshold, min_inliers=min_inliers)
        self.interpolator = DelaunayFlowInterpolator(output_size=output_size)

    def construct(self, frame_prev: np.ndarray, frame_curr: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes the full pipeline: extract -> match -> filter -> interpolate.

        Args:
            frame_prev (np.ndarray): Previous image frame.
            frame_curr (np.ndarray): Current image frame.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: 
                - flow_field tensor of shape (2, H', W')
                - info_dict with pipeline metadata (num_keypoints, num_matches, num_inliers, fundamental_matrix)
        """
        h, w = frame_curr.shape[:2]
        image_size = (h, w)
        
        info_dict = {
            "num_keypoints": 0,
            "num_matches": 0,
            "num_inliers": 0,
            "fundamental_matrix": None
        }
        
        # 1. Extract and match keypoints
        kp_prev, kp_curr, matches = self.extractor.extract_and_match(frame_prev, frame_curr)
        info_dict["num_keypoints"] = len(kp_prev)
        info_dict["num_matches"] = len(matches)
        
        # 2. Filter matches using epipolar geometry constraint
        pts_prev, pts_curr, F_mat = self.filter.filter_matches(kp_prev, kp_curr, matches)
        info_dict["num_inliers"] = len(pts_prev)
        info_dict["fundamental_matrix"] = F_mat
        
        # 3. Interpolate flow field from filtered matches
        if len(pts_prev) < 3:
            # Fallback if there are not enough points for Delaunay triangulation
            flow_field = torch.zeros((2, self.output_size[0], self.output_size[1]), dtype=torch.float32)
        else:
            flow_field = self.interpolator.construct_graph(pts_prev, pts_curr, image_size)
            
        return flow_field, info_dict

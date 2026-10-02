import numpy as np
import torch
import scipy.spatial
from typing import Tuple

class DelaunayFlowInterpolator:
    """
    Interpolates sparse motion vectors into a dense flow field using Delaunay triangulation.
    The resulting flow field maps from the original image resolution to the feature map resolution.
    """
    def __init__(self, output_size: Tuple[int, int]):
        """
        Initializes the flow interpolator.

        Args:
            output_size (Tuple[int, int]): The (height, width) of the target feature map.
        """
        self.output_size = output_size
        self.h, self.w = output_size

        # Precompute target meshgrid
        x = np.arange(self.w)
        y = np.arange(self.h)
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        self.grid_pts = np.stack((self.grid_x.ravel(), self.grid_y.ravel()), axis=1)

    def interpolate(self, pts_prev: np.ndarray, pts_curr: np.ndarray, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Interpolates sparse points to create a pseudo-dense flow field.

        Args:
            pts_prev (np.ndarray): Keypoints in previous frame, shape (N, 2).
            pts_curr (np.ndarray): Keypoints in current frame, shape (N, 2).
            image_size (Tuple[int, int]): The (height, width) of the original image frames.

        Returns:
            torch.Tensor: Interpolated flow field of shape (2, H', W'), where H', W' are the output_size.
                          Channel 0 corresponds to flow_x and Channel 1 corresponds to flow_y.
        """
        if len(pts_prev) < 3:
            # Minimum 3 points required for Delaunay triangulation
            return torch.zeros((2, self.h, self.w), dtype=torch.float32)

        flow = pts_curr - pts_prev  # (N, 2)
        
        orig_h, orig_w = image_size
        
        # Scale spatial positions and flow vectors to output resolution
        scale_x = self.w / orig_w
        scale_y = self.h / orig_h
        
        scaled_pts_curr = pts_curr * np.array([scale_x, scale_y])
        scaled_flow = flow * np.array([scale_x, scale_y])

        try:
            tri = scipy.spatial.Delaunay(scaled_pts_curr)
        except scipy.spatial.qhull.QhullError:
            # Handling collinear points or other Qhull failures gracefully
            return torch.zeros((2, self.h, self.w), dtype=torch.float32)

        simplices = tri.find_simplex(self.grid_pts)
        valid_mask = simplices != -1
        
        dense_flow_x = np.zeros(self.h * self.w, dtype=np.float32)
        dense_flow_y = np.zeros(self.h * self.w, dtype=np.float32)

        if np.any(valid_mask):
            valid_pts = self.grid_pts[valid_mask]
            valid_simplices = simplices[valid_mask]

            # Get the vertices of the simplices containing the points
            tri_vertices = tri.simplices[valid_simplices]

            # Compute barycentric coordinates
            transform = tri.transform[valid_simplices]
            bary = np.einsum('ijk,ik->ij', transform[:, :2, :2], valid_pts - transform[:, 2, :])
            bary_last = 1.0 - bary.sum(axis=1)
            bary_coords = np.c_[bary, bary_last]

            # Get the flow vectors at the vertices
            vertex_flow_x = scaled_flow[:, 0][tri_vertices]
            vertex_flow_y = scaled_flow[:, 1][tri_vertices]

            # Interpolate the flow using barycentric weights
            interp_flow_x = np.sum(vertex_flow_x * bary_coords, axis=1)
            interp_flow_y = np.sum(vertex_flow_y * bary_coords, axis=1)

            dense_flow_x[valid_mask] = interp_flow_x
            dense_flow_y[valid_mask] = interp_flow_y

        flow_field = np.stack((dense_flow_x.reshape(self.h, self.w), 
                               dense_flow_y.reshape(self.h, self.w)), axis=0)

        return torch.from_numpy(flow_field)

    def construct_graph(self, pts_prev: np.ndarray, pts_curr: np.ndarray, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Convenience wrapper to construct the flow field graph.

        Args:
            pts_prev (np.ndarray): Previous frame points.
            pts_curr (np.ndarray): Current frame points.
            image_size (Tuple[int, int]): (H, W) of input image.

        Returns:
            torch.Tensor: Dense flow field.
        """
        return self.interpolate(pts_prev, pts_curr, image_size)

"""Epipolar graph construction and correspondence utilities for EGASN."""

from typing import Tuple, List, Optional, Any, Dict
import cv2
import numpy as np
import torch
import torch.nn as nn

from egasn.graph.feature_warper import warp_features
from egasn.graph.keypoint_extractor import ORBKeypointExtractor
from egasn.graph.epipolar_filter import EpipolarFilter
from egasn.graph.flow_interpolator import DelaunayFlowInterpolator


class FeatureWarper(nn.Module):
    """Warps feature maps using dense flow fields."""

    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor, flow_field: torch.Tensor) -> torch.Tensor:
        """Warps feature maps using bilinear grid sampling.

        Args:
            features: Tensor of shape (B, C, H, W).
            flow_field: Tensor of shape (B, 2, H, W) or (2, H, W).

        Returns:
            Warped feature tensor of shape (B, C, H, W).
        """
        return warp_features(features, flow_field)


class EpipolarGraphConstructor:
    """Epipolar graph constructor integrating keypoints, RANSAC, and flow interpolation."""

    def __init__(
        self,
        output_size: Optional[Tuple[int, int]] = None,
        max_features: int = 2000,
        ransac_threshold: float = 3.0,
        min_inliers: int = 8,
        ratio_test: float = 0.75,
        **kwargs: Any,
    ):
        """Initializes the graph constructor.

        Args:
            output_size: Output flow dimensions (H, W). If None, inferred from input.
            max_features: Maximum ORB features to detect.
            ransac_threshold: RANSAC threshold for Fundamental matrix.
            min_inliers: Minimum inliers to accept epipolar fit.
            ratio_test: Lowe's ratio test threshold.
        """
        n_features = kwargs.get("n_features", max_features)
        self.output_size = output_size
        self.max_features = n_features
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.ratio_test = ratio_test

        self.extractor = ORBKeypointExtractor(
            n_features=n_features,
            ratio_test=ratio_test,
        )
        self.filter = EpipolarFilter(
            ransac_threshold=ransac_threshold,
            min_inliers=min_inliers,
        )

    def extract_features(self, img: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Extracts ORB keypoints and descriptors from an image.

        Args:
            img: Input BGR/RGB image array.

        Returns:
            Tuple of (keypoints, descriptors).
        """
        return self.extractor.extract_keypoints(img)

    def match_features(self, des1: np.ndarray, des2: np.ndarray) -> List[cv2.DMatch]:
        """Matches descriptors using FLANN with ratio test.

        Args:
            des1: Descriptors from frame 1.
            des2: Descriptors from frame 2.

        Returns:
            List of filtered DMatch objects.
        """
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []
        matches = self.extractor.matcher.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.ratio_test * n.distance:
                    good.append(m)
        return good

    def filter_epipolar(
        self, pts1: np.ndarray, pts2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filters matched points using Fundamental matrix RANSAC.

        Args:
            pts1: Points in frame 1 of shape (N, 2).
            pts2: Points in frame 2 of shape (N, 2).

        Returns:
            Tuple of inlier points (pts1_inliers, pts2_inliers).
        """
        if len(pts1) < self.min_inliers or len(pts2) < self.min_inliers:
            return pts1, pts2

        F, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.FM_RANSAC, self.ransac_threshold, 0.99
        )
        if F is None or mask is None:
            return pts1, pts2

        mask = mask.ravel() == 1
        if np.sum(mask) >= self.min_inliers:
            return pts1[mask], pts2[mask]
        return pts1, pts2

    def construct(
        self, frame_prev: np.ndarray, frame_curr: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Constructs pseudo-dense flow field and valid mask between consecutive frames.

        Args:
            frame_prev: Previous frame (H, W, 3).
            frame_curr: Current frame (H, W, 3).

        Returns:
            Tuple of (flow_field, valid_mask):
                - flow_field: Tensor of shape (2, H_out, W_out).
                - valid_mask: Tensor of shape (1, H_out, W_out).
        """
        h, w = frame_curr.shape[:2]
        target_size = self.output_size if self.output_size is not None else (h, w)
        h_out, w_out = target_size

        # Check for identical frames
        if np.array_equal(frame_prev, frame_curr):
            flow = torch.zeros((2, h_out, w_out), dtype=torch.float32)
            valid_mask = torch.ones((1, h_out, w_out), dtype=torch.float32)
            return flow, valid_mask

        kp_prev, des_prev = self.extract_features(frame_prev)
        kp_curr, des_curr = self.extract_features(frame_curr)

        matches = self.match_features(des_prev, des_curr)

        if len(matches) < self.min_inliers:
            flow = torch.zeros((2, h_out, w_out), dtype=torch.float32)
            valid_mask = torch.zeros((1, h_out, w_out), dtype=torch.float32)
            return flow, valid_mask

        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches])

        pts_prev_inliers, pts_curr_inliers, _ = self.filter.filter_matches(
            kp_prev, kp_curr, matches
        )

        if len(pts_prev_inliers) < 3:
            flow = torch.zeros((2, h_out, w_out), dtype=torch.float32)
            valid_mask = torch.zeros((1, h_out, w_out), dtype=torch.float32)
            return flow, valid_mask

        interpolator = DelaunayFlowInterpolator(output_size=target_size)
        flow = interpolator.interpolate(pts_prev_inliers, pts_curr_inliers, (h, w))
        valid_mask = torch.ones((1, h_out, w_out), dtype=torch.float32)

        return flow, valid_mask


__all__ = [
    "EpipolarGraphConstructor",
    "FeatureWarper",
    "warp_features",
]

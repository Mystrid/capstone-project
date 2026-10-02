import cv2
import numpy as np
from typing import Tuple, List

class EpipolarFilter:
    """
    Filters keypoint matches using epipolar geometry via RANSAC.
    Ensures that matched keypoints satisfy the epipolar constraint.
    """
    def __init__(self, ransac_threshold: float = 3.0, min_inliers: int = 8):
        """
        Initializes the epipolar filter.

        Args:
            ransac_threshold (float): RANSAC reprojection threshold for fundamental matrix estimation.
            min_inliers (int): Minimum number of inliers required to form a valid match set.
        """
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers

    def filter_matches(self, 
                       kp_prev: List[cv2.KeyPoint], 
                       kp_curr: List[cv2.KeyPoint], 
                       matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filters matches by estimating the Fundamental matrix.

        Args:
            kp_prev (List[cv2.KeyPoint]): Keypoints from the previous frame.
            kp_curr (List[cv2.KeyPoint]): Keypoints from the current frame.
            matches (List[cv2.DMatch]): Matches between the keypoints.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Inlier points from previous frame (N, 2),
            inlier points from current frame (N, 2), and the estimated Fundamental matrix (3, 3).
            Returns empty arrays if the number of inliers is below min_inliers.
        """
        if len(matches) < self.min_inliers:
            return np.array([]), np.array([]), np.array([])

        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        if len(pts_prev) < 8:
            # Cannot compute fundamental matrix reliably with less than 8 points
            return np.array([]), np.array([]), np.array([])

        F, mask = cv2.findFundamentalMat(
            pts_prev, 
            pts_curr, 
            cv2.FM_RANSAC, 
            ransacReprojThreshold=self.ransac_threshold
        )
        
        if F is None or mask is None:
            return np.array([]), np.array([]), np.array([])

        mask = mask.ravel() == 1
        num_inliers = np.sum(mask)

        if num_inliers < self.min_inliers:
            return np.array([]), np.array([]), np.array([])

        inlier_pts_prev = pts_prev[mask]
        inlier_pts_curr = pts_curr[mask]

        return inlier_pts_prev, inlier_pts_curr, F

    def compute_epipolar_error(self, pts_prev: np.ndarray, pts_curr: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Computes the per-point epipolar error: |p_t^T * F * p_{t-1}|.

        Args:
            pts_prev (np.ndarray): Points from previous frame of shape (N, 2).
            pts_curr (np.ndarray): Points from current frame of shape (N, 2).
            F (np.ndarray): Fundamental matrix of shape (3, 3).

        Returns:
            np.ndarray: Array of errors for each point pair of shape (N,).
        """
        if len(pts_prev) == 0:
            return np.array([])
            
        pts_prev_homo = np.hstack((pts_prev, np.ones((pts_prev.shape[0], 1))))
        pts_curr_homo = np.hstack((pts_curr, np.ones((pts_curr.shape[0], 1))))

        # Compute |p_t^T * F * p_{t-1}|
        errors = np.abs(np.sum(pts_curr_homo * (F @ pts_prev_homo.T).T, axis=1))
        return errors

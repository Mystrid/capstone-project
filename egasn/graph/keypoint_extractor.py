import cv2
import numpy as np
from typing import Tuple, List

class ORBKeypointExtractor:
    """
    Extracts and matches ORB keypoints between consecutive frames for the EGASN framework.
    """
    def __init__(self, 
                 n_features: int = 2000, 
                 flann_table_number: int = 6, 
                 flann_key_size: int = 12, 
                 flann_multi_probe_level: int = 1, 
                 ratio_test: float = 0.75):
        """
        Initializes the ORB keypoint extractor and FLANN matcher.

        Args:
            n_features (int): Maximum number of features to retain.
            flann_table_number (int): Number of LSH tables for FLANN.
            flann_key_size (int): Size of the key in LSH for FLANN.
            flann_multi_probe_level (int): Multi-probe level for LSH for FLANN.
            ratio_test (float): Lowe's ratio test threshold.
        """
        self.ratio_test = ratio_test
        self.orb = cv2.ORB_create(nfeatures=n_features)
        
        # FLANN parameters for binary descriptors (like ORB)
        index_params = dict(
            algorithm=6,  # FLANN_INDEX_LSH
            table_number=flann_table_number,
            key_size=flann_key_size,
            multi_probe_level=flann_multi_probe_level
        )
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_keypoints(self, frame: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detects and computes ORB keypoints and descriptors.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            Tuple[List[cv2.KeyPoint], np.ndarray]: Keypoints and their descriptors.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_frames(self, desc_prev: np.ndarray, desc_curr: np.ndarray) -> List[cv2.DMatch]:
        """
        Matches descriptors using FLANN matcher and applies Lowe's ratio test.

        Args:
            desc_prev (np.ndarray): Descriptors from the previous frame.
            desc_curr (np.ndarray): Descriptors from the current frame.

        Returns:
            List[cv2.DMatch]: Good matches that pass the ratio test.
        """
        if desc_prev is None or desc_curr is None or len(desc_prev) < 2 or len(desc_curr) < 2:
            return []
            
        matches = self.matcher.knnMatch(desc_prev, desc_curr, k=2)
        good_matches = []
        
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < self.ratio_test * n.distance:
                    good_matches.append(m)
                    
        return good_matches

    def extract_and_match(self, frame_prev: np.ndarray, frame_curr: np.ndarray) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
        """
        Extracts keypoints from both frames and computes good matches.

        Args:
            frame_prev (np.ndarray): Previous image frame.
            frame_curr (np.ndarray): Current image frame.

        Returns:
            Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]: 
                Keypoints of previous frame, keypoints of current frame, and good matches.
        """
        kp_prev, desc_prev = self.extract_keypoints(frame_prev)
        kp_curr, desc_curr = self.extract_keypoints(frame_curr)
        matches = self.match_frames(desc_prev, desc_curr)
        return kp_prev, kp_curr, matches

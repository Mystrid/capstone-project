import pytest
import numpy as np
import torch
import cv2

from egasn.graph.epipolar import EpipolarGraphConstructor, FeatureWarper
from egasn.graph.keypoint_extractor import ORBKeypointExtractor


def test_orb_extraction():
    """Verify ORB extracts keypoints from an image with distinct features."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw shapes to create corners/edges for ORB to detect
    cv2.rectangle(img, (25, 25), (75, 75), (255, 255, 255), -1)
    cv2.rectangle(img, (100, 100), (175, 175), (128, 128, 128), -1)
    cv2.circle(img, (150, 50), 30, (200, 200, 200), -1)

    extractor = ORBKeypointExtractor(n_features=500)
    kp, desc = extractor.extract_keypoints(img)

    assert len(kp) > 0
    assert desc is not None


def test_matching_shifted_images():
    """Two shifted images should produce some keypoint matches."""
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img1, (25, 25), (75, 75), (255, 255, 255), -1)
    cv2.rectangle(img1, (100, 100), (175, 175), (128, 128, 128), -1)

    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img2, (30, 30), (80, 80), (255, 255, 255), -1)
    cv2.rectangle(img2, (105, 105), (180, 180), (128, 128, 128), -1)

    extractor = ORBKeypointExtractor(n_features=500)
    kp_prev, kp_curr, matches = extractor.extract_and_match(img1, img2)

    # Should find at least some matches
    assert len(kp_prev) > 0
    assert len(kp_curr) > 0


def test_flow_interpolation_shape():
    """Flow field output should match expected dimensions."""
    constructor = EpipolarGraphConstructor(output_size=(32, 32))
    img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    flow, valid_mask = constructor.construct(img1, img2)

    assert flow.shape == (2, 32, 32)
    assert valid_mask.shape == (1, 32, 32)


def test_zero_flow_on_identical_frames():
    """Identical frames should produce zero flow."""
    constructor = EpipolarGraphConstructor(output_size=(32, 32))
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    flow, valid_mask = constructor.construct(img, img.copy())

    assert torch.allclose(flow, torch.zeros_like(flow), atol=1e-3)


def test_feature_warper_identity():
    """Zero flow should produce identity warping (output ≈ input)."""
    warper = FeatureWarper()
    features = torch.randn(1, 64, 32, 32)
    flow = torch.zeros(1, 2, 32, 32)

    warped = warper(features, flow)

    assert torch.allclose(features, warped, atol=1e-4)

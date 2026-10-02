import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# Standard Cityscapes colors mapped to 19 train IDs
CITYSCAPES_PALETTE = [
    (128, 64, 128),  # 0: road
    (244, 35, 232),  # 1: sidewalk
    (70, 70, 70),    # 2: building
    (102, 102, 156), # 3: wall
    (190, 153, 153), # 4: fence
    (153, 153, 153), # 5: pole
    (250, 170, 30),  # 6: traffic light
    (220, 220, 0),   # 7: traffic sign
    (107, 142, 35),  # 8: vegetation
    (152, 251, 152), # 9: terrain
    (70, 130, 180),  # 10: sky
    (220, 20, 60),   # 11: person
    (255, 0, 0),     # 12: rider
    (0, 0, 142),     # 13: car
    (0, 0, 70),      # 14: truck
    (0, 60, 100),    # 15: bus
    (0, 80, 100),    # 16: train
    (0, 0, 230),     # 17: motorcycle
    (119, 11, 32)    # 18: bicycle
]

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", 
    "traffic_light", "traffic_sign", "vegetation", "terrain", 
    "sky", "person", "rider", "car", "truck", "bus", "train", 
    "motorcycle", "bicycle"
]

def colorize_mask(mask: np.ndarray, palette: List[Tuple[int, int, int]] = CITYSCAPES_PALETTE) -> np.ndarray:
    """
    Converts a class mask into an RGB image using the specified palette.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for c, color in enumerate(palette):
        color_mask[mask == c] = color
        
    return color_mask

def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a colorized mask on an image.
    """
    color_mask = colorize_mask(mask)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
        
    overlay = (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    return overlay

def plot_uncertainty_heatmap(uncertainty_map: np.ndarray, save_path: Optional[str] = None):
    """
    Plots and optionally saves an uncertainty heatmap.
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(uncertainty_map, cmap='jet')
    plt.colorbar(label='Uncertainty')
    plt.title('Uncertainty Heatmap')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_comparison_grid(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, uncertainty: Optional[np.ndarray] = None, save_path: Optional[str] = None):
    """
    Creates a matplotlib figure with subplots comparing image, ground truth, prediction, and optionally uncertainty.
    """
    num_plots = 4 if uncertainty is not None else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    gt_color = colorize_mask(gt_mask)
    axes[1].imshow(gt_color)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    pred_color = colorize_mask(pred_mask)
    axes[2].imshow(pred_color)
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    
    if uncertainty is not None:
        im = axes[3].imshow(uncertainty, cmap='jet')
        axes[3].set_title("Uncertainty")
        axes[3].axis('off')
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


class Evaluator:
    """Accumulates predictions over batches and computes segmentation metrics.

    Used by training and evaluation scripts to track mIoU, per-class IoU,
    and frequency-weighted IoU across an entire epoch.
    """

    def __init__(self, num_classes: int = 19, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        """Resets the confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def add_batch(self, target: torch.Tensor, pred: torch.Tensor):
        """Adds a batch of predictions to the confusion matrix.

        Args:
            target: Ground truth labels (B, H, W).
            pred: Predicted labels (B, H, W).
        """
        target_np = target.detach().cpu().numpy().flatten()
        pred_np = pred.detach().cpu().numpy().flatten()
        valid = target_np != self.ignore_index
        target_valid = target_np[valid]
        pred_valid = pred_np[valid]
        self.confusion_matrix += np.bincount(
            self.num_classes * target_valid.astype(int) + pred_valid.astype(int),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

    def Intersection_over_Union(self) -> np.ndarray:
        """Returns per-class IoU array."""
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.zeros(self.num_classes)
        valid = denom > 0
        iou[valid] = tp[valid] / denom[valid]
        return iou

    def Mean_Intersection_over_Union(self) -> float:
        """Returns mean IoU across classes with at least one sample."""
        iou = self.Intersection_over_Union()
        valid = (self.confusion_matrix.sum(axis=0) + self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)) > 0
        return float(np.mean(iou[valid])) if np.any(valid) else 0.0

    def Frequency_Weighted_Intersection_over_Union(self) -> float:
        """Returns frequency-weighted IoU."""
        freq = self.confusion_matrix.sum(axis=1) / (self.confusion_matrix.sum() + 1e-10)
        iou = self.Intersection_over_Union()
        return float(np.sum(freq * iou))

def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 19, ignore_index: int = 255) -> Tuple[float, np.ndarray]:
    """
    Computes mean Intersection over Union (mIoU).
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    valid_mask = target_np != ignore_index
    pred_valid = pred_np[valid_mask]
    target_valid = target_np[valid_mask]
    
    # Compute confusion matrix
    confusion_matrix = np.bincount(
        num_classes * target_valid + pred_valid, 
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    
    # Compute IoU
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    
    denominator = tp + fp + fn
    valid_classes = denominator > 0
    
    per_class_iou = np.zeros(num_classes)
    per_class_iou[valid_classes] = tp[valid_classes] / denominator[valid_classes]
    
    mean_iou = np.mean(per_class_iou[valid_classes]) if np.any(valid_classes) else 0.0
    
    return float(mean_iou), per_class_iou

def compute_fwiou(pred_curr: torch.Tensor, pred_prev: torch.Tensor, flow_field: torch.Tensor, num_classes: int = 19) -> float:
    """
    Computes Flow-Warped Intersection over Union (FWIoU) for temporal consistency.
    """
    if pred_prev.dim() == 3:
        pred_prev = pred_prev.unsqueeze(1).float()
    if pred_curr.dim() == 3:
        pred_curr = pred_curr.unsqueeze(1).float()
        
    B, C, H, W = pred_prev.shape
    
    if flow_field.shape[1] == 2:
        # Create base grid
        y, x = torch.meshgrid(torch.arange(H, device=flow_field.device), torch.arange(W, device=flow_field.device), indexing='ij')
        grid = torch.stack((x, y), dim=0).float()  # (2, H, W)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)
        
        # Add flow
        grid = grid + flow_field
        
        # Normalize to [-1, 1]
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / (W - 1) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / (H - 1) - 1.0
        
        grid = grid.permute(0, 2, 3, 1)  # (B, H, W, 2)
    else:
        grid = flow_field
        
    warped_pred_prev = F.grid_sample(pred_prev, grid, mode='nearest', padding_mode='zeros', align_corners=True)
    
    if C > 1:
        pred_curr_cls = torch.argmax(pred_curr, dim=1)
        warped_pred_prev_cls = torch.argmax(warped_pred_prev, dim=1)
    else:
        pred_curr_cls = pred_curr.squeeze(1).long()
        warped_pred_prev_cls = warped_pred_prev.squeeze(1).long()
        
    miou, _ = compute_miou(pred_curr_cls, warped_pred_prev_cls, num_classes=num_classes)
    return miou

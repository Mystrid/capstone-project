import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Callable

class CityscapesVideoDataset(Dataset):
    """
    Cityscapes Video Dataset loader for EGASN framework.
    Loads adjacent frames for temporal consistency and epipolar geometry.
    """
    def __init__(self, root: str, split: str = 'train', transform: Optional[Callable] = None, frame_offset: int = 1):
        """
        Args:
            root: Path to Cityscapes root directory
            split: 'train', 'val', or 'test'
            transform: Joint transforms to apply to frames and labels
            frame_offset: Frame distance between current and previous frame
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.frame_offset = frame_offset
        
        self.img_dir = os.path.join(root, 'leftImg8bit_sequence', split)
        self.gt_dir = os.path.join(root, 'gtFine', split)
        
        self.samples = []
        
        # Create lookup table for label mapping (34 Cityscapes label IDs to 19 train IDs)
        self.mapping = np.full(256, 255, dtype=np.uint8)
        valid_classes = {7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9, 
                         23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18}
        for k, v in valid_classes.items():
            self.mapping[k] = v
            
        self._load_samples()
        
    def _load_samples(self):
        gt_pattern = os.path.join(self.gt_dir, '*', '*_gtFine_labelIds.png')
        gt_files = sorted(glob.glob(gt_pattern))
        
        for gt_path in gt_files:
            # Parse filename: {city}_{seq:06d}_{frame:06d}_gtFine_labelIds.png
            basename = os.path.basename(gt_path)
            parts = basename.split('_')
            city = parts[0]
            seq = parts[1]
            frame_idx = int(parts[2])
            
            curr_frame_str = f"{frame_idx:06d}"
            prev_frame_str = f"{frame_idx - self.frame_offset:06d}"
            
            curr_img_path = os.path.join(self.img_dir, city, f"{city}_{seq}_{curr_frame_str}_leftImg8bit.png")
            prev_img_path = os.path.join(self.img_dir, city, f"{city}_{seq}_{prev_frame_str}_leftImg8bit.png")
            
            if os.path.exists(curr_img_path) and os.path.exists(prev_img_path):
                self.samples.append({
                    'curr_img': curr_img_path,
                    'prev_img': prev_img_path,
                    'label': gt_path
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load as PIL Image for transforms
        img_curr = Image.open(sample['curr_img']).convert('RGB')
        img_prev = Image.open(sample['prev_img']).convert('RGB')
        label_img = Image.open(sample['label'])
        
        # Convert label to train IDs
        label_np = np.array(label_img)
        label_train_id = self.mapping[label_np]
        label = Image.fromarray(label_train_id)
        
        # Keep original numpy arrays for OpenCV processing
        img_curr_np = np.array(img_curr)
        img_prev_np = np.array(img_prev)
        
        if self.transform is not None:
            img_prev, img_curr, label = self.transform(img_prev, img_curr, label)
            
        return {
            'frame_prev': img_prev,
            'frame_curr': img_curr,
            'label': label,
            'frame_prev_np': img_prev_np,
            'frame_curr_np': img_curr_np
        }

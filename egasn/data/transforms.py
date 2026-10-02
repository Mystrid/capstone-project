import random
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from typing import Tuple, List, Callable

class JointCompose:
    """Applies multiple joint transformations sequentially."""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img_prev: Image.Image, img_curr: Image.Image, label: Image.Image):
        for t in self.transforms:
            img_prev, img_curr, label = t(img_prev, img_curr, label)
        return img_prev, img_curr, label

class JointRandomHorizontalFlip:
    """Randomly flips both images and the label horizontally."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img_prev: Image.Image, img_curr: Image.Image, label: Image.Image):
        if random.random() < self.p:
            img_prev = TF.hflip(img_prev)
            img_curr = TF.hflip(img_curr)
            label = TF.hflip(label)
        return img_prev, img_curr, label

class JointRandomCrop:
    """Randomly crops both images and the label at the same location."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img_prev: Image.Image, img_curr: Image.Image, label: Image.Image):
        w, h = img_curr.size
        th, tw = self.size
        if w == tw and h == th:
            return img_prev, img_curr, label
        
        i = random.randint(0, max(0, h - th))
        j = random.randint(0, max(0, w - tw))
        
        img_prev = TF.crop(img_prev, i, j, th, tw)
        img_curr = TF.crop(img_curr, i, j, th, tw)
        label = TF.crop(label, i, j, th, tw)
        
        return img_prev, img_curr, label

class JointColorJitter:
    """Randomly changes brightness, contrast, saturation, and hue of the images."""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        import torchvision.transforms as transforms
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, img_prev: Image.Image, img_curr: Image.Image, label: Image.Image):
        if random.random() < self.p:
            img_prev = self.jitter(img_prev)
            img_curr = self.jitter(img_curr)
        return img_prev, img_curr, label

class JointToTensor:
    """Converts images and label to tensors."""
    def __call__(self, img_prev: Image.Image, img_curr: Image.Image, label: Image.Image):
        img_prev = TF.to_tensor(img_prev)
        img_curr = TF.to_tensor(img_curr)
        label = torch.as_tensor(np.array(label), dtype=torch.long)
        return img_prev, img_curr, label

class JointNormalize:
    """Normalizes the images."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img_prev: torch.Tensor, img_curr: torch.Tensor, label: torch.Tensor):
        img_prev = TF.normalize(img_prev, self.mean, self.std)
        img_curr = TF.normalize(img_curr, self.mean, self.std)
        return img_prev, img_curr, label

def get_train_transforms(crop_size: Tuple[int, int]) -> JointCompose:
    """Factory function for training transforms."""
    return JointCompose([
        JointRandomCrop(crop_size),
        JointRandomHorizontalFlip(),
        JointColorJitter(),
        JointToTensor(),
        JointNormalize()
    ])

def get_val_transforms() -> JointCompose:
    """Factory function for validation transforms."""
    return JointCompose([
        JointToTensor(),
        JointNormalize()
    ])

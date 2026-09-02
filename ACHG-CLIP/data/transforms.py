from PIL import Image
import torch
import numpy as np
from typing import Callable

def get_transforms(dataset_name: str, is_train: bool) -> Callable:
    """
    Returns the preprocessing pipeline for the given dataset using pure PyTorch/PIL 
    to avoid torchvision dependency issues.
    
    IMPLEMENTATION-CHOICE:
    Since the target paper does not specify the exact preprocessing pipeline (resize, crop, normalization),
    and standard CLIP backbones typically require 224x224 input resolution, we adopt the standard 
    CLIP preprocessing pipeline as a functional substitution.
    
    Values:
    - Resize(224) 
    - CenterCrop(224)
    - Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
    """
class CLIPTransform:
    def __init__(self, resolution=224):
        self.resolution = resolution
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        img = img.resize((self.resolution, self.resolution), Image.Resampling.BICUBIC)
        
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            img_arr = np.stack([img_arr]*3, axis=-1)
            
        tensor = torch.from_numpy(np.array(img_arr, copy=True)).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor

def get_transforms(dataset_name: str, is_train: bool) -> Callable:
    """
    Returns the preprocessing pipeline for the given dataset.
    """
    return CLIPTransform(resolution=224)

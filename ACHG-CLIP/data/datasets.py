import os
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder
from typing import Optional, Callable, Tuple, List
from PIL import Image
import numpy as np

class FSCILSubset(Dataset):
    def __init__(self, full_dataset, indices):
        self.subset = Subset(full_dataset, indices)
        self.targets = [full_dataset.targets[i] for i in indices]
        
    def __getitem__(self, idx):
        return self.subset[idx]
        
    def __len__(self):
        return len(self.subset)

class SyntheticFSCILDataset(Dataset):
    """
    Synthetic dataset for testing the FSCIL pipeline when real data is unavailable.
    """
    def __init__(self, num_classes: int, images_per_class: int, resolution: int = 224, transform: Optional[Callable] = None):
        self.num_classes = num_classes
        self.images_per_class = images_per_class
        self.resolution = resolution
        self.transform = transform
        
        # Create synthetic data
        self.total_samples = num_classes * images_per_class
        self.labels = []
        for c in range(num_classes):
            self.labels.extend([c] * images_per_class)
        self.labels = np.array(self.labels)
        
        # We don't store 60k full-resolution images in memory. We'll generate them on the fly.
        
    def __len__(self) -> int:
        return self.total_samples
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = self.labels[idx]
        
        # Fast caching
        if not hasattr(self, '_dummy_img'):
            np.random.seed(0)
            img_data = np.random.randint(0, 256, (self.resolution, self.resolution, 3), dtype=np.uint8)
            self._dummy_img = Image.fromarray(img_data)
            if self.transform is not None:
                self._dummy_img = self.transform(self._dummy_img)
            
        return self._dummy_img, label

class BaseFSCILDataset:
    """
    Base class wrapper for FSCIL datasets to unify the interface.
    """
    def __init__(self, root: str, name: str, train: bool, transform: Optional[Callable] = None, synthetic: bool = False):
        self.root = root
        self.name = name
        self.train = train
        self.transform = transform
        self.synthetic = synthetic
        
        self.dataset = None
        self._load_dataset()
        
    def _load_dataset(self):
        """
        Loads the underlying dataset. If synthetic is True, loads a synthetic fixture.
        """
        raise NotImplementedError

    def get_dataset(self) -> Dataset:
        return self.dataset
        
class CIFAR100Dataset(BaseFSCILDataset):
    def _load_dataset(self):
        if self.synthetic:
            # Synthetic 100 classes, very small to speed up tests
            images_per_class = 5 if self.train else 2
            self.dataset = SyntheticFSCILDataset(num_classes=100, images_per_class=images_per_class, resolution=32, transform=self.transform)
        else:
            from torchvision.datasets import CIFAR100
            self.dataset = CIFAR100(root=self.root, train=self.train, transform=self.transform, download=True)

class MiniImageNetDataset(BaseFSCILDataset):
    def _load_dataset(self):
        if self.synthetic:
            images_per_class = 500 if self.train else 100
            self.dataset = SyntheticFSCILDataset(num_classes=100, images_per_class=images_per_class, resolution=84, transform=self.transform)
        else:
            path = os.path.join(self.root, "miniimagenet")
            if not os.path.exists(path):
                path = self.root
            if os.path.exists(os.path.join(path, "images")) and os.path.isdir(os.path.join(path, "images")):
                path = os.path.join(path, "images")
            if not os.path.exists(path):
                raise FileNotFoundError(f"miniImageNet not found at {path} or {self.root}. Real data validation failed.")
            
            # Use ImageFolder to parse the 100 class directories
            full_dataset = ImageFolder(root=path, transform=self.transform)
            
            # Deterministic split: first 500 images per class for train, rest (100) for test
            # ImageFolder automatically sorts classes alphabetically and files alphabetically.
            train_indices = []
            test_indices = []
            
            class_counts = {i: 0 for i in range(100)}
            for idx, (_, label) in enumerate(full_dataset.samples):
                if class_counts[label] < 500:
                    train_indices.append(idx)
                else:
                    test_indices.append(idx)
                class_counts[label] += 1
                
            if self.train:
                self.dataset = FSCILSubset(full_dataset, train_indices)
            else:
                self.dataset = FSCILSubset(full_dataset, test_indices)
            
class CUB200Dataset(BaseFSCILDataset):
    def _load_dataset(self):
        if self.synthetic:
            images_per_class = 30 if self.train else 20
            self.dataset = SyntheticFSCILDataset(num_classes=200, images_per_class=images_per_class, resolution=224, transform=self.transform)
        else:
            path = os.path.join(self.root, "CUB_200_2011")
            if not os.path.exists(path):
                raise FileNotFoundError(f"CUB-200-2011 not found at {path}. Real data validation failed.")
            
            # The images are in CUB_200_2011/images
            images_path = os.path.join(path, "images")
            full_dataset = ImageFolder(root=images_path, transform=self.transform)
            
            # Read official train/test splits
            split_file = os.path.join(path, "train_test_split.txt")
            image_file = os.path.join(path, "images.txt")
            
            if os.path.exists(split_file) and os.path.exists(image_file):
                # Map image_id -> is_train (1 for train, 0 for test)
                is_train_dict = {}
                with open(split_file, "r") as f:
                    for line in f:
                        img_id, is_tr = line.strip().split()
                        is_train_dict[img_id] = int(is_tr)
                        
                # Map image_id -> image_name (e.g. 001.Black_footed_Albatross/Albatross_0001_29574.jpg)
                # But ImageFolder sorts classes and files differently. We need to match by path.
                # ImageFolder.samples gives (full_path, label)
                
                # We'll just build a set of relative filenames that belong to train
                train_names = set()
                with open(image_file, "r") as f:
                    for line in f:
                        img_id, img_name = line.strip().split()
                        if is_train_dict.get(img_id, 0) == 1:
                            # img_name is like "001.Black_footed_Albatross/Albatross_0001_29574.jpg"
                            # In windows, paths might use backslashes, so keep it normalized
                            train_names.add(os.path.normpath(img_name))
                            
                train_indices = []
                test_indices = []
                for idx, (p, _) in enumerate(full_dataset.samples):
                    # Extract the relative path parts: class_dir/file.jpg
                    rel_path = os.path.normpath(os.path.relpath(p, images_path))
                    if rel_path in train_names:
                        train_indices.append(idx)
                    else:
                        test_indices.append(idx)
            else:
                # Fallback to deterministic 50/50 split if metadata missing
                train_indices = []
                test_indices = []
                class_counts = {i: 0 for i in range(200)}
                for idx, (_, label) in enumerate(full_dataset.samples):
                    if class_counts[label] % 2 == 0:
                        train_indices.append(idx)
                    else:
                        test_indices.append(idx)
                    class_counts[label] += 1
            
            if self.train:
                self.dataset = FSCILSubset(full_dataset, train_indices)
            else:
                self.dataset = FSCILSubset(full_dataset, test_indices)

import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Dict, Optional
import numpy as np
import os

class SessionData:
    """
    Holds the data for a single FSCIL session.
    """
    def __init__(self, session_id: int, classes: List[int], train_loader: DataLoader, test_loader: DataLoader):
        self.session_id = session_id
        self.classes = classes
        self.train_loader = train_loader
        self.test_loader = test_loader

class FSCILDataManager:
    """
    Manages the creation of Base and Incremental sessions.
    
    IMPLEMENTATION-CHOICE:
    Since the exact FSCIL class permutation is unrecoverable, this class provides a
    deterministic pseudo-random permutation based on the provided seed. This conceptually
    matches the CEC/FACT setup.
    """
    def __init__(self, 
                 train_dataset, 
                 test_dataset, 
                 total_classes: int,
                 base_classes: int,
                 incremental_classes: int,
                 classes_per_session: int,
                 shots_per_class: int,
                 base_batch_size: int,
                 incremental_batch_size: int,
                 seed: int = 42):
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.total_classes = total_classes
        self.base_classes = base_classes
        self.incremental_classes = incremental_classes
        self.classes_per_session = classes_per_session
        self.shots_per_class = shots_per_class
        self.base_batch_size = base_batch_size
        self.incremental_batch_size = incremental_batch_size
        self.seed = seed
        
        # Generate the class ordering
        np.random.seed(seed)
        self.class_ordering = np.random.permutation(total_classes).tolist()
        
        # Verify sizes
        assert base_classes + incremental_classes == total_classes
        self.num_sessions = 1 + (incremental_classes // classes_per_session)
        
        # Precompute indices for each class in the dataset to make subsetting faster
        self._train_indices_by_class = self._group_by_class(self.train_dataset)
        self._test_indices_by_class = self._group_by_class(self.test_dataset)

    def _group_by_class(self, dataset) -> Dict[int, List[int]]:
        indices_by_class = {i: [] for i in range(self.total_classes)}
        
        # This is a generic way to extract labels. For torchvision datasets,
        # usually dataset.targets is available, but for synthetic datasets
        # we might have self.labels.
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        elif hasattr(dataset, 'labels'):
            targets = dataset.labels
        elif hasattr(dataset.dataset, 'labels'):
            targets = dataset.dataset.labels
        else:
            raise ValueError("Dataset must have 'targets' or 'labels' attribute for FSCIL grouping.")
            
        for idx, label in enumerate(targets):
            indices_by_class[int(label)].append(idx)
            
        return indices_by_class

    def get_session(self, session_id: int) -> SessionData:
        """
        Returns the SessionData for the given session ID.
        Session 0 is the base session.
        Sessions > 0 are incremental sessions.
        """
        assert 0 <= session_id < self.num_sessions
        
        if session_id == 0:
            session_classes = self.class_ordering[:self.base_classes]
            batch_size = self.base_batch_size
            is_incremental = False
        else:
            start_idx = self.base_classes + (session_id - 1) * self.classes_per_session
            end_idx = start_idx + self.classes_per_session
            session_classes = self.class_ordering[start_idx:end_idx]
            batch_size = self.incremental_batch_size
            is_incremental = True
            
        # Build training subset
        train_indices = []
        for c in session_classes:
            class_indices = self._train_indices_by_class[c]
            if is_incremental:
                # Deterministic pseudo-random few-shot sampling
                # IMPLEMENTATION-CHOICE: Sample exactly `shots_per_class` elements deterministically
                np.random.seed(self.seed + session_id + c)
                sampled_indices = np.random.choice(class_indices, size=self.shots_per_class, replace=False).tolist()
                train_indices.extend(sampled_indices)
            else:
                train_indices.extend(class_indices)
                
        # Build cumulative test subset (contains all classes seen SO FAR, per target paper evaluation rules)
        # "per-session accuracy columns evaluate over all classes seen so far" (PAPER-FACT)
        test_classes = []
        if session_id == 0:
            test_classes = session_classes
        else:
            test_classes = self.class_ordering[:self.base_classes + session_id * self.classes_per_session]
            
        test_indices = []
        for c in test_classes:
            test_indices.extend(self._test_indices_by_class[c])
            
        train_subset = Subset(self.train_dataset, train_indices)
        test_subset = Subset(self.test_dataset, test_indices)
        
        # Use num_workers=2 on Linux/Kaggle to prevent PyTorch worker IPC deadlocks
        num_workers = 4 if os.name == 'nt' else 2
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=False)
        
        return SessionData(
            session_id=session_id,
            classes=session_classes,
            train_loader=train_loader,
            test_loader=test_loader
        )

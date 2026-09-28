import unittest
import torch
import yaml
import os
from data.registry import get_data_manager

class TestDatasetPipeline(unittest.TestCase):
    def setUp(self):
        # Create a dummy config for CIFAR-100 style
        self.config = {
            "dataset_name": {"value": "CIFAR-100"},
            "total_classes": {"value": 100},
            "base_classes": {"value": 60},
            "incremental_classes_total": {"value": 40},
            "classes_per_incremental_session": {"value": 5},
            "shots_per_class": {"value": 5},
            "base_batch_size": {"value": 4},
            "incremental_batch_size": {"value": 4},
            "seed": 42
        }
        self.data_root = "./data_dummy"
        
    def test_manager_initialization(self):
        manager = get_data_manager(self.config, self.data_root, synthetic=True)
        self.assertIsNotNone(manager)
        self.assertEqual(manager.num_sessions, 9) # Base (1) + 8 incremental

    def test_base_session_creation(self):
        manager = get_data_manager(self.config, self.data_root, synthetic=True)
        session_0 = manager.get_session(0)
        
        self.assertEqual(session_0.session_id, 0)
        self.assertEqual(len(session_0.classes), 60)
        
        # Base batch size is 4
        self.assertEqual(session_0.train_loader.batch_size, 4)
        
        # Verify train/test split works
        x, y = next(iter(session_0.train_loader))
        self.assertEqual(x.shape[0], 4) # batch size
        self.assertEqual(x.shape[1], 3) # channels
        self.assertEqual(x.shape[2], 224) # implementation choice resize
        self.assertEqual(x.shape[3], 224)

    def test_incremental_session_creation(self):
        manager = get_data_manager(self.config, self.data_root, synthetic=True)
        session_1 = manager.get_session(1)
        
        self.assertEqual(session_1.session_id, 1)
        self.assertEqual(len(session_1.classes), 5)
        
        # Verify exactly 5 shots per class = 25 total samples in train loader
        total_train_samples = len(session_1.train_loader.dataset)
        self.assertEqual(total_train_samples, 25)
        
    def test_cumulative_test_set(self):
        manager = get_data_manager(self.config, self.data_root, synthetic=True)
        session_1 = manager.get_session(1)
        
        # Test set for session 1 should have classes from session 0 (60) + session 1 (5) = 65 classes
        # 65 classes * 2 test images per class = 130 samples
        total_test_samples = len(session_1.test_loader.dataset)
        self.assertEqual(total_test_samples, 130)

    def test_deterministic_ordering(self):
        manager1 = get_data_manager(self.config, self.data_root, synthetic=True)
        
        # Use same seed
        config2 = self.config.copy()
        config2["seed"] = 42
        manager2 = get_data_manager(config2, self.data_root, synthetic=True)
        
        self.assertEqual(manager1.class_ordering, manager2.class_ordering)
        
        # Test different seed
        config3 = self.config.copy()
        config3["seed"] = 99
        manager3 = get_data_manager(config3, self.data_root, synthetic=True)
        
        self.assertNotEqual(manager1.class_ordering, manager3.class_ordering)
        
    def test_invalid_configurations(self):
        bad_config = self.config.copy()
        bad_config["dataset_name"]["value"] = "UNKNOWN_DATASET"
        
        with self.assertRaises(ValueError):
            get_data_manager(bad_config, self.data_root, synthetic=True)

if __name__ == '__main__':
    unittest.main()

import unittest
import torch
import os
import json
import tempfile
from torch.utils.data import DataLoader, TensorDataset
from evaluation.metrics import calculate_accuracy
from evaluation.evaluator import FSCILEvaluator
from evaluation.session_evaluator import FSCILSessionEvaluator
from evaluation.result_writer import ResultWriter
from data.registry import get_data_manager

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 224 * 224, 100)
        # Fix weights so logits are predictable for a test
        with torch.no_grad():
            self.linear.weight.fill_(0.1)
            self.linear.bias.fill_(0.0)
            # Make class 2 always win if input is positive
            self.linear.weight[2, :] = 1.0

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))

class TestEvaluationPipeline(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = DummyModel().to(self.device)
        self.config = {
            "dataset_name": {"value": "CIFAR-100"},
            "total_classes": {"value": 100},
            "base_classes": {"value": 60},
            "incremental_classes_total": {"value": 40},
            "classes_per_incremental_session": {"value": 5},
            "shots_per_class": {"value": 5},
            "base_batch_size": {"value": 4},
            "incremental_batch_size": {"value": 4},
            "synthetic_samples_per_class": {"value": 5},
            "seed": 42
        }
        self.data_root = "./data_dummy"
        
    def test_calculate_accuracy(self):
        # 3 samples, 4 classes
        logits = torch.tensor([
            [1.0, 0.5, 0.2, 0.1], # pred: 0
            [0.1, 2.0, 0.5, 0.1], # pred: 1
            [0.1, 0.1, 3.0, 0.1]  # pred: 2
        ])
        targets = torch.tensor([0, 2, 2]) # 0 matches, 1 fails, 2 matches -> 2/3
        
        acc = calculate_accuracy(logits, targets)
        self.assertAlmostEqual(acc, 2.0/3.0)

    def test_evaluator_no_grad_and_eval_mode(self):
        # Ensure model is in train mode initially
        self.model.train()
        
        # Create a dummy dataloader
        x = torch.randn(10, 3, 224, 224)
        y = torch.zeros(10, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        
        evaluator = FSCILEvaluator(self.model, self.device)
        metrics = evaluator.evaluate(loader)
        
        # Model should be in eval mode
        self.assertFalse(self.model.training)
        
        # Accuracy should be computable
        self.assertEqual(metrics["samples"], 10)

    def test_session_evaluator_counts_and_classes_seen(self):
        manager = get_data_manager(self.config, self.data_root, synthetic=True)
        session_eval = FSCILSessionEvaluator(self.model, self.device, manager)
        
        # Base session (0) -> 60 classes seen
        res_0 = session_eval.evaluate_session(0)
        self.assertEqual(res_0["classes_seen"], 60)
        self.assertEqual(res_0["session_id"], 0)
        
        # Session 1 -> 65 classes seen
        res_1 = session_eval.evaluate_session(1)
        self.assertEqual(res_1["classes_seen"], 65)
        self.assertEqual(res_1["session_id"], 1)
        
        # Ensure evaluate_all_sessions returns 9 results for CIFAR
        all_res = session_eval.evaluate_all_sessions()
        self.assertEqual(len(all_res), 9)

    def test_result_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ResultWriter(tmpdir)
            
            mock_results = [
                {"session_id": 0, "accuracy": 0.8, "samples": 6000, "classes_seen": 60},
                {"session_id": 1, "accuracy": 0.7, "samples": 6500, "classes_seen": 65}
            ]
            
            file_path = writer.write("CIFAR-100", "test_run_1", self.config, mock_results)
            
            self.assertTrue(os.path.exists(file_path))
            
            with open(file_path, "r") as f:
                data = json.load(f)
                
            self.assertEqual(data["dataset"], "CIFAR-100")
            self.assertEqual(data["run_id"], "test_run_1")
            self.assertEqual(data["seed"], 42)
            self.assertEqual(len(data["results"]), 2)
            self.assertIn("accuracy", data["provenance"])

    def test_checkpoint_callback_during_eval(self):
        manager = get_data_manager(self.config, self.data_root, synthetic=True)
        session_eval = FSCILSessionEvaluator(self.model, self.device, manager)
        
        loaded_sessions = []
        def mock_load(session_id):
            loaded_sessions.append(session_id)
            
        session_eval.evaluate_all_sessions(load_checkpoint_fn=mock_load)
        
        # Ensure the load callback was triggered for all 9 sessions in order
        self.assertEqual(loaded_sessions, list(range(9)))

if __name__ == '__main__':
    unittest.main()

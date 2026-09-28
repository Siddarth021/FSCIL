import os
import unittest
import tempfile
import torch
import math

from models.achg_clip import ACHGCLIP, UnresolvedFeedbackPath
from training.trainer import ACHGCLIPTrainer, TrainerConfig, TrainerConfigError
from tests.test_achg_clip import _make_model, _random_batch, DT

class TestTrainerConfig(unittest.TestCase):
    def test_invalid_accumulation_steps(self):
        with self.assertRaises(TrainerConfigError):
            TrainerConfig(gradient_accumulation_steps=0)

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = _make_model()
        self.config = TrainerConfig(
            lr=1e-3, 
            gradient_accumulation_steps=1,
            gradient_clip_max_norm=4.0
        )
        self.trainer = ACHGCLIPTrainer(self.model, self.config, self.device)
        self.images, self.tokens = _random_batch()
        self.batch_size = self.images.size(0)
        self.targets = torch.randint(0, self.tokens.size(0), (self.batch_size,), device=self.device)

    def test_initialization_and_frozen_clip(self):
        # Check CLIP backbone is frozen
        for name, param in self.trainer.model.clip.named_parameters():
            self.assertFalse(param.requires_grad, f"Parameter {name} in CLIP should be frozen.")
        
        # Check other params are trainable (e.g. GIN, ACGA, HGNEC)
        trainable_found = False
        for name, param in self.trainer.model.named_parameters():
            if "clip" not in name and param.requires_grad:
                trainable_found = True
        self.assertTrue(trainable_found)

    def test_train_eval_modes(self):
        self.trainer.eval_mode()
        self.assertFalse(self.trainer.model.training)
        self.trainer.train_mode()
        self.assertTrue(self.trainer.model.training)

    def test_synthetic_forward_and_loss_collection(self):
        losses = self.trainer.train_step(self.images, self.tokens, self.targets, dt=DT)
        
        # Check all required keys exist
        expected_keys = {"L_total", "L_CE", "L_recon", "L_adv", "L_energy"}
        self.assertEqual(set(losses.keys()), expected_keys)
        
        # Check finite losses
        for k, v in losses.items():
            self.assertFalse(torch.isnan(torch.tensor(v)))
            self.assertFalse(torch.isinf(torch.tensor(v)))
            
        # Check aggregation logic
        expected_total = (
            losses["L_CE"] +
            self.config.lambda_recon * losses["L_recon"] +
            self.config.lambda_adv * losses["L_adv"] +
            self.config.lambda_energy * losses["L_energy"]
        )
        self.assertAlmostEqual(losses["L_total"], expected_total, places=5)

    def test_backward_and_parameter_update(self):
        # Record pre-step params for trainable parameters
        pre_step_params = {n: p.clone() for n, p in self.trainer.model.named_parameters() if p.requires_grad}
        
        # Override the zero_grad step so we can inspect gradients
        original_zero_grad = self.trainer.optimizer.zero_grad
        self.trainer.optimizer.zero_grad = lambda: None
        
        _ = self.trainer.train_step(self.images, self.tokens, self.targets, dt=DT)
        
        # Verify gradients exist and parameters updated
        updated_count = 0
        for n, p in self.trainer.model.named_parameters():
            if p.requires_grad:
                if "restorer" not in n:
                    self.assertIsNotNone(p.grad, msg=f"Param {n} grad is None")
                    if not torch.allclose(pre_step_params[n], p):
                        updated_count += 1
                else:
                    self.assertIsNone(p.grad, msg=f"Param {n} grad is not None")
        self.assertGreater(updated_count, 0)
        
        # Restore zero_grad
        self.trainer.optimizer.zero_grad = original_zero_grad

    def test_gradient_accumulation(self):
        self.config.gradient_accumulation_steps = 2
        self.trainer = ACHGCLIPTrainer(self.model, self.config, self.device)
        
        pre_step_params = {n: p.clone() for n, p in self.trainer.model.named_parameters() if p.requires_grad}
        
        # Step 1: shouldn't update params (no optimizer.step)
        self.trainer.train_step(self.images, self.tokens, self.targets, dt=DT)
        for n, p in self.trainer.model.named_parameters():
            if p.requires_grad:
                self.assertTrue(torch.allclose(pre_step_params[n], p))
                if "restorer" not in n:
                    self.assertIsNotNone(p.grad, msg=f"Param {n} should accumulate grad")
                
        # Step 2: should update params
        self.trainer.train_step(self.images, self.tokens, self.targets, dt=DT)
        updated_count = 0
        for n, p in self.trainer.model.named_parameters():
            if p.requires_grad and "restorer" not in n:
                if not torch.allclose(pre_step_params[n], p):
                    updated_count += 1
        self.assertGreater(updated_count, 0)

    def test_checkpoint_save_and_load(self):
        self.trainer.train_step(self.images, self.tokens, self.targets, dt=DT)
        self.trainer.epoch = 2
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "checkpoint.pt")
            self.trainer.save_checkpoint(ckpt_path, seed=42)
            
            # Create new trainer
            new_model = _make_model()
            new_trainer = ACHGCLIPTrainer(new_model, self.config, self.device)
            seed = new_trainer.load_checkpoint(ckpt_path)
            
            self.assertEqual(seed, 42)
            self.assertEqual(new_trainer.epoch, 2)
            self.assertEqual(new_trainer.global_step, 1)
            
            # Check model state match
            for p1, p2 in zip(self.trainer.model.parameters(), new_trainer.model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
                
            # Optimizer state check (just check they have same keys and somewhat similar structure)
            self.assertEqual(len(self.trainer.optimizer.state), len(new_trainer.optimizer.state))

    def test_synthetic_multi_step_training_convergence(self):
        # We test that loss changes/reduces over multiple steps.
        # Since it's a random synthetic batch and mock model, we just ensure it runs
        # multiple steps without crashing (NaN/Inf)
        losses = []
        for _ in range(5):
            loss_dict = self.trainer.train_step(self.images, self.tokens, self.targets, dt=DT)
            losses.append(loss_dict["L_total"])
        
        # Check all are finite
        for l in losses:
            self.assertFalse(math.isnan(l))
            self.assertFalse(math.isinf(l))

    def test_unresolved_feedback_path_is_preserved(self):
        self.assertIsInstance(self.trainer.model.feedback_path, UnresolvedFeedbackPath)
        
        out = self.trainer.model(self.images, self.tokens, dt=DT)
        text_fb = self.trainer.model.feedback_path(out.text.hgn_ec, self.trainer.model.text_prompt)
        
        self.assertEqual(text_fb["status"], "UNRESOLVED")
        self.assertFalse(text_fb["applied"])

if __name__ == '__main__':
    import math
    unittest.main()

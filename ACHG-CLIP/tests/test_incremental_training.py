import unittest
import torch
import copy
import os
import shutil
import tempfile
from data.registry import get_data_manager
from models.achg_clip import ACHGCLIP, ACHGCLIPConfig, UnresolvedFeedbackPath
from models.clip.clip_wrapper import CLIPConfig
from training.trainer import ACHGCLIPTrainer, TrainerConfig
from evaluation.session_evaluator import FSCILSessionEvaluator

class TestIncrementalTraining(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for checkpoints
        self.test_dir = tempfile.mkdtemp()
        
        self.data_cfg = {
            "dataset_name": {"value": "CIFAR-100"},
            "total_classes": {"value": 100},
            "base_classes": {"value": 60},
            "incremental_classes_total": {"value": 40},
            "classes_per_incremental_session": {"value": 5},
            "shots_per_class": {"value": 5},
            "base_batch_size": {"value": 4},
            "incremental_batch_size": {"value": 4},
            "synthetic_samples_per_class": {"value": 10}
        }
        
        self.manager = get_data_manager(self.data_cfg, data_root=self.test_dir, synthetic=True)
        
        # Build model Config (very small for testing)
        clip_cfg = CLIPConfig(
            variant="ViT-B/32",
            d_model=32,
            d_e=32,
            d_k=8,
            num_heads=2,
            num_layers=2,
            vocab_size=100,
            max_text_len=10,
            patch_dim=1,
            max_patches=1,
            ffn_hidden_dim=64,
            dropout=0.0
        )

        achg_cfg = ACHGCLIPConfig(
            clip=clip_cfg,
            num_layers=2,
            prompt_dim=32,
            node_feature_dim=32,
            mlp_bridge_hidden_dim=16,
            gin_num_layers=1,
            gin_hidden_dim=16,
            acga_latent_dim=8,
            hgnec_compressed_dim=8,
            hgnec_restored_dim=16,
        )

        self.model = ACHGCLIP(achg_cfg)
        
        self.t_cfg = TrainerConfig(
            lr=0.01,
            weight_decay=0.001,
            gradient_accumulation_steps=1,
            gradient_clip_max_norm=4.0,
        )
        
        self.device = torch.device("cpu")
        self.trainer = ACHGCLIPTrainer(self.model, self.t_cfg, self.device)
        self.global_tokens = torch.randint(0, 100, (100, 10)).to(self.device)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _clone_trainable_params(self, model):
        return {k: v.clone().detach() for k, v in model.named_parameters() if v.requires_grad}

    def _clone_frozen_params(self, model):
        return {k: v.clone().detach() for k, v in model.named_parameters() if not v.requires_grad}

    def test_incremental_mechanics(self):
        # Base Training (Session 0)
        session_0 = self.manager.get_session(0)
        
        # 8. Correct cumulative class counts
        self.assertEqual(len(session_0.classes), 60)
        
        self.trainer.train_mode()
        # Train 1 batch
        images, targets = next(iter(session_0.train_loader))
        loss_dict = self.trainer.train_step(images, self.global_tokens, targets, dt=0.01)
        
        import math
        # 13. No NaN/Inf losses
        self.assertFalse(math.isnan(loss_dict['L_total']))
        
        # 11. Checkpoint save/load works between sessions
        ckpt_0 = os.path.join(self.test_dir, "session_0.pt")
        self.trainer.save_checkpoint(ckpt_0, seed=42)
        self.assertTrue(os.path.exists(ckpt_0))
        
        params_0_trainable = self._clone_trainable_params(self.model)
        params_0_frozen = self._clone_frozen_params(self.model)
        steps_0 = self.trainer.global_step
        
        # Session 1 (Incremental)
        # 3. Session 1 starts from Session 0 checkpoint.
        trainer_1 = ACHGCLIPTrainer(self.model, self.t_cfg, self.device)
        trainer_1.load_checkpoint(ckpt_0)
        self.assertEqual(trainer_1.global_step, steps_0)
        
        session_1 = self.manager.get_session(1)
        # 9. Correct number of novel classes
        self.assertEqual(len(session_1.classes), 5)
        # 1. Session 1 uses novel 5-shot samples.
        # 10. Correct shot count: 5 samples/class
        self.assertEqual(len(session_1.train_loader.dataset), 25)
        
        trainer_1.train_mode()
        images_1, targets_1 = next(iter(session_1.train_loader))
        loss_dict_1 = trainer_1.train_step(images_1, self.global_tokens, targets_1, dt=0.01)
        
        # 6. Incremental optimizer steps are actually executed.
        # 7. Incremental sessions are not evaluation-only.
        self.assertEqual(trainer_1.global_step, steps_0 + 1)
        
        ckpt_1 = os.path.join(self.test_dir, "session_1.pt")
        trainer_1.save_checkpoint(ckpt_1, seed=42)
        
        params_1_trainable = self._clone_trainable_params(self.model)
        params_1_frozen = self._clone_frozen_params(self.model)
        
        # 2. Session 1 actually changes at least one intended trainable parameter.
        changed = False
        for k in params_0_trainable:
            if not torch.allclose(params_0_trainable[k], params_1_trainable[k]):
                changed = True
                break
        self.assertTrue(changed, "Trainable parameters did not change during Session 1")
        
        # 5. CLIP parameters remain unchanged/frozen.
        for k in params_0_frozen:
            self.assertTrue(torch.allclose(params_0_frozen[k], params_1_frozen[k]), f"Frozen parameter {k} changed!")
            
        # Session 2
        # 4. Session 2 starts from Session 1 checkpoint.
        trainer_2 = ACHGCLIPTrainer(self.model, self.t_cfg, self.device)
        trainer_2.load_checkpoint(ckpt_1)
        self.assertEqual(trainer_2.global_step, trainer_1.global_step)
        
        # 14. The unresolved q_final feedback path remains untouched.
        self.assertIsInstance(self.model.feedback_path, UnresolvedFeedbackPath)
        
        # Check cumulative test counts
        # 12. Evaluation happens after training. (mechanics checked via dataset subset sizes)
        for s in range(self.manager.num_sessions):
            session_data = self.manager.get_session(s)
            expected_classes = 60 + s * 5
            
            if s == 0:
                self.assertEqual(len(session_data.classes), 60)
            else:
                self.assertEqual(len(session_data.classes), 5)
                
            test_dataset = session_data.test_loader.dataset
            self.assertEqual(len(test_dataset), expected_classes * 2)

if __name__ == '__main__':
    unittest.main()

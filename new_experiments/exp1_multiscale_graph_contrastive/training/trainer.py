import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from models.achg_clip import ACHGCLIP, ACHGCLIPOutput
from training.optim import Lion
from training.scheduler import CosineAnnealingWarmupRestarts


class TrainerConfigError(Exception):
    pass


@dataclass
class TrainerConfig:
    lr: float = 0.000325
    weight_decay: float = 1e-3
    warmup_steps: Optional[int] = None
    restart_period: Optional[int] = None
    gradient_accumulation_steps: int = 3
    gradient_clip_max_norm: float = 4.0
    lambda_recon: float = 0.04
    lambda_adv: float = 0.04
    lambda_energy: float = 0.04
    
    def __post_init__(self):
        if self.gradient_accumulation_steps <= 0:
            raise TrainerConfigError("gradient_accumulation_steps must be > 0")


class ACHGCLIPTrainer:
    """
    Stage 9 Trainer for ACHG-CLIP.
    Responsible for:
    - Optimizer (Lion) and Scheduler (CosineAnnealingWarmupRestarts)
    - Forward pass, Loss collection and aggregation
    - Backward pass with gradient accumulation and clipping
    - Parameter freezing (CLIP backbone)
    - Checkpoint save/load
    """
    def __init__(self, model: ACHGCLIP, config: TrainerConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.global_step = 0
        self.epoch = 0
        
        self._freeze_clip_backbone()
        self._setup_optimizer_and_scheduler()

    def _freeze_clip_backbone(self):
        """Freezes the CLIP backbone parameters as required by the paper."""
        for param in self.model.clip.parameters():
            param.requires_grad = False
            
    def _setup_optimizer_and_scheduler(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Lion(trainable_params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer, 
            warmup_steps=self.config.warmup_steps, 
            restart_period=self.config.restart_period,
            max_lr=self.config.lr
        )

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def _compute_ce_loss(self, image_features, text_features, targets, seen_classes_mask=None):
        """
        Global Textual Contrastive Regularization (GTCR) Cross-Entropy.
        Contrasts visual features against ALL text features (or all seen text features).
        """
        logit_scale = self.model.clip.mock_logit_scale.exp() if hasattr(self.model.clip, 'mock_logit_scale') else 100.0
        
        # normalized features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # logits shape: (batch_size, num_text_tokens)
        logits = logit_scale * image_features @ text_features.T
        
        if seen_classes_mask is not None:
            # Mask out unseen classes by setting their logits to -inf
            unseen_mask = ~seen_classes_mask
            logits[:, unseen_mask] = -1e9
            
        # Global Cross Entropy (Image-to-Text only, since Text-to-Image requires unique matching)
        loss_i2t = F.cross_entropy(logits, targets)
        return loss_i2t

    def train_step(self, images: torch.Tensor, text_tokens: torch.Tensor, targets: torch.Tensor, dt: float, seen_classes_mask: Optional[torch.Tensor] = None, base_text_prompts: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Performs a single training step.
        Supports gradient accumulation.
        """
        images = images.to(self.device)
        text_tokens = text_tokens.to(self.device)
        targets = targets.to(self.device)
        
        # GTCR: Do NOT filter text_tokens! We pass ALL text tokens to generate prompts globally.
        batch_text_tokens = text_tokens
        
        # Forward pass
        out: ACHGCLIPOutput = self.model(images, batch_text_tokens, dt=dt)
        
        # Losses
        l_ce = self._compute_ce_loss(out.h_vision, out.h_text, targets, seen_classes_mask)
        
        # Base Prompt Distillation (KD)
        l_distill = torch.tensor(0.0, device=self.device)
        if base_text_prompts is not None and out.text.feedback["q_reshaped"] is not None:
            # We want to penalize changes to the generated text prompts for the base classes (assumed to be the first N classes)
            num_base = base_text_prompts.size(0)
            current_base_prompts = out.text.feedback["q_reshaped"][:num_base]
            l_distill = F.mse_loss(current_base_prompts, base_text_prompts)
        
        # Sum ACGA and Energy losses from both modalities
        l_recon = out.text.acga.reconstruction_loss + out.vision.acga.reconstruction_loss
        l_adv = out.text.acga.adversarial_loss + out.vision.acga.adversarial_loss
        l_energy = out.text.hgn_ec.energy_loss + out.vision.hgn_ec.energy_loss
        
        # Total loss aggregation
        # With GRL, minimizing L_adv correctly trains both Generator and Discriminator
        # KD weight (e.g. lambda_distill = 10.0)
        lambda_distill = getattr(self.config, 'lambda_distill', 10.0)
        total_loss = (
            l_ce + 
            self.config.lambda_recon * l_recon + 
            self.config.lambda_adv * l_adv + 
            self.config.lambda_energy * l_energy +
            lambda_distill * l_distill
        )
        
        # Normalize loss for gradient accumulation
        loss_for_backward = total_loss / self.config.gradient_accumulation_steps
        
        # Single backward pass!
        loss_for_backward.backward()
        
        self.global_step += 1
        
        # Step optimizer
        if self.global_step % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_max_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
        return {
            "L_total": total_loss.item(),
            "L_CE": l_ce.item(),
            "L_recon": l_recon.item(),
            "L_adv": l_adv.item(),
            "L_energy": l_energy.item(),
            "L_distill": l_distill.item() if isinstance(l_distill, torch.Tensor) else l_distill,
        }

    def save_checkpoint(self, path: str, seed: int):
        # Filter out the massive CLIP backbone to prevent disk space exhaustion
        model_state_dict = {k: v for k, v in self.model.state_dict().items() if not k.startswith("clip.")}
        state = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "seed": seed,
            "config": self.config
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"], strict=False)
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.epoch = state.get("epoch", 0)
        self.global_step = state.get("global_step", 0)
        # Note: Do not overwrite self.config completely as it may have different structure, 
        # but the provenance demands we keep track of it.
        return state.get("seed", 42)

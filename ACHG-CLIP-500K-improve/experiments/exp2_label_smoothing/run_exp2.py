"""
EXP2 — Label Smoothing + Learnable Temperature  [KAGGLE]
=========================================================
Hypothesis: Hard-label CE is overconfident and sensitive to noise.
Label smoothing with epsilon=0.1 regularizes class boundaries.
A learnable logit scale (initialized at log(100)) lets the model tune
the sharpness of its own prediction distribution.

Changes vs baseline:
  - CE loss uses label_smoothing=0.1
  - logit_scale is a trainable nn.Parameter (not hardcoded 100.0)
  - warmup_steps=500, restart_period=7500 (3 epochs)
  - base_epochs = 3 (paper value)
"""
import os, sys, yaml, time, json, torch, torch.nn as nn, argparse, shutil, math
import torch.nn.functional as F
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.registry import get_data_manager
from models.achg_clip import ACHGCLIP, ACHGCLIPConfig
from models.clip.clip_wrapper import CLIPConfig
from training.trainer import ACHGCLIPTrainer, TrainerConfig
from training.optim import Lion
from training.scheduler import CosineAnnealingWarmupRestarts
from evaluation.session_evaluator import FSCILSessionEvaluator
import transformers

# ─── EXP2 OVERRIDES ────────────────────────────────────────────────────────────
EXP_NAME          = "exp2_label_smoothing"
BASE_EPOCHS       = 3
WARMUP_STEPS      = 500
RESTART_PERIOD    = 7500
LR                = 0.000325
WEIGHT_DECAY      = 0.001
GRAD_ACCUM        = 3
GRAD_CLIP         = 4.0
LABEL_SMOOTHING   = 0.1
LOGIT_SCALE_INIT  = math.log(100.0)   # learnable, initialized at log(100)
SEED              = 123
# ────────────────────────────────────────────────────────────────────────────────

MAX_TEXT_LEN = 77
VOCAB_SIZE   = 49408


class LoggerTee:
    def __init__(self, fn):
        self.terminal = sys.stdout
        self.log = open(fn, "a", encoding="utf-8")
    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()
    def isatty(self): return hasattr(self.terminal, 'isatty') and self.terminal.isatty()


class SimplePatchifier(nn.Module):
    def forward(self, images): return images


class LearnableScaleTrainer(ACHGCLIPTrainer):
    """
    Extends ACHGCLIPTrainer to add:
      1. A learnable logit_scale parameter
      2. Label-smoothed CE loss
    """
    def __init__(self, model, config, device, label_smoothing=0.1, logit_scale_init=math.log(100.0)):
        super().__init__(model, config, device)
        self.label_smoothing = label_smoothing
        # Learnable logit scale — added to the optimizer
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init, dtype=torch.float32))
        self.logit_scale = self.logit_scale.to(device)
        # Re-build optimizer including logit_scale
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        trainable.append(self.logit_scale)
        self.optimizer = Lion(trainable, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer, warmup_steps=config.warmup_steps,
            restart_period=config.restart_period, max_lr=config.lr)

    def _compute_ce_loss(self, image_features, text_features, targets, seen_classes_mask=None):
        scale = self.logit_scale.exp().clamp(1.0, 200.0)   # clamp for stability
        image_features = F.normalize(image_features, dim=-1)
        text_features  = F.normalize(text_features,  dim=-1)
        logits = scale * image_features @ text_features.T
        if seen_classes_mask is not None:
            logits[:, ~seen_classes_mask] = -1e9
        # Label-smoothed cross entropy
        loss_i2t = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        return loss_i2t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--variant",   type=str, default="ViT-L/14")
    args = parser.parse_args()

    results_base = os.environ.get("ACHG_RESULTS_DIR", "results")
    run_dir = os.path.join(results_base, f"{EXP_NAME}_{datetime.now().strftime('%d%m%Y_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    sys.stdout = LoggerTee(os.path.join(run_dir, "train_log.txt"))
    sys.stderr = sys.stdout
    print(f"[EXP2] Label Smoothing + Learnable Temperature | {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_root = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
    with open(os.path.join(cfg_root, "data/cifar100.yaml"))  as f: data_cfg = yaml.safe_load(f)
    with open(os.path.join(cfg_root, "training.yaml"))       as f: train_cfg = yaml.safe_load(f)
    cfg_dest = os.path.join(run_dir, "configs"); os.makedirs(cfg_dest, exist_ok=True)
    for p in ["data/cifar100.yaml", "training.yaml", "model/clip_backbone.yaml"]:
        src = os.path.join(cfg_root, p)
        if os.path.exists(src): shutil.copy(src, cfg_dest)
    data_cfg["seed"] = SEED

    variant = args.variant
    data_root = args.data_root
    if not os.path.exists(data_root):
        for c in ["D:/FSCIL/datasets", "f:/FSCIL/datasets", "/kaggle/input/fscil-datasets"]:
            if os.path.exists(c): data_root = c; break
        else:
            os.makedirs(data_root, exist_ok=True)

    manager = get_data_manager(data_cfg, data_root=data_root, synthetic=False)
    print(f"Base classes: {manager.base_classes} | Sessions: {manager.num_sessions}")

    hf_map = {"ViT-B/32": "openai/clip-vit-base-patch32",
               "ViT-B/16": "openai/clip-vit-base-patch16",
               "ViT-L/14": "openai/clip-vit-large-patch14"}
    processor = transformers.CLIPProcessor.from_pretrained(hf_map[variant])

    if hasattr(manager.train_dataset, 'dataset') and hasattr(manager.train_dataset.dataset, 'classes'):
        class_names = manager.train_dataset.dataset.classes
    elif hasattr(manager.train_dataset, 'classes'):
        class_names = manager.train_dataset.classes
    else:
        class_names = [f"class {i}" for i in range(100)]

    text_prompts = [f"a photo of a {n}" for n in class_names]
    text_inputs  = processor(text=text_prompts, return_tensors="pt",
                             padding="max_length", max_length=77, truncation=True)
    global_tokens = text_inputs.input_ids.to(device)

    d_model=1024; d_e=768; num_heads=16; num_layers=24; ffn=4096
    clip_cfg = CLIPConfig(variant=variant, d_model=d_model, d_e=d_e, d_k=64,
                          num_heads=num_heads, num_layers=num_layers,
                          vocab_size=VOCAB_SIZE, max_text_len=MAX_TEXT_LEN,
                          patch_dim=1, max_patches=1, ffn_hidden_dim=ffn, dropout=0.0)
    achg_cfg = ACHGCLIPConfig(clip=clip_cfg, num_layers=num_layers,
                               prompt_dim=d_model, node_feature_dim=d_model,
                               mlp_bridge_hidden_dim=128, gin_num_layers=4,
                               gin_hidden_dim=16, acga_latent_dim=64,
                               hgnec_compressed_dim=8, hgnec_restored_dim=16)
    print("Building model...")
    model = ACHGCLIP(achg_cfg)
    print(f"Total trainable parameters: {model.num_trainable_parameters():,}")

    t_cfg = TrainerConfig(lr=LR, weight_decay=WEIGHT_DECAY,
                          warmup_steps=WARMUP_STEPS, restart_period=RESTART_PERIOD,
                          gradient_accumulation_steps=GRAD_ACCUM,
                          gradient_clip_max_norm=GRAD_CLIP)

    # ─── EXP2 KEY CHANGE: use our learnable-scale trainer ────────────────────────
    trainer = LearnableScaleTrainer(model, t_cfg, device,
                                    label_smoothing=LABEL_SMOOTHING,
                                    logit_scale_init=LOGIT_SCALE_INIT)
    print(f"[EXP2] label_smoothing={LABEL_SMOOTHING} | logit_scale_init=log({math.exp(LOGIT_SCALE_INIT):.0f})")
    # ─────────────────────────────────────────────────────────────────────────────

    patchifier = SimplePatchifier().to(device)

    class WrapperModel(nn.Module):
        def __init__(self, core, patch, scale):
            super().__init__()
            self.core_model = core; self.patchifier = patch; self.scale = scale
        def forward(self, images, labels=None):
            patches = self.patchifier(images)
            tokens  = global_tokens[torch.arange(100, device=images.device)]
            out     = self.core_model(patches, tokens, dt=0.01)
            h_v = F.normalize(out.h_vision, dim=-1)
            h_t = F.normalize(out.h_text,   dim=-1)
            return self.scale.exp().clamp(1.0, 200.0) * h_v @ h_t.T

    eval_wrapper = WrapperModel(model, patchifier, trainer.logit_scale).to(device)
    evaluator    = FSCILSessionEvaluator(model=eval_wrapper, device=device, data_manager=manager)
    incremental_epochs = data_cfg.get("incremental_epochs", {}).get("value", 5)

    print(f"\n{'='*50}\nSESSION 0 (BASE) -- {BASE_EPOCHS} epochs\n{'='*50}")
    session_0 = manager.get_session(0)
    seen_mask  = torch.zeros(len(global_tokens), dtype=torch.bool, device=device)
    seen_mask[session_0.classes if hasattr(session_0, 'classes')
              else list(range(manager.base_classes))] = True

    best_acc = 0.0; start = time.time()
    accuracies = {}

    for epoch in range(BASE_EPOCHS):
        print(f"\nEpoch {epoch+1}/{BASE_EPOCHS}")
        trainer.train_mode()
        t0 = time.time()
        for bi, (images, targets) in enumerate(session_0.train_loader):
            images = images.to(device); targets = targets.to(device)
            loss_d = trainer.train_step(patchifier(images), global_tokens, targets,
                                        dt=0.01, seen_classes_mask=seen_mask)
            if bi % 20 == 0:
                ms = (time.time()-t0)/20*1000 if bi > 0 else 0
                scale_v = trainer.logit_scale.exp().item()
                print(f"  Batch {bi}: L={loss_d['L_total']:.4f} | scale={scale_v:.1f} | {ms:.1f}ms/b")
                t0 = time.time()
            if bi > 0 and bi % 500 == 0: torch.cuda.empty_cache()

        metrics = evaluator.evaluate_session(0)
        acc = metrics['accuracy']
        print(f"  Epoch {epoch+1} Acc: {acc:.4f}")
        trainer.save_checkpoint(os.path.join(run_dir, "latest_checkpoint.pt"), seed=SEED)
        if acc > best_acc:
            best_acc = acc
            trainer.save_checkpoint(os.path.join(run_dir, "best_checkpoint.pt"), seed=SEED)
            print(f"  --> New best! {best_acc:.4f}")

    accuracies[0] = best_acc
    print(f"\nSession 0 done. Best acc = {best_acc:.4f}")

    for si in range(1, manager.num_sessions):
        print(f"\n{'='*50}\nSESSION {si}\n{'='*50}")
        trainer.load_checkpoint(os.path.join(run_dir, "best_checkpoint.pt" if si == 1 else f"session_{si-1}.pt"))
        session = manager.get_session(si); trainer.train_mode()
        for ep in range(incremental_epochs):
            t0 = time.time()
            for bi, (images, targets) in enumerate(session.train_loader):
                images = images.to(device); targets = targets.to(device)
                loss_d = trainer.train_step(patchifier(images), global_tokens, targets, dt=0.01)
                if bi % 20 == 0:
                    ms = (time.time()-t0)/20*1000 if bi > 0 else 0
                    print(f"  Batch {bi}: {loss_d['L_total']:.4f} | {ms:.1f}ms/b"); t0 = time.time()
        trainer.save_checkpoint(os.path.join(run_dir, f"session_{si}.pt"), seed=SEED)
        metrics = evaluator.evaluate_session(si)
        print(f"Session {si} acc: {metrics['accuracy']:.4f}")
        accuracies[si] = metrics['accuracy']

    a_base = accuracies[0]; a_last = accuracies[manager.num_sessions-1]
    mean_a = sum(accuracies.values()) / len(accuracies); total_t = time.time() - start
    print(f"\n{'='*60}\n[EXP2] FINAL\n{'='*60}")
    print(f"Base Acc : {a_base*100:.2f}% | Final: {a_last*100:.2f}% | Mean: {mean_a*100:.2f}%")
    print(f"PD={((a_base-a_last)*100):.2f}% | Runtime={total_t:.1f}s")
    for s in sorted(accuracies): print(f"  S{s}: {accuracies[s]*100:.2f}%")
    with open(os.path.join(run_dir, "eval_summary.json"), "w") as f:
        json.dump({"exp": EXP_NAME, "base_acc": a_base, "accuracies": accuracies,
                   "mean_acc": mean_a, "perf_drop": a_base-a_last, "runtime": total_t,
                   "final_logit_scale": trainer.logit_scale.exp().item()}, f, indent=2)

if __name__ == "__main__":
    main()

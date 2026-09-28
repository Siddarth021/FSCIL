"""
EXP4 — Bidirectional Contrastive Loss  [KAGGLE]
================================================
Hypothesis: The current loss is image-to-text (i2t) only.
CLIP's original training uses a SYMMETRIC i2t + t2i InfoNCE loss.
Adding t2i doubles the gradient signal and enforces full alignment
of BOTH the vision and text branches simultaneously.

Changes vs baseline:
  - Loss = L_i2t + L_t2i  (symmetric cross-entropy)
  - warmup_steps=500, restart_period=7500
  - base_epochs = 3 (paper value)
  
L_t2i: For each text, find the correct image in the batch.
This requires unique image-text pairs per batch, which is satisfied
since each image has a unique class label and we compute over all 100 classes.
"""
import os, sys, yaml, time, json, torch, torch.nn as nn, argparse, shutil
import torch.nn.functional as F
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.registry import get_data_manager
from models.achg_clip import ACHGCLIP, ACHGCLIPConfig, ACHGCLIPOutput
from models.clip.clip_wrapper import CLIPConfig
from training.trainer import ACHGCLIPTrainer, TrainerConfig
from evaluation.session_evaluator import FSCILSessionEvaluator
import transformers

# ─── EXP4 OVERRIDES ────────────────────────────────────────────────────────────
EXP_NAME       = "exp4_bidirectional_loss"
BASE_EPOCHS    = 3
WARMUP_STEPS   = 500
RESTART_PERIOD = 7500
LR             = 0.000325
WEIGHT_DECAY   = 0.001
GRAD_ACCUM     = 3
GRAD_CLIP      = 4.0
SEED           = 123
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


class BidirectionalTrainer(ACHGCLIPTrainer):
    """
    Extends ACHGCLIPTrainer to compute a symmetric bidirectional
    cross-entropy loss (i2t + t2i), identical to how CLIP is trained.
    """
    def _compute_ce_loss(self, image_features, text_features, targets, seen_classes_mask=None):
        logit_scale = 100.0
        image_features = F.normalize(image_features, dim=-1)
        text_features  = F.normalize(text_features,  dim=-1)

        # Full similarity matrix: (B, C) where C = num_classes
        logits_i2t = logit_scale * image_features @ text_features.T

        if seen_classes_mask is not None:
            logits_i2t[:, ~seen_classes_mask] = -1e9

        # i2t: each image should match its class text
        loss_i2t = F.cross_entropy(logits_i2t, targets)

        # t2i: for each seen class text, identify which images in the batch belong to it
        # This builds a "soft target" — for each text class c, the probability of each
        # image i is proportional to (target_i == c).
        # We restrict to seen classes only to avoid gradients from unseen-class texts.
        seen_indices = seen_classes_mask.nonzero(as_tuple=True)[0] if seen_classes_mask is not None else None

        if seen_indices is not None and len(seen_indices) > 0:
            # (B, |seen|) logit matrix: how well does image i match seen class j?
            seen_text_feats = text_features[seen_indices]  # (|seen|, d)
            logits_t2i = logit_scale * seen_text_feats @ image_features.T  # (|seen|, B)

            # Build target for each seen class: which images in the batch are from this class?
            # Target is a probability distribution summing to 1
            # targets: (B,) with class indices in global space
            # Map to seen-class index space
            seen_set = seen_indices.tolist()
            global_to_seen = {g: i for i, g in enumerate(seen_set)}

            # For each seen class, compute the mean indicator for which images match
            batch_targets_t2i = torch.zeros(len(seen_indices), len(targets), device=targets.device)
            for i, g_cls in enumerate(seen_set):
                matches = (targets == g_cls).float()
                n_matches = matches.sum()
                if n_matches > 0:
                    batch_targets_t2i[i] = matches / n_matches

            # Only rows with at least one matching image contribute meaningful gradients
            has_match = batch_targets_t2i.sum(dim=1) > 0
            if has_match.any():
                loss_t2i = F.cross_entropy(
                    logits_t2i[has_match],
                    batch_targets_t2i[has_match]
                )
            else:
                loss_t2i = torch.tensor(0.0, device=targets.device)
        else:
            loss_t2i = torch.tensor(0.0, device=targets.device)

        # Symmetric loss: average i2t and t2i
        return (loss_i2t + loss_t2i) * 0.5


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
    print(f"[EXP4] Bidirectional Contrastive Loss | {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_root = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
    with open(os.path.join(cfg_root, "data/cifar100.yaml"))  as f: data_cfg = yaml.safe_load(f)
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

    text_inputs = processor(text=[f"a photo of a {n}" for n in class_names],
                             return_tensors="pt", padding="max_length",
                             max_length=77, truncation=True)
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

    # ─── EXP4 KEY CHANGE: use bidirectional trainer ───────────────────────────
    trainer    = BidirectionalTrainer(model, t_cfg, device)
    print("[EXP4] Using symmetric i2t + t2i cross-entropy loss")
    # ─────────────────────────────────────────────────────────────────────────────

    patchifier = SimplePatchifier().to(device)

    class WrapperModel(nn.Module):
        def __init__(self, core, patch):
            super().__init__(); self.core_model = core; self.patchifier = patch
        def forward(self, images, labels=None):
            out = self.core_model(self.patchifier(images),
                                  global_tokens[torch.arange(100, device=images.device)], dt=0.01)
            h_v = F.normalize(out.h_vision, dim=-1)
            h_t = F.normalize(out.h_text,   dim=-1)
            return 100.0 * h_v @ h_t.T

    eval_wrapper = WrapperModel(model, patchifier).to(device)
    evaluator    = FSCILSessionEvaluator(model=eval_wrapper, device=device, data_manager=manager)
    incremental_epochs = data_cfg.get("incremental_epochs", {}).get("value", 5)

    print(f"\n{'='*50}\nSESSION 0 (BASE) -- {BASE_EPOCHS} epochs\n{'='*50}")
    session_0 = manager.get_session(0)
    seen_mask  = torch.zeros(len(global_tokens), dtype=torch.bool, device=device)
    seen_mask[session_0.classes if hasattr(session_0, 'classes')
              else list(range(manager.base_classes))] = True

    best_acc = 0.0; start = time.time(); accuracies = {}

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
                print(f"  Batch {bi}: L={loss_d['L_total']:.4f} | {ms:.1f}ms/b"); t0 = time.time()
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
        trainer.load_checkpoint(os.path.join(run_dir, "best_checkpoint.pt" if si==1 else f"session_{si-1}.pt"))
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
    print(f"\n{'='*60}\n[EXP4] FINAL\n{'='*60}")
    print(f"Base={a_base*100:.2f}% Final={a_last*100:.2f}% Mean={mean_a*100:.2f}% PD={((a_base-a_last)*100):.2f}%")
    for s in sorted(accuracies): print(f"  S{s}: {accuracies[s]*100:.2f}%")
    with open(os.path.join(run_dir, "eval_summary.json"), "w") as f:
        json.dump({"exp": EXP_NAME, "base_acc": a_base, "accuracies": accuracies,
                   "mean_acc": mean_a, "perf_drop": a_base-a_last, "runtime": total_t}, f, indent=2)

if __name__ == "__main__":
    main()

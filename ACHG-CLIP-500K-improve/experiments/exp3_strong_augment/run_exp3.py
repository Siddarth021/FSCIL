"""
EXP3 — Strong Data Augmentation  [KAGGLE]
==========================================
Hypothesis: The current transform is a plain resize → center crop. The model
overfits to the exact spatial layout of images. Strong augmentation during
base training forces the vision encoder to learn more robust representations.

Changes vs baseline:
  - Train transform: RandomResizedCrop(224, scale=(0.6,1.0)) + RandomHFlip
                     + ColorJitter(b=0.3,c=0.3,s=0.3,h=0.1) + CLIP norm
  - Test transform: unchanged (standard CLIP resize)
  - warmup_steps=500, restart_period=7500 (3 epochs, paper value)
  
Note: We patch the transform at DataLoader level so the model code is unchanged.
"""
import os, sys, yaml, time, json, torch, torch.nn as nn, argparse, shutil, random
import torchvision.transforms as TV
import torch.nn.functional as F
from PIL import Image
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.registry import get_data_manager
from models.achg_clip import ACHGCLIP, ACHGCLIPConfig
from models.clip.clip_wrapper import CLIPConfig
from training.trainer import ACHGCLIPTrainer, TrainerConfig
from evaluation.session_evaluator import FSCILSessionEvaluator
import transformers

# ─── EXP3 OVERRIDES ────────────────────────────────────────────────────────────
EXP_NAME       = "exp3_strong_augment"
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

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


class LoggerTee:
    def __init__(self, fn):
        self.terminal = sys.stdout
        self.log = open(fn, "a", encoding="utf-8")
    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()
    def isatty(self): return hasattr(self.terminal, 'isatty') and self.terminal.isatty()


class SimplePatchifier(nn.Module):
    def forward(self, images): return images


# ─── EXP3 KEY CHANGE: Strong augmentation transform ──────────────────────────
class StrongAugmentTransform:
    """
    Aggressive augmentation for training only:
      - RandomResizedCrop: forces the model to handle partial views
      - RandomHorizontalFlip: basic orientation invariance
      - ColorJitter: color/lighting robustness
      - Normalize: standard CLIP normalization
    """
    def __init__(self, resolution=224):
        self.train_transform = TV.Compose([
            TV.RandomResizedCrop(resolution, scale=(0.6, 1.0), interpolation=TV.InterpolationMode.BICUBIC),
            TV.RandomHorizontalFlip(p=0.5),
            TV.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            TV.ToTensor(),
            TV.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return self.train_transform(img)


class CLIPTestTransform:
    """Standard CLIP test transform (unchanged)."""
    def __init__(self, resolution=224):
        self.transform = TV.Compose([
            TV.Resize(resolution, interpolation=TV.InterpolationMode.BICUBIC),
            TV.CenterCrop(resolution),
            TV.ToTensor(),
            TV.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return self.transform(img)
# ─────────────────────────────────────────────────────────────────────────────


def apply_augmented_transform(dataset, transform):
    """Replaces the transform on a dataset or subset in-place."""
    if hasattr(dataset, 'transform'):
        dataset.transform = transform
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'transform'):
        dataset.dataset.transform = transform
    return dataset


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
    print(f"[EXP3] Strong Data Augmentation | {run_dir}")

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

    # ─── EXP3: Load data then REPLACE train transforms with strong augmentation ──
    manager = get_data_manager(data_cfg, data_root=data_root, synthetic=False)
    print(f"Base classes: {manager.base_classes} | Sessions: {manager.num_sessions}")
    print("[EXP3] Applying strong augmentation to train data...")
    strong_aug = StrongAugmentTransform(224)
    apply_augmented_transform(manager.train_dataset, strong_aug)
    print(f"  Train transform: {strong_aug.train_transform}")
    # ─────────────────────────────────────────────────────────────────────────────

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
    trainer    = ACHGCLIPTrainer(model, t_cfg, device)
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
    print(f"\n{'='*60}\n[EXP3] FINAL\n{'='*60}")
    print(f"Base={a_base*100:.2f}% Final={a_last*100:.2f}% Mean={mean_a*100:.2f}% PD={((a_base-a_last)*100):.2f}%")
    for s in sorted(accuracies): print(f"  S{s}: {accuracies[s]*100:.2f}%")
    with open(os.path.join(run_dir, "eval_summary.json"), "w") as f:
        json.dump({"exp": EXP_NAME, "base_acc": a_base, "accuracies": accuracies,
                   "mean_acc": mean_a, "perf_drop": a_base-a_last, "runtime": total_t}, f, indent=2)

if __name__ == "__main__":
    main()

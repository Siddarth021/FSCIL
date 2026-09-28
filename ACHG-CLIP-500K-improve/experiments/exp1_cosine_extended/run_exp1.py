"""
EXP1 — Extended Cosine LR Training  [LOCAL]
============================================
Hypothesis: Flat LR (the original bug) prevents convergence past 85%.
Fix: Cosine LR with warmup over 5 base epochs should reach ~87-88%.

Changes vs baseline:
  - base_epochs = 5  (paper says 3, ablation here)
  - warmup_steps = 300
  - restart_period = 12500 (5 * 2500 steps/epoch)
  - Everything else identical to baseline
"""
import os, sys, yaml, time, json, torch, torch.nn as nn, argparse, shutil
from datetime import datetime

# Ensure we can import from the parent project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.registry import get_data_manager
from models.achg_clip import ACHGCLIP, ACHGCLIPConfig
from models.clip.clip_wrapper import CLIPConfig
from training.trainer import ACHGCLIPTrainer, TrainerConfig
from evaluation.session_evaluator import FSCILSessionEvaluator
import transformers

# ─── EXP1 OVERRIDES ────────────────────────────────────────────────────────────
EXP_NAME       = "exp1_cosine_extended"
BASE_EPOCHS    = 5           # Extended from paper's 3
WARMUP_STEPS   = 300
RESTART_PERIOD = 12500       # 5 epochs x 2500 steps
LR             = 0.000325
WEIGHT_DECAY   = 0.001
GRAD_ACCUM     = 3
GRAD_CLIP      = 4.0
SEED           = 123
# ────────────────────────────────────────────────────────────────────────────────

MAX_TEXT_LEN = 77
VOCAB_SIZE   = 49408

class LoggerTee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message); self.log.write(message); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()
    def isatty(self):
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

class SimplePatchifier(nn.Module):
    def forward(self, images): return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--variant",   type=str, default="ViT-L/14")
    args = parser.parse_args()

    # ── Run directory ────────────────────────────────────────────────────────────
    results_base = os.environ.get("ACHG_RESULTS_DIR", "results")
    run_dir = os.path.join(results_base, f"{EXP_NAME}_{datetime.now().strftime('%d%m%Y_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    sys.stdout = LoggerTee(os.path.join(run_dir, "train_log.txt"))
    sys.stderr = sys.stdout
    print(f"[EXP1] Extended Cosine LR Training | Run dir: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load configs ─────────────────────────────────────────────────────────────
    cfg_root = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
    with open(os.path.join(cfg_root, "data/cifar100.yaml"))  as f: data_cfg  = yaml.safe_load(f)
    with open(os.path.join(cfg_root, "training.yaml"))       as f: train_cfg = yaml.safe_load(f)
    with open(os.path.join(cfg_root, "model/clip_backbone.yaml")) as f: model_cfg = yaml.safe_load(f)

    # Copy configs for reproducibility
    cfg_dest = os.path.join(run_dir, "configs"); os.makedirs(cfg_dest, exist_ok=True)
    for p in ["data/cifar100.yaml", "training.yaml", "model/clip_backbone.yaml"]:
        shutil.copy(os.path.join(cfg_root, p), cfg_dest)

    # Apply seed
    data_cfg["seed"] = SEED

    variant = args.variant

    # ── Data root fallback ───────────────────────────────────────────────────────
    data_root = args.data_root
    if not os.path.exists(data_root):
        for candidate in ["D:/FSCIL/datasets", "f:/FSCIL/datasets", "/kaggle/input"]:
            if os.path.exists(candidate): data_root = candidate; break
        else:
            os.makedirs(data_root, exist_ok=True)

    print(f"\nInitializing Data Manager  data_root={data_root}")
    manager = get_data_manager(data_cfg, data_root=data_root, synthetic=False)
    print(f"Base classes: {manager.base_classes} | Total sessions: {manager.num_sessions}")

    # ── Processor & text tokens ──────────────────────────────────────────────────
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

    print(f"\nSelected CLIP Variant: {variant}")

    # ── Model config ─────────────────────────────────────────────────────────────
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

    # ─── EXP1 KEY CHANGE: pass warmup + restart_period ──────────────────────────
    t_cfg = TrainerConfig(
        lr=LR, weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        restart_period=RESTART_PERIOD,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_clip_max_norm=GRAD_CLIP,
    )
    print(f"[EXP1] LR={LR} | warmup={WARMUP_STEPS} | restart_period={RESTART_PERIOD} | base_epochs={BASE_EPOCHS}")
    # ─────────────────────────────────────────────────────────────────────────────

    trainer     = ACHGCLIPTrainer(model, t_cfg, device)
    patchifier  = SimplePatchifier().to(device)

    def get_text_tokens(labels, dev): return global_tokens[labels]

    class WrapperModel(nn.Module):
        def __init__(self, core, patch):
            super().__init__(); self.core_model = core; self.patchifier = patch
        def forward(self, images, labels=None):
            patches   = self.patchifier(images)
            tokens    = get_text_tokens(torch.arange(100, device=images.device), images.device)
            out       = self.core_model(patches, tokens, dt=0.01)
            h_v = nn.functional.normalize(out.h_vision, dim=-1)
            h_t = nn.functional.normalize(out.h_text,   dim=-1)
            return 100.0 * h_v @ h_t.T

    eval_wrapper = WrapperModel(model, patchifier).to(device)
    evaluator    = FSCILSessionEvaluator(model=eval_wrapper, device=device, data_manager=manager)

    incremental_epochs = data_cfg.get("incremental_epochs", {}).get("value", 5)

    # ── Session 0: Base Training ─────────────────────────────────────────────────
    print(f"\n{'='*50}\nSESSION 0 (BASE) -- {BASE_EPOCHS} epochs\n{'='*50}")
    session_0 = manager.get_session(0)
    seen_mask  = torch.zeros(len(global_tokens), dtype=torch.bool, device=device)
    seen_mask[session_0.classes if hasattr(session_0, 'classes')
              else list(range(len(manager.base_classes)))] = True

    best_acc = 0.0; start = time.time()

    for epoch in range(BASE_EPOCHS):
        print(f"\nEpoch {epoch+1}/{BASE_EPOCHS}")
        trainer.train_mode()
        t0 = time.time()
        for bi, (images, targets) in enumerate(session_0.train_loader):
            images  = images.to(device); targets = targets.to(device)
            patches = patchifier(images)
            loss_d  = trainer.train_step(patches, global_tokens, targets, dt=0.01,
                                         seen_classes_mask=seen_mask)
            if bi % 20 == 0:
                ms = (time.time()-t0)/20*1000 if bi > 0 else 0
                print(f"  Batch {bi}: L_total={loss_d['L_total']:.4f} | {ms:.1f}ms/b")
                t0 = time.time()
            if bi > 0 and bi % 500 == 0: torch.cuda.empty_cache()

        print("  Evaluating...")
        metrics = evaluator.evaluate_session(0)
        acc = metrics['accuracy']
        print(f"  Epoch {epoch+1} Acc: {acc:.4f}")
        trainer.save_checkpoint(os.path.join(run_dir, "latest_checkpoint.pt"), seed=SEED)
        if acc > best_acc:
            best_acc = acc
            trainer.save_checkpoint(os.path.join(run_dir, "best_checkpoint.pt"), seed=SEED)
            print(f"  --> New best! Saved best_checkpoint.pt")

    end = time.time()
    print(f"\nSession 0 done in {end-start:.1f}s. Best acc = {best_acc:.4f}")
    accuracies = {0: best_acc}

    # ── Incremental Sessions 1-8 ─────────────────────────────────────────────────
    print("\n--- Incremental Sessions ---")
    for si in range(1, manager.num_sessions):
        print(f"\n{'='*50}\nSESSION {si}\n{'='*50}")
        prev_ckpt = os.path.join(run_dir, "best_checkpoint.pt" if si == 1 else f"session_{si-1}.pt")
        trainer.load_checkpoint(prev_ckpt)
        session = manager.get_session(si)
        trainer.train_mode()
        for ep in range(incremental_epochs):
            t0 = time.time()
            for bi, (images, targets) in enumerate(session.train_loader):
                images  = images.to(device); targets = targets.to(device)
                patches = patchifier(images)
                loss_d  = trainer.train_step(patches, global_tokens, targets, dt=0.01)
                if bi % 20 == 0:
                    ms = (time.time()-t0)/20*1000 if bi > 0 else 0
                    print(f"  Batch {bi}: {loss_d['L_total']:.4f} | {ms:.1f}ms/b"); t0 = time.time()
        ckpt = os.path.join(run_dir, f"session_{si}.pt")
        trainer.save_checkpoint(ckpt, seed=SEED)
        metrics = evaluator.evaluate_session(si)
        print(f"Session {si} cumulative acc: {metrics['accuracy']:.4f}")
        accuracies[si] = metrics['accuracy']

    # ── Final Report ──────────────────────────────────────────────────────────────
    a_base = accuracies[0]; a_last = accuracies[manager.num_sessions-1]
    mean_a = sum(accuracies.values()) / len(accuracies)
    total_t = time.time() - start
    print(f"\n{'='*60}\n[EXP1] FINAL RESULTS\n{'='*60}")
    print(f"Variant        : {variant}")
    print(f"Base Accuracy  : {a_base*100:.2f}%")
    print(f"Final Accuracy : {a_last*100:.2f}%")
    print(f"Mean Accuracy  : {mean_a*100:.2f}%")
    print(f"Perf Drop      : {(a_base-a_last)*100:.2f}%")
    print(f"Runtime        : {total_t:.1f}s")
    for s in sorted(accuracies): print(f"  Session {s}: {accuracies[s]*100:.2f}%")
    print("="*60)
    with open(os.path.join(run_dir, "eval_summary.json"), "w") as f:
        json.dump({"exp": EXP_NAME, "base_acc": a_base, "accuracies": accuracies,
                   "mean_acc": mean_a, "perf_drop": a_base-a_last, "runtime": total_t}, f, indent=2)

if __name__ == "__main__":
    main()

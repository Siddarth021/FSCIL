import os
import sys
import yaml
import time
import json
import torch
import torch.nn as nn
import argparse
import shutil
from datetime import datetime

from data.registry import get_data_manager
from models.achg_clip import ACHGCLIP, ACHGCLIPConfig
from models.clip.clip_wrapper import CLIPConfig
from training.trainer import ACHGCLIPTrainer, TrainerConfig
from evaluation.session_evaluator import FSCILSessionEvaluator
import transformers

class LoggerTee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

# CIFAR100 images are resized to 224x224 in our pipeline.
# HF CLIP takes 224x224 raw images.
MAX_TEXT_LEN = 77 # HF max length
VOCAB_SIZE = 49408 # HF CLIP vocab size

class SimplePatchifier(nn.Module):
    def __init__(self, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        
    def forward(self, images):
        # HF CLIP accepts images directly, so we just return them for HF.
        return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", action="store_true", help="Run short validation (2 base epochs, 1 inc epoch)")
    parser.add_argument("--resume_base", type=str, default="", help="Path to a checkpoint to resume base training from")
    parser.add_argument("--variant", type=str, default="", help="CLIP backbone variant (ViT-B/32, ViT-B/16, ViT-L/14)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for class split / data manager")
    parser.add_argument("--data_root", type=str, default="./datasets", help="Path to data root directory")
    parser.add_argument("--base_epochs", type=int, default=None, help="Base epochs (overrides config)")
    parser.add_argument("--incremental_epochs", type=int, default=None, help="Incremental epochs (overrides config)")
    parser.add_argument("--base_batch_size", type=int, default=None, help="Base batch size")
    parser.add_argument("--incremental_batch_size", type=int, default=None, help="Incremental batch size")
    args = parser.parse_args()

    # 1. Generate run folder name
    run_dir = os.path.join("results", datetime.now().strftime("run_%d%m%Y_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    
    # 2. Redirect stdout/stderr to train_log.txt inside the run folder
    sys.stdout = LoggerTee(os.path.join(run_dir, "train_log.txt"))
    sys.stderr = sys.stdout
    
    print(f"Starting run. Logs and checkpoints will be saved to {run_dir}")
    
    # 3. Copy configuration files to the run folder for reproducibility
    config_dest = os.path.join(run_dir, "configs")
    os.makedirs(config_dest, exist_ok=True)
    for cfg_file in ["configs/data/cifar100.yaml", "configs/training.yaml", "configs/model/clip_backbone.yaml"]:
        if os.path.exists(cfg_file):
            shutil.copy(cfg_file, config_dest)
            print(f"Copied {cfg_file} to run directory.")
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("configs/data/cifar100.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/training.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
    with open("configs/model/clip_backbone.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)

    # Apply overrides
    if args.seed is not None:
        if isinstance(data_cfg.get("seed"), dict):
            data_cfg["seed"]["value"] = args.seed
        else:
            data_cfg["seed"] = args.seed
    if args.base_batch_size is not None:
        data_cfg["base_batch_size"]["value"] = args.base_batch_size
    if args.incremental_batch_size is not None:
        data_cfg["incremental_batch_size"]["value"] = args.incremental_batch_size
    if args.base_epochs is not None:
        data_cfg["base_epochs"]["value"] = args.base_epochs
    if args.incremental_epochs is not None:
        data_cfg["incremental_epochs"]["value"] = args.incremental_epochs

    variant = args.variant if args.variant else model_cfg.get("variant", {}).get("value", "ViT-B/32")

    # Resolve data root path
    data_root = args.data_root
    if not os.path.exists(data_root):
        if os.path.exists("D:/FSCIL/datasets"):
            data_root = "D:/FSCIL/datasets"
        elif os.path.exists("f:/FSCIL/datasets"):
            data_root = "f:/FSCIL/datasets"
        else:
            os.makedirs(data_root, exist_ok=True)

    # Instantiate the data manager
    print(f"\nInitializing Data Manager with data_root={data_root}...")
    manager = get_data_manager(data_cfg, data_root=data_root, synthetic=False)
    
    print(f"Dataset availability: Verified.")
    print(f"Base classes (Session 0): {manager.base_classes}")
    print(f"Incremental classes: {manager.incremental_classes}")
    print(f"Total sessions: {manager.num_sessions}")

    # Initialize CLIPProcessor matching variant
    hf_model_map = {
        "ViT-B/32": "openai/clip-vit-base-patch32",
        "ViT-B/16": "openai/clip-vit-base-patch16",
        "ViT-L/14": "openai/clip-vit-large-patch14"
    }
    hf_model_name = hf_model_map.get(variant, "openai/clip-vit-base-patch32")
    print(f"\nLoading HuggingFace CLIPProcessor for {variant} ({hf_model_name})...")
    processor = transformers.CLIPProcessor.from_pretrained(hf_model_name)

    # Get class names
    if hasattr(manager.train_dataset, 'dataset') and hasattr(manager.train_dataset.dataset, 'classes'):
        class_names = manager.train_dataset.dataset.classes
    elif hasattr(manager.train_dataset, 'classes'):
        class_names = manager.train_dataset.classes
    else:
        # Fallback if classes attribute missing
        class_names = [f"class {i}" for i in range(100)]
    
    # Pre-tokenize all 100 classes
    text_prompts = [f"a photo of a {name}" for name in class_names]
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    global_tokens = text_inputs.input_ids.to(device)

    print(f"Selected CLIP Variant: {variant}")

    if variant in ["ViT-B/32", "ViT-B/16"]:
        d_model = 768
        d_e = 512
        d_k = 64
        num_heads = 12
        num_layers = 12
        ffn_hidden_dim = 3072
    elif variant == "ViT-L/14":
        d_model = 1024
        d_e = 768
        d_k = 64
        num_heads = 16
        num_layers = 24
        ffn_hidden_dim = 4096
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Build model Config
    clip_cfg = CLIPConfig(
        variant=variant,
        d_model=d_model,
        d_e=d_e,
        d_k=d_k,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=VOCAB_SIZE,
        max_text_len=MAX_TEXT_LEN,
        patch_dim=1, # not used by HF, but must be > 0 for validation
        max_patches=1, # not used by HF, but must be > 0 for validation
        ffn_hidden_dim=ffn_hidden_dim,
        dropout=0.0
    )

    achg_cfg = ACHGCLIPConfig(
        clip=clip_cfg,
        num_layers=num_layers,
        prompt_dim=d_model,
        node_feature_dim=d_model,
        mlp_bridge_hidden_dim=256,
        gin_num_layers=4,
        gin_hidden_dim=16,
        acga_latent_dim=64,
        hgnec_compressed_dim=8,
        hgnec_restored_dim=16, # restored to gin_hidden_dim
    )

    print("Building model...")
    model = ACHGCLIP(achg_cfg)
    
    print("Setting up TrainerConfig...")
    t_cfg = TrainerConfig(
        lr=train_cfg.get("learning_rate", {}).get("value", 0.000325),
        weight_decay=train_cfg.get("weight_decay", {}).get("value", 0.001),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", {}).get("value", 3),
        gradient_clip_max_norm=train_cfg.get("gradient_clip_max_norm", {}).get("value", 4.0),
    )
    
    print("Setting up Trainer...")
    trainer = ACHGCLIPTrainer(model, t_cfg, device)
    
    print("Setting up Patchifier...")
    patchifier = SimplePatchifier().to(device)
    
    # Evaluation Wrapper Setup
    def get_text_tokens(labels, device):
        return global_tokens[labels]
        
    class WrapperModel(nn.Module):
        def __init__(self, core_model, patchifier):
            super().__init__()
            self.core_model = core_model
            self.patchifier = patchifier
            
        def forward(self, images, labels=None):
            patches = self.patchifier(images)
            num_classes = 100
            tokens = get_text_tokens(torch.arange(num_classes, device=images.device), images.device)
            out = self.core_model(patches, tokens, dt=0.01)
            
            h_v = nn.functional.normalize(out.h_vision, dim=-1)
            h_t = nn.functional.normalize(out.h_text, dim=-1)
            
            logit_scale = 100.0
            logits = logit_scale * h_v @ h_t.T
            return logits

    eval_wrapper = WrapperModel(model, patchifier).to(device)
    evaluator = FSCILSessionEvaluator(
        model=eval_wrapper,
        device=device,
        data_manager=manager,
    )

    # ---------------------------------------------------------
    # Base Training (Session 0)
    # ---------------------------------------------------------
    base_epochs = 2 if args.short else data_cfg.get("base_epochs", {}).get("value", 3)
    incremental_epochs = 1 if args.short else data_cfg.get("incremental_epochs", {}).get("value", 5)
    
    print("\n==================================================")
    print(f"SESSION 0 (BASE TRAINING) - {base_epochs} epochs")
    print("==================================================")
    session_0 = manager.get_session(0)
    
    if args.resume_base:
        print(f"Resuming base training from {args.resume_base}")
        trainer.load_checkpoint(args.resume_base)
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(base_epochs):
        print(f"\nEpoch {epoch+1}/{base_epochs}")
        trainer.train_mode()
        
        batch_start_time = time.time()
        for batch_idx, (images, targets) in enumerate(session_0.train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            tokens = global_tokens
            patches = patchifier(images)
            loss_dict = trainer.train_step(patches, tokens, targets, dt=0.01)
            
            if batch_idx % 20 == 0:
                elapsed = time.time() - batch_start_time
                ms_per_batch = (elapsed / 20 * 1000) if batch_idx > 0 else 0
                print(f"  Batch {batch_idx}: Total Loss = {loss_dict['L_total']:.4f} | {ms_per_batch:.1f} ms/batch")
                batch_start_time = time.time()
                
            if batch_idx > 0 and batch_idx % 500 == 0:
                torch.cuda.empty_cache()
                
        # Evaluate after epoch
        print("  Evaluating base session accuracy...")
        metrics_0 = evaluator.evaluate_session(0)
        acc = metrics_0['accuracy']
        print(f"  Epoch {epoch+1} Accuracy: {acc:.4f}")
        
        # 4. Save Checkpoints
        trainer.save_checkpoint(os.path.join(run_dir, "latest_checkpoint.pt"), seed=42)
        
        if acc > best_acc:
            best_acc = acc
            trainer.save_checkpoint(os.path.join(run_dir, "best_checkpoint.pt"), seed=42)
            print(f"  --> New best accuracy! Saved best_checkpoint.pt")
            
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(os.path.join(run_dir, f"epoch_{epoch+1}.pt"), seed=42)

    end_time = time.time()
    print(f"\nSession 0 Training completed in {end_time - start_time:.2f} seconds.")
    
    accuracies = {0: best_acc}
    
    # ---------------------------------------------------------
    # Incremental Sessions (1-8)
    # ---------------------------------------------------------
    print("\n--- Starting Incremental Training & Evaluation ---")
    
    for session_idx in range(1, manager.num_sessions):
        print(f"\n==================================================")
        print(f"SESSION {session_idx}")
        print(f"==================================================")
        
        # Load previous session checkpoint or best base checkpoint
        prev_session_ckpt = os.path.join(run_dir, f"session_{session_idx-1}.pt")
        if session_idx == 1:
            prev_session_ckpt = os.path.join(run_dir, "best_checkpoint.pt")
            
        print(f"Loading checkpoint from: {prev_session_ckpt}")
        trainer.load_checkpoint(prev_session_ckpt)
        
        session_data = manager.get_session(session_idx)
        
        print("Training...")
        trainer.train_mode()
        
        for epoch in range(incremental_epochs):
            batch_start_time = time.time()
            for batch_idx, (images, targets) in enumerate(session_data.train_loader):
                images = images.to(device)
                targets = targets.to(device)
                
                tokens = global_tokens
                patches = patchifier(images)
                
                loss_dict = trainer.train_step(patches, tokens, targets, dt=0.01)
                
                if batch_idx % 20 == 0:
                    elapsed = time.time() - batch_start_time
                    ms_per_batch = (elapsed / 20 * 1000) if batch_idx > 0 else 0
                    print(f"  Batch {batch_idx}: Total Loss = {loss_dict['L_total']:.4f} | {ms_per_batch:.1f} ms/batch")
                    batch_start_time = time.time()
        
        curr_session_ckpt = os.path.join(run_dir, f"session_{session_idx}.pt")
        trainer.save_checkpoint(curr_session_ckpt, seed=42)
        print(f"Saved checkpoint to {curr_session_ckpt}")
        
        print(f"Evaluating cumulative classes...")
        metrics = evaluator.evaluate_session(session_idx)
        print(f"Cumulative accuracy: {metrics['accuracy']:.4f}")
        accuracies[session_idx] = metrics['accuracy']
        
    print("\nValidation Complete.")
    
    # Calculate final metrics requested by the user
    a_base = accuracies[0]
    a_last = accuracies[manager.num_sessions - 1]
    pd = a_base - a_last
    mean_acc = sum(accuracies.values()) / len(accuracies)
    
    total_runtime = time.time() - start_time # Rough approximation for full script runtime
    
    final_metrics = {
        "Base accuracy (A_base)": a_base,
        "Accuracy after every incremental session": accuracies,
        "A_last": a_last,
        "Mean accuracy": mean_acc,
        "Performance Drop (PD)": pd,
        "Runtime": total_runtime,
        "Exact configuration used": {
            "Data Config": data_cfg,
            "Train Config": train_cfg,
        }
    }
    
    print("\n" + "="*60)
    print("--- FINAL EXPERIMENTAL RESULTS ---")
    print("="*60)
    print(f"Backbone Variant : {variant}")
    print(f"Random Seed      : {data_cfg.get('seed', {}).get('value', 42) if isinstance(data_cfg.get('seed'), dict) else data_cfg.get('seed', 42)}")
    print(f"Base Accuracy    : {a_base*100:.2f}%")
    print(f"Final Accuracy   : {a_last*100:.2f}%")
    print(f"Performance Drop : {pd*100:.2f}%")
    print(f"Mean Accuracy    : {mean_acc*100:.2f}%")
    print(f"Total Runtime    : {total_runtime:.1f}s")
    print("\nSession Accuracies:")
    for s_idx in sorted(accuracies.keys()):
        print(f"  Session {s_idx}: {accuracies[s_idx]*100:.2f}%")
    print("="*60 + "\n")
    
    # Write to a summary log in results directory
    with open(os.path.join(run_dir, "eval_summary.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)

if __name__ == "__main__":
    main()

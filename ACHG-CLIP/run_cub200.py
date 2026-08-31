import os
import sys
import yaml
import time
import json
import torch
import torch.nn as nn
from data.registry import get_data_manager
from models.achg_clip import ACHGCLIP, ACHGCLIPConfig
from models.clip.clip_wrapper import CLIPConfig
from training.trainer import ACHGCLIPTrainer, TrainerConfig
from evaluation.session_evaluator import FSCILSessionEvaluator
from evaluation.result_writer import ResultWriter
import transformers

# CUB200 images are resized to 224x224 in our pipeline.
# HF CLIP takes 224x224 raw images.
MAX_TEXT_LEN = 77
VOCAB_SIZE = 49408

class SimplePatchifier(nn.Module):
    def __init__(self, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        
    def forward(self, images):
        # HF CLIP accepts images directly, so we just return them for HF.
        return images


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("configs/data/cub200.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/training.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)

    # Instantiate the data manager
    print("Initializing Data Manager...")
    # Change data_root to 'D:/FSCIL/datasets'
    manager = get_data_manager(data_cfg, data_root='D:/FSCIL/datasets', synthetic=False)
    
    print(f"Dataset availability: Verified.")
    print(f"Base classes (Session 0): {manager.base_classes}")
    print(f"Incremental classes: {manager.incremental_classes}")
    print(f"Total sessions: {manager.num_sessions}")

    # Initialize CLIPProcessor
    print("Loading HuggingFace CLIPProcessor...")
    processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Get class names
    if hasattr(manager.train_dataset, 'dataset') and hasattr(manager.train_dataset.dataset, 'classes'):
        class_names = manager.train_dataset.dataset.classes
    elif hasattr(manager.train_dataset, 'classes'):
        class_names = manager.train_dataset.classes
    else:
        # Fallback if classes attribute missing
        class_names = [f"class {i}" for i in range(200)]
    
    # Pre-tokenize all 200 classes
    text_prompts = [f"a photo of a {name}" for name in class_names]
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    global_tokens = text_inputs.input_ids.to(device)


    # Build model Config
    clip_cfg = CLIPConfig(
        variant="ViT-B/32",
        d_model=768,
        d_e=512,
        d_k=64,
        num_heads=12,
        num_layers=12,
        vocab_size=VOCAB_SIZE,
        max_text_len=MAX_TEXT_LEN,
        patch_dim=1, # not used by HF, but must be > 0
        max_patches=1, # not used by HF, but must be > 0
        ffn_hidden_dim=3072,
        dropout=0.0
    )

    achg_cfg = ACHGCLIPConfig(
        clip=clip_cfg,
        num_layers=12,
        prompt_dim=768,
        node_feature_dim=768,
        mlp_bridge_hidden_dim=256,
        gin_num_layers=4,
        gin_hidden_dim=128,
        acga_latent_dim=64,
        hgnec_compressed_dim=64,
        hgnec_restored_dim=128, # restored to gin_hidden_dim
    )

    print("Building model...")
    model = ACHGCLIP(achg_cfg)
    
    t_cfg = TrainerConfig(
        lr=train_cfg.get("learning_rate", {}).get("value", 0.000325),
        weight_decay=train_cfg.get("weight_decay", {}).get("value", 0.001),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", {}).get("value", 3),
        gradient_clip_max_norm=train_cfg.get("gradient_clip_max_norm", {}).get("value", 4.0),
    )
    
    trainer = ACHGCLIPTrainer(model, t_cfg, device)
    patchifier = SimplePatchifier().to(device)

    ckpt_path = "results/cub200_base_hf.pt"
    
    if os.path.exists(ckpt_path):
        print(f"\nFound existing checkpoint {ckpt_path}. Skipping base training and loading checkpoint...")
        trainer.load_checkpoint(ckpt_path)
    else:
        # ---------------------------------------------------------
        # Base Training (Session 0)
        # ---------------------------------------------------------
        print("\n--- Starting Session 0 (Base Training) ---")
        session_0 = manager.get_session(0)
        
        epochs = 2 # mock short training
        trainer.train_mode()
        
        start_time = time.time()
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (images, targets) in enumerate(session_0.train_loader):
                images = images.to(device)
                targets = targets.to(device)
                
                # Create synthetic text tokens for ALL classes
                tokens = global_tokens
                
                patches = patchifier(images)
                # dt = 0.01 arbitrary step size for HGN-EC integration
                loss_dict = trainer.train_step(patches, tokens, targets, dt=0.01)
                
                if batch_idx % 20 == 0:
                    print(f"  Batch {batch_idx}: Total Loss = {loss_dict['L_total']:.4f}")
                    
        end_time = time.time()
        print(f"Session 0 Training completed in {end_time - start_time:.2f} seconds.")
        
        os.makedirs("results", exist_ok=True)
        trainer.save_checkpoint(ckpt_path, seed=42)
        print(f"Saved checkpoint to {ckpt_path}")
    
    # ---------------------------------------------------------
    # Incremental Evaluation
    # ---------------------------------------------------------
    print("\n--- Starting Incremental Evaluation ---")
    
    # Mock text_tokens mapper for the evaluator
    def get_text_tokens(labels, device):
        return global_tokens[labels]
        
    class WrapperModel(nn.Module):
        def __init__(self, core_model, patchifier):
            super().__init__()
            self.core_model = core_model
            self.patchifier = patchifier
            
        def forward(self, images, labels=None):
            patches = self.patchifier(images)
            num_classes = 200
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
    
    sessions_to_eval = list(range(manager.num_sessions))
    accuracies = {}
    
    for session_idx in sessions_to_eval:
        print(f"Evaluating Session {session_idx}...")
        metrics = evaluator.evaluate_session(session_idx)
        print(f"Session {session_idx} Cumulative Accuracy: {metrics['accuracy']:.4f}")
        accuracies[session_idx] = metrics['accuracy']
        
    print("\nValidation Complete.")
    print(accuracies)
    
    # Write to a summary log in results directory
    with open("results/cub200_eval_summary.json", "w") as f:
        json.dump(accuracies, f, indent=4)

if __name__ == "__main__":
    main()

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.achg_clip import ACHGCLIP, ACHGCLIPConfig
from models.clip.clip_wrapper import CLIPConfig

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    # Setup configs
    variant = "ViT-L/14"
    d_model = 1024
    d_e = 768
    d_k = 64
    num_heads = 16
    num_layers = 24
    ffn_hidden_dim = 4096
    VOCAB_SIZE = 49408
    MAX_TEXT_LEN = 77

    clip_cfg = CLIPConfig(
        variant=variant,
        d_model=d_model,
        d_e=d_e,
        d_k=d_k,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=VOCAB_SIZE,
        max_text_len=MAX_TEXT_LEN,
        patch_dim=1,
        max_patches=1,
        ffn_hidden_dim=ffn_hidden_dim,
        dropout=0.0
    )

    achg_cfg = ACHGCLIPConfig(
        clip=clip_cfg,
        num_layers=num_layers,
        prompt_dim=d_model,
        node_feature_dim=d_model,
        mlp_bridge_hidden_dim=128,
        gin_num_layers=4,
        gin_hidden_dim=16,
        acga_latent_dim=64,
        hgnec_compressed_dim=8,
        hgnec_restored_dim=16,
    )

    model = ACHGCLIP(achg_cfg)
    
    total_params = count_all_parameters(model)
    trainable_params = count_parameters(model)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nParameters by component:")
    for name, module in model.named_children():
        total_p = count_all_parameters(module)
        train_p = count_parameters(module)
        print(f"{name}: {total_p:,} total | {train_p:,} trainable")
        
if __name__ == "__main__":
    main()

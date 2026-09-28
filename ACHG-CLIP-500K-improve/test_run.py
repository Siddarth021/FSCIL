import torch
import torch.nn as nn
from models.achg_clip import ACHGCLIP, ACHGCLIPConfig

class DummyCLIP(nn.Module):
    def __init__(self, d_model=512, d_e=512):
        super().__init__()
        self.d_model = d_model
        self.d_e = d_e
        
    def encode_image(self, images):
        B = images.shape[0]
        return torch.randn(B, self.d_e)
        
    def encode_text(self, tokens):
        B = tokens.shape[0]
        return torch.randn(B, self.d_e)

def test_achgclip_forward():
    print("Initializing ACHG-CLIP...")
    clip = DummyCLIP(d_model=512, d_e=512)
    clip.hf_model = None
    
    # We need to mock text_encoder.blocks and vision_encoder.blocks for hooks
    class DummyBlock(nn.Module):
        def forward(self, x, *args, **kwargs): return x
    
    class DummyEncoder:
        def __init__(self, L):
            self.blocks = nn.ModuleList([DummyBlock() for _ in range(L)])
            
    clip.text_encoder = DummyEncoder(12)
    clip.vision_encoder = DummyEncoder(12)

    class DummyCLIPConfig:
        def __init__(self):
            self.text_num_layers = 12
            self.vision_num_layers = 12
            self.text_hidden_dim = 512
            self.vision_hidden_dim = 512
            self.num_layers = 12
            self.d_model = 512
            self.d_e = 512
    
    clip.config = DummyCLIPConfig()

    config = ACHGCLIPConfig(
        clip=clip.config,
        num_layers=12,
        num_prompts=1,
        prompt_dim=512,
        node_feature_dim=16,
        mlp_bridge_hidden_dim=32,
        acga_latent_dim=8,
        gin_hidden_dim=16,
        hgnec_compressed_dim=16,
        hgnec_restored_dim=512
    )
    model = ACHGCLIP(config=config, clip_wrapper=clip)
    
    B = 2
    L_img = 50
    L_txt = 10
    images = torch.randn(B, 3, 224, 224)
    tokens = torch.randint(0, 1000, (B, L_txt))
    
    print("Running forward pass...")
    out = model(images, tokens, dt=0.1, num_steps=2)
    
    print("h_vision shape:", out.h_vision.shape)
    print("h_text shape:", out.h_text.shape)
    print("Success!")

if __name__ == "__main__":
    test_achgclip_forward()

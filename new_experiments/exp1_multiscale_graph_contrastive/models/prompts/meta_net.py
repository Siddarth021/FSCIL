import torch
import torch.nn as nn
from typing import Optional

class MetaNet(nn.Module):
    """
    Meta-Network for generating instance-conditioned dynamic prompts.
    It takes global features (e.g., from a frozen CLIP pass) and generates the initial
    Hamiltonian state (q, p) offsets, which are added to the static base prompts.
    
    This solves the issue of static prompts overfitting to base classes in FSCIL.
    """
    def __init__(self, global_feature_dim: int, prompt_dim: int, num_layers: int, num_prompts: int = 1, hidden_dim: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        
        self.net = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_layers * num_prompts * prompt_dim)
        )
        
        # Initialize output to near zero so we start close to the static prompt
        nn.init.constant_(self.net[-1].weight, 0.0)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, global_features: torch.Tensor, static_prompts: torch.Tensor) -> torch.Tensor:
        """
        global_features: (B, global_feature_dim)
        static_prompts: (L, M, d)
        
        Returns:
        dynamic_prompts: (B, L, M, d)
        """
        B = global_features.shape[0]
        # Generate prompt offsets
        offsets = self.net(global_features) # (B, L * M * d)
        offsets = offsets.view(B, self.num_layers, self.num_prompts, self.prompt_dim)
        
        # Add to static prompts (which are broadcasted to batch)
        # static_prompts: (L, M, d) -> (1, L, M, d) -> (B, L, M, d)
        static_prompts_b = static_prompts.unsqueeze(0)
        return static_prompts_b + offsets

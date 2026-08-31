"""
models/clip/transformer_block.py
==================================

Stage 2 — CLIP backbone: one Transformer block (Eqs. 3-4).

Paper text (Section IV.A.1, "Feature Interaction"):

    X'  = LayerNorm(X + MHA(X_query, X_key, X_value))          (Eq. 3)
    X'' = LayerNorm(X' + FFN(X'))                                (Eq. 4)

This block is shared, unmodified, between the text tower (`text_encoder.py`) and the vision
tower (`vision_encoder.py`), per Section IV.A's "symmetric text and vision encoders, both
based on improved Transformer structures" — one `TransformerBlock` class, separate weight
instances stacked `L` times per tower.

The paper never specifies the FFN's hidden-layer width or activation function (Eq. 4 only
names `FFN(X')` symbolically). This module defaults to the conventional Transformer choice
(hidden = 4 * d_model, GELU activation) as an explicit `IMPLEMENTATION_CHOICE`, not a paper
fact — callers must not describe this as reproducing a specific paper-stated FFN design.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.clip.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise FFN inside a Transformer block.

    IMPLEMENTATION_CHOICE: hidden_dim and activation are not specified by the paper for this
    block (distinct from GIN's/ACGA's MLP, whose composition IS partially paper-stated
    elsewhere). Default hidden_dim = 4 * d_model (conventional Transformer ratio), GELU
    activation. Both are constructor arguments so a future config entry can override them
    without touching this file.
    """

    def __init__(self, d_model: int, hidden_dim: "int | None" = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """One pre-residual Transformer block implementing Eqs. 3-4 exactly.

    Note on LayerNorm placement: Eq. 3/4's `LayerNorm(X + sublayer(X))` is literally a
    *post*-norm residual (norm applied to the sum), which is what is implemented here — this
    is a direct, unambiguous reading of the equations, not an inference.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int,
        ffn_hidden_dim: "int | None" = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, d_k=d_k, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, hidden_dim=ffn_hidden_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: "torch.Tensor | None" = None) -> torch.Tensor:
        """`x`: `(B, seq, d_model)` -> `(B, seq, d_model)` (Eqs. 3-4)."""
        attn_out = self.attn(x, x, x, attn_mask=attn_mask)  # Eq. 3's MHA(Xq=Xk=Xv=X)
        x_prime = self.norm1(x + attn_out)  # Eq. 3
        ffn_out = self.ffn(x_prime)
        x_double_prime = self.norm2(x_prime + ffn_out)  # Eq. 4
        return x_double_prime

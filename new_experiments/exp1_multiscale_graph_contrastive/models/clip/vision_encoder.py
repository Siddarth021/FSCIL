"""
models/clip/vision_encoder.py
================================

Stage 2 — CLIP backbone: vision tower (Eqs. 7-8).

Paper text (Section IV.A.2):

    X = [x_CLS; E_V(p_1), ..., E_V(p_m)] + P_V(m+1),  E_V ∈ R^{d×|p|}, P_V ∈ R^{(m+1)×d}   (Eq. 7)
    ... L Transformer blocks (Eqs. 3-4, shared with the text tower) ...
    h*_V = normalize( W_V . X^(L)_[CLS] ),  W_V ∈ R^{d_e×d}                                (Eq. 8)

The vision tower is "symmetric" to the text tower (Section IV.A) and reuses the exact same
`TransformerBlock` class from `transformer_block.py`, with its own separate weights.

Unresolved paper details (see `configs/model/clip_backbone.yaml`, all UNRESOLVED — not
silently defaulted here): patch dimension `|p|` / patch extraction scheme, `d`, `d_e`, `L`,
`d_k`/head count, and positional-encoding type are all required constructor arguments,
exactly as in `text_encoder.py`. This module also does not implement raw-image ->
patch-sequence extraction (e.g. Conv2d patchify) — Eq. 7 already takes `patches` as its
input, so patch extraction is left to a caller-supplied preprocessing step (out of scope for
Stage 2, which implements only Eqs. 1-8's Transformer math on already-tokenized/patchified
inputs, per `FINAL_IMPLEMENTATION_BLUEPRINT.md` Part 2's module mapping).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip.transformer_block import TransformerBlock


class VisionEmbedding(nn.Module):
    """Patch linear-projection + [CLS] token + learned positional encoding (Eq. 7)."""

    def __init__(self, patch_dim: int, d_model: int, max_patches: int):
        super().__init__()
        self.patch_dim = patch_dim
        self.d_model = d_model
        self.max_patches = max_patches

        # E_V, Eq. 7: R^{d x |p|}, applied as a linear projection per patch.
        self.patch_projection = nn.Linear(patch_dim, d_model, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # P_V, Eq. 7: R^{(m+1) x d} -- IMPLEMENTATION_CHOICE: learned table, same rationale as
        # TextEmbedding's positional_embedding (positional_encoding_type is UNRESOLVED).
        self.positional_embedding = nn.Parameter(torch.zeros(max_patches + 1, d_model))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)
        nn.init.normal_(self.patch_projection.weight, std=0.02)
        nn.init.zeros_(self.patch_projection.bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """`patches`: `(B, m, patch_dim)` -> `(B, m+1, d)` (Eq. 7)."""
        if patches.dim() != 3:
            raise ValueError(f"patches must be (B, m, patch_dim); got shape {tuple(patches.shape)}.")
        b, m, patch_dim = patches.shape
        if patch_dim != self.patch_dim:
            raise ValueError(f"Expected patch_dim={self.patch_dim}, got {patch_dim}.")
        if m > self.max_patches:
            raise ValueError(f"Number of patches {m} exceeds configured max_patches={self.max_patches}.")

        projected = self.patch_projection(patches)  # E_V(p_i): (B, m, d)
        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, d)
        x = torch.cat([cls, projected], dim=1)  # [x_CLS; E_V(p_1),...,E_V(p_m)]: (B, m+1, d)
        x = x + self.positional_embedding[: m + 1, :].unsqueeze(0)  # + P_V(m+1)
        return x


class VisionEncoder(nn.Module):
    """Full vision tower: patch embedding -> L Transformer blocks -> CLS projection (Eqs. 7-8).

    Does not yet insert learnable vision prompts (Eq. 10) -- that is Stage 3
    (`models/prompts/vision_prompt.py`).
    """

    def __init__(
        self,
        patch_dim: int,
        d_model: int,
        d_e: int,
        num_layers: int,
        num_heads: int,
        d_k: int,
        max_patches: int,
        ffn_hidden_dim: "int | None" = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_e = d_e
        self.num_layers = num_layers

        self.embedding = VisionEmbedding(patch_dim=patch_dim, d_model=d_model, max_patches=max_patches)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_k=d_k,
                    ffn_hidden_dim=ffn_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(d_model)
        # W_V, Eq. 8
        self.projection = nn.Linear(d_model, d_e, bias=False)

    def forward(self, patches: torch.Tensor, return_sequence: bool = False):
        """`patches`: `(B, m, patch_dim)` -> `h*_V`: `(B, d_e)` (L2-normalized, Eq. 8).

        Args:
            patches: `(B, m, patch_dim)` raw patch vectors.
            return_sequence: if True, also return the final-layer sequence output `X^(L)`
                (`(B, m+1, d)`, CLS token at index 0), needed by later stages, not by Eqs. 7-8
                themselves.
        """
        x = self.embedding(patches)  # Eq. 7: (B, m+1, d)
        for block in self.blocks:
            x = block(x)  # Eqs. 3-4, applied L times
        x = self.final_layer_norm(x)  # X^(L)

        cls_final = x[:, 0, :]  # X^(L)_[CLS], Eq. 8: (B, d)
        projected = self.projection(cls_final)  # W_V . X^(L)_[CLS]: (B, d_e)
        h_vision = F.normalize(projected, p=2, dim=-1)  # Eq. 8's L2 normalization

        if return_sequence:
            return h_vision, x
        return h_vision

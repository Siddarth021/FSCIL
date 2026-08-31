"""
models/clip/text_encoder.py
=============================

Stage 2 — CLIP backbone: text tower (Eqs. 1-6).

Paper text (Section IV.A.1):

    X = E_T(tokens) ∈ R^{n×d}                                            (Eq. 1)
    X <- X + P_T(n),  P_T ∈ R^{n×d}                                       (Eq. 2)
    ... L Transformer blocks (Eqs. 3-4, transformer_block.py) ...
    h*_T = normalize( W_T . AvgPool(X^(L)) ),  W_T ∈ R^{d_e×d}            (Eq. 6)

Notes on unresolved paper details (all recorded in `configs/model/clip_backbone.yaml` as
UNRESOLVED, not silently defaulted here):
  - `d` (token embedding dim), `d_e` (projection dim), `L` (layer count), `d_k`/num_heads
    (attention head sizing) have no paper-stated numeric value. This module takes them as
    required constructor arguments; `CLIPConfig` (`clip_wrapper.py`) is responsible for
    supplying them (from a resolved config with an explicit override, or from a test-only
    synthetic config) and refuses to proceed silently if they are missing.
  - `configs/model/clip_backbone.yaml: positional_encoding_type` is UNRESOLVED ("learned vs.
    sinusoidal" -- paper doesn't say). This module implements a **learned** positional
    embedding table as the default (`P_T` as an `nn.Embedding`/parameter, trained like any
    other weight) -- the simplest reading consistent with Eq. 2's matrix notation, recorded
    here as an IMPLEMENTATION_CHOICE, not a paper fact.
  - No causal attention mask is applied: no equation or prose describes one, so
    `TransformerBlock` is called with `attn_mask=None` throughout (Eq. 3 shows plain,
    unmasked self-attention).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip.transformer_block import TransformerBlock


class TextEmbedding(nn.Module):
    """Token embedding + learned positional encoding, per Eqs. 1-2."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # E_T, Eq. 1
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # P_T, Eq. 2 -- IMPLEMENTATION_CHOICE: learned positional table (see module docstring)
        self.positional_embedding = nn.Parameter(torch.zeros(max_seq_len, d_model))
        nn.init.normal_(self.positional_embedding, std=0.02)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """`tokens`: `(B, n)` int64 -> `(B, n, d)` (Eqs. 1-2)."""
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be (B, n); got shape {tuple(tokens.shape)}.")
        b, n = tokens.shape
        if n > self.max_seq_len:
            raise ValueError(
                f"Sequence length {n} exceeds configured max_seq_len={self.max_seq_len}."
            )
        x = self.token_embedding(tokens)  # Eq. 1: (B, n, d)
        x = x + self.positional_embedding[:n, :].unsqueeze(0)  # Eq. 2
        return x


class TextEncoder(nn.Module):
    """Full text tower: embedding -> L Transformer blocks -> projection (Eqs. 1-6).

    This class does not yet insert learnable prompts (Eq. 9) -- that is Stage 3
    (`models/prompts/text_prompt.py`), which will wrap or extend this class without altering
    Eqs. 1-6's implementation here.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_e: int,
        num_layers: int,
        num_heads: int,
        d_k: int,
        max_seq_len: int,
        ffn_hidden_dim: "int | None" = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_e = d_e
        self.num_layers = num_layers

        self.embedding = TextEmbedding(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)
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
        # W_T, Eq. 6
        self.projection = nn.Linear(d_model, d_e, bias=False)

    def forward(self, tokens: torch.Tensor, return_sequence: bool = False):
        """`tokens`: `(B, n)` -> `h*_T`: `(B, d_e)` (L2-normalized, Eq. 6).

        Args:
            tokens: `(B, n)` int64 token ids.
            return_sequence: if True, also return the final-layer sequence output
                `X^(L)` (`(B, n, d)`, pre-pool/pre-projection) -- needed by later stages that
                consume per-layer prompt state, not required by Eqs. 1-6 themselves.
        """
        x = self.embedding(tokens)  # Eq. 1-2: (B, n, d)
        for block in self.blocks:
            x = block(x)  # Eqs. 3-4, applied L times
        x = self.final_layer_norm(x)  # X^(L)

        pooled = x.mean(dim=1)  # AvgPool(X^(L)), Eq. 6: (B, d)
        projected = self.projection(pooled)  # W_T . AvgPool(X^(L)): (B, d_e)
        h_text = F.normalize(projected, p=2, dim=-1)  # Eq. 6's L2 normalization

        if return_sequence:
            return h_text, x
        return h_text

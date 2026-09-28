"""
models/clip/attention.py
=========================

Stage 2 — CLIP backbone: multi-head self-attention (Eq. 5).

Paper text (Section IV.A.1, "Feature Interaction"):

    head_i = Softmax( X W^i_Query (X W^i_Key)^T / sqrt(d_k) ) X W^i_Value      (Eq. 5)
    MHA(Q, K, V) = Concat(head_1, ..., head_h) W_Output

`d_k` is the per-head query/key dimension; the paper states it is used "to scale the
dot-product attention scores for stable training" but never gives it, or the head count `h`,
a numeric value anywhere (`configs/model/clip_backbone.yaml: attention_head_dim_dk`,
provenance UNRESOLVED). This module treats both `d_k` and `num_heads` as required
constructor arguments supplied by the caller (`CLIPConfig`, see `clip_wrapper.py`) rather than
inventing a default here — consistent with Stage 1's "never silently assign a value to an
UNRESOLVED parameter" rule. `CLIPConfig` is responsible for resolving/validating these before
this module is ever instantiated.

This is a single, shared implementation used identically by both the text and vision towers
(Section IV.A: "symmetric text and vision encoders, both based on improved Transformer
structures") — one `MultiHeadAttention` class, two separate weight instances (one per tower),
exactly like `TransformerBlock` in `transformer_block.py`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product self-attention, per Eq. 5.

    Note: the paper's Eq. 5 shows a single, shared input `X` used to derive per-head
    query/key/value projections (`X W^i_Query`, `X W^i_Key`, `X W^i_Value`) — i.e. this is
    self-attention (`Xquery == Xkey == Xvalue == X`), matching how Eq. 3 calls
    `MHA(Xquery, Xkey, Xvalue)` with all three symbols bound to the same `X` at every call
    site in the paper. The `forward` signature keeps the three named arguments distinct
    (matching Eq. 3's notation exactly) purely for readability / future extensibility; every
    current call in this codebase passes the same tensor for all three.
    """

    def __init__(self, d_model: int, num_heads: int, d_k: int, dropout: float = 0.0):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if d_k <= 0:
            raise ValueError(f"d_k must be positive, got {d_k}.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        # Eq. 5 concatenates h heads then applies W_Output; the paper does not state that
        # h * d_k must equal d_model, but this is the only architecture-consistent choice
        # (W_Output must map the concatenated head dimension back to d_model for the residual
        # add in Eq. 3 to type-check). Recorded as an IMPLEMENTATION_CHOICE constraint, not a
        # paper fact, in `CLIPConfig` (see clip_wrapper.py).
        self.concat_dim = num_heads * d_k

        self.w_query = nn.Linear(d_model, self.concat_dim, bias=False)
        self.w_key = nn.Linear(d_model, self.concat_dim, bias=False)
        self.w_value = nn.Linear(d_model, self.concat_dim, bias=False)
        self.w_output = nn.Linear(self.concat_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = 1.0 / math.sqrt(d_k)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, seq, concat_dim) -> (B, num_heads, seq, d_k)
        b, seq, _ = x.shape
        x = x.view(b, seq, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        x_query: torch.Tensor,
        x_key: torch.Tensor,
        x_value: torch.Tensor,
        attn_mask: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """Compute Eq. 5's multi-head attention.

        Args:
            x_query, x_key, x_value: each `(B, seq, d_model)`. In every current call site
                these are the same tensor (self-attention), per the module docstring.
            attn_mask: optional additive mask broadcastable to `(B, num_heads, seq_q, seq_k)`
                (e.g. `-inf` at disallowed positions). Not required by any paper equation;
                exposed only so causal/padding masking can be added later without changing
                this module's interface (Eqs. 1-8 do not describe masking).

        Returns:
            `(B, seq, d_model)` — already projected back by `W_Output`.
        """
        if x_query.dim() != 3 or x_key.dim() != 3 or x_value.dim() != 3:
            raise ValueError(
                "MultiHeadAttention expects 3D tensors (B, seq, d_model); got shapes "
                f"{tuple(x_query.shape)}, {tuple(x_key.shape)}, {tuple(x_value.shape)}."
            )

        q = self._split_heads(self.w_query(x_query))  # (B, h, seq_q, d_k)
        k = self._split_heads(self.w_key(x_key))  # (B, h, seq_k, d_k)
        v = self._split_heads(self.w_value(x_value))  # (B, h, seq_k, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, seq_q, seq_k)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, h, seq_q, d_k)
        b, h, seq_q, d_k = out.shape
        out = out.permute(0, 2, 1, 3).reshape(b, seq_q, h * d_k)  # (B, seq_q, concat_dim)
        return self.w_output(out)  # (B, seq_q, d_model)

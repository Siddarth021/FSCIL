"""
models/acga/encoder.py
========================

Stage 6 — ACGA graph convolutional encoder (Eq. 22).

    Z^(l+1) = GINLayer(Z^(l), A)                                                    (Eq. 22)

Section IV.C.1: "The encoder employs an improved Graph Isomorphism Network (GIN)... Finally,
the encoded output is Z = GIN(X, A) in R^{N x K}, where N denotes the number of nodes in the
graph and K represents the dimension of the latent space... N and K are independent, with no
functional relationship between them."

This module does NOT reimplement GIN's message-passing arithmetic. Per `configs/model/
gin.yaml`'s own module docstring ("reused verbatim inside ACGA's encoder... per
equation_mapping.md's 'Notes on equations appearing twice'") and `models/gnn/gin_layer.py`'s
`GIN.forward_tensors` docstring (explicitly naming ACGA's encoder as a reuse site), the
encoder is built directly on top of the frozen Stage-5 `models.gnn.gin_layer.GIN` stack.

--------------------------------------------------------------------------------------------
REFERENCE TRACEABILITY
--------------------------------------------------------------------------------------------

Component: ACGA encoder architecture (GIN-based message passing).
Equation/reference: Eq. 22.
Source: Section IV.C.1, "The encoder employs an improved Graph Isomorphism Network (GIN),
    which fuses node features and topological information through a multi-layer
    message-passing mechanism."
Evidence type: PAPER-FACT.
Confidence: High.

Component: Encoder MLP composition (Linear -> BatchNorm -> GELU, reused per layer).
Equation/reference: Eq. 22, sentence immediately following.
Source: Section IV.C.1: "MLP is composed of linear layers, batch normalization, and an
    activation function (GELU)." Matches `configs/model/acga.yaml: encoder_mlp_composition`.
Evidence type: PAPER-FACT.
Confidence: High.

Component: Encoder input dimension `D`.
Equation/reference: Eq. 22's `Z^(l+1) = GINLayer(Z^(l), A)`.
Source: The encoder consumes the Stage-5 GIN output exactly (per this Stage-6 task's INPUT
    section: "ACGA consumes the GIN output according to the frozen
    FINAL_IMPLEMENTATION_BLUEPRINT. Do not redesign the Stage-5 GIN interface."), i.e.
    `Graph.feature_dim == configs/model/gin.yaml: hidden_dim` (16, PAPER-FACT) once the
    upstream GIN stack has run. Encoder's own `input_dim` is left caller-supplied (no
    hard-coded 16) so this module is not silently coupled to a specific upstream config value.
Evidence type: IMPLEMENTATION-CHOICE (decoupling), inheriting a PAPER-FACT upstream value.
Confidence: High.

Component: Latent dimension `K`.
Equation/reference: Eq. 22 / Section IV.C.1.
Source: `configs/model/acga.yaml: latent_dim_K` -- "No numeric value given anywhere" in the
    paper. UNRESOLVED.
Evidence type: UNRESOLVED. `ACGAEncoderConfig.latent_dim` is therefore a REQUIRED constructor
    argument with NO default (mirrors `GINConfig.input_dim`'s established convention for
    UNRESOLVED dimensions in this codebase) -- this module never silently invents a numeric K.
Confidence: N/A (unresolved).

Component: Number of stacked GIN layers inside the encoder (`encoder_num_layers`).
Equation/reference: Eq. 22 is written as a single generic per-layer update
    `Z^(l+1) = GINLayer(Z^(l), A)`; the paper never states how many times this update is
    applied before "the encoded output" `Z` is taken.
Source: Not stated anywhere in Section IV.C. Not present in `configs/model/acga.yaml` (no
    `encoder_num_layers` key exists there, mirroring `configs/model/gin.yaml`'s own precedent
    of leaving `mlp_hidden_dim`-style internal knobs code-level rather than YAML-tracked when
    the paper gives no numeric hook to record).
Evidence type: IMPLEMENTATION-CHOICE. Default `encoder_num_layers=1` -- the most literal
    reading of Eq. 22's single-arrow notation (`Z^(l+1) = GINLayer(Z^(l), A)` describes one
    application producing the next state; "the encoded output" is read as "the result after
    one such application" absent any stated layer count). Configurable via
    `ACGAEncoderConfig.num_layers` so this reading can be revisited without touching call
    sites. NEVER labeled PAPER-FACT.
Confidence: Low.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.gnn.gin_layer import GIN, GINConfig, GINLayerShapeError

_SUPPORTED_MLP_COMPOSITIONS = {"linear_batchnorm_gelu"}


class ACGAEncoderConfigError(Exception):
    """Raised when an `ACGAEncoderConfig`'s own fields are invalid."""


@dataclass
class ACGAEncoderConfig:
    """Config for `ACGAEncoder` (Eq. 22).

    input_dim:  `D`, the node feature dimension the encoder consumes -- i.e. the Stage-5
                `GIN.forward`/`Graph.feature_dim` output width (`configs/model/gin.yaml:
                hidden_dim`, PAPER-FACT 16, but never hard-coded here -- see module
                docstring). Required, no default (mirrors `GINConfig.input_dim`).
    latent_dim: `K`, the latent space dimension (`configs/model/acga.yaml: latent_dim_K`,
                UNRESOLVED in the paper). Required, no default -- this module never silently
                invents a numeric K.
    num_layers: number of stacked `GINLayer` applications of Eq. 22 before "the encoded
                output" `Z` is taken. IMPLEMENTATION-CHOICE default `1` (see module
                docstring's reference-traceability block). Every layer after the first maps
                `latent_dim -> latent_dim` (mirrors `GINConfig`'s own layer-width convention).
    mlp_hidden_dim: forwarded to the underlying `GINConfig`; `None` -> defaults to
                `latent_dim` per layer (matches `GINLayerConfig.mlp_hidden_dim`'s own
                convention).
    eps_init:   forwarded to the underlying `GINConfig` (Eq. 13's learnable epsilon,
                IMPLEMENTATION-CHOICE default `0.0`, same as Stage 5).
    provenance: free-form dict, mirrors the rest of the codebase's config dataclasses.
    """

    input_dim: int
    latent_dim: int
    num_layers: int = 1
    mlp_hidden_dim: Optional[int] = None
    eps_init: float = 0.0
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("input_dim", "latent_dim", "num_layers"):
            val = getattr(self, name)
            if not isinstance(val, int) or val <= 0:
                raise ACGAEncoderConfigError(
                    f"ACGAEncoderConfig.{name} must be a positive int, got {val!r}."
                )


class GATLayer(nn.Module):
    """Simple Multi-Head Graph Attention Layer (GATv2 style) for ACGA."""
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        # Ensure divisibility
        while heads > 1 and out_dim % heads != 0:
            heads -= 1
        self.heads = heads
        self.out_dim = out_dim
        self.head_dim = out_dim // self.heads
        
        self.W = nn.Linear(in_dim, out_dim)
        self.a = nn.Linear(2 * self.head_dim, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # X: (..., N, in_dim)
        h = self.W(X) # (..., N, out_dim)
        
        # Reshape for multi-head
        leading = h.shape[:-2]
        N = h.shape[-2]
        h = h.view(*leading, N, self.heads, self.head_dim)
        
        # h_i || h_j
        h_i = h.unsqueeze(-3).expand(*leading, N, N, self.heads, self.head_dim)
        h_j = h.unsqueeze(-4).expand(*leading, N, N, self.heads, self.head_dim)
        
        h_cat = torch.cat([h_i, h_j], dim=-1) # (..., N, N, heads, 2*head_dim)
        e = self.leakyrelu(self.a(h_cat).squeeze(-1)) # (..., N, N, heads)
        
        # Mask
        A_mask = A.unsqueeze(-1).expand_as(e)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(A_mask > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=-3) # softmax over j
        
        # attention is (..., N, N, heads)
        # h is (..., N, heads, head_dim)
        # We need sum_j (attention * h_j)
        h_prime = torch.einsum('...ijh,...jhd->...ihd', attention, h)
        h_prime = h_prime.reshape(*leading, N, self.out_dim)
        
        return torch.nn.functional.elu(h_prime)

class ACGAEncoder(nn.Module):
    """ACGA's graph convolutional encoder (Eq. 22): `X, A -> Z in R^{N x K}`.

    Novelty 3: Replaces GIN with a Multi-Head Graph Attention Network (GAT).
    """

    def __init__(self, config: ACGAEncoderConfig):
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.num_layers):
            in_dim = config.input_dim if i == 0 else config.latent_dim
            layers.append(GATLayer(in_dim, config.latent_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """`X`: `(..., N, input_dim)`, `A`: `(..., N, N)` -> `Z`: `(..., N, latent_dim)`."""
        if X.shape[-1] != self.config.input_dim:
            raise GINLayerShapeError(
                f"ACGAEncoder: X last dim {X.shape[-1]} does not match "
                f"config.input_dim {self.config.input_dim}."
            )
        out = X
        for layer in self.layers:
            out = layer(out, A)
        return out

    def extra_repr(self) -> str:
        return f"input_dim={self.config.input_dim}, latent_dim={self.config.latent_dim}, num_layers={self.config.num_layers}"

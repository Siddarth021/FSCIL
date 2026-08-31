"""
models/graph/adjacency.py
============================

Stage 4 — adjacency matrix construction (Eqs. 14-18) and top-level `build_graph`.

Behavior: cosine similarity between node feature vectors.
Source: Eq. 14: sim_matrix_{i,j} = cos(x_i, x_j).
Evidence type: PAPER-FACT (`configs/model/graph.yaml: cosine_similarity`).

Behavior: binarize with a STRICT '>' threshold of 0.8.
Source: Eq. 15; Section V.D.2 / Fig. 3 sensitivity heatmap selects 0.8 as optimal.
Evidence type: PAPER-FACT (`configs/model/graph.yaml: adjacency_threshold,
    threshold_comparator`). Boundary value (similarity exactly == threshold) maps to 0, per
    the strict inequality.

Behavior: symmetrize via `Z = (A + A^T) / 2`.
Source: Eq. 16.
Evidence type: PAPER-FACT (`configs/model/graph.yaml: symmetrization`).

Behavior: degree-normalize via `A~ = D^(-1/2) . Z . D^(-1/2)`.
Source: Eq. 17.
Evidence type: PAPER-FACT (`configs/model/graph.yaml: degree_normalization`).
Implementation note: isolated nodes (degree 0) would divide by zero in `D^(-1/2)`; the paper
    does not address this edge case. This module adds a small `eps` to the degree before the
    inverse-sqrt (IMPLEMENTATION-CHOICE, not a paper fact) purely for numerical stability --
    documented, not silently baked in as if specified.

Behavior: optional attention-based reweighting, `A~ = A~ . attention`, `attention =
    softmax(sim_matrix)`.
Source: Eq. 18, explicitly marked "optional" in prose; the paper never states whether the
    final reported model actually uses it.
Evidence type: JUSTIFIED-INFERENCE (`configs/model/graph.yaml:
    optional_attention_reweight_enabled = false` by default -- ambiguity_log.md item A7). This
    module implements the reweight as a togglable step (`build_adjacency(..., config)`
    reads `config.attention_reweight_enabled`) but never turns it on by default.

Behavior: adjacency construction operates per modality; graph node count N = L (Blocker 1, see
    `node_builder.py`).
Source: Section IV.B "respectively" (dual per-modality GIN/graph processing).
Evidence type: PAPER-FACT (per-modality processing) + IMPLEMENTATION-CHOICE (N=L itself,
    Blocker 1).
Implementation note: `build_graph` builds exactly one `Graph` per call (one modality); a
    caller builds the text and vision graphs with two separate calls, matching Blocker 5's
    independent-per-modality-processing decision (not re-derived here, only respected).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch

from models.graph.graph_data import Graph
from models.graph.node_builder import NodeBuilderConfig, build_nodes
from models.prompts.mlp_bridge import PromptToNodeMLP

_EPS = 1e-8  # IMPLEMENTATION-CHOICE numerical-stability epsilon for isolated-node degree norm.


class AdjacencyConfigError(Exception):
    """Raised when an `AdjacencyConfig`'s own fields are invalid."""


class AdjacencyShapeError(Exception):
    """Raised when input to an adjacency-construction function has an invalid shape."""


@dataclass
class AdjacencyConfig:
    """Config for adjacency construction (Eqs. 14-18).

    threshold: PAPER-FACT default 0.8 (Section V.D.2). Kept configurable, not hard-coded,
               so a caller can point it at `configs/model/graph.yaml` explicitly rather than
               this module silently re-stating the paper value as a Python constant.
    attention_reweight_enabled: PAPER-FACT-adjacent JUSTIFIED-INFERENCE default `False`
               (Eq. 18 marked "optional"; paper never confirms it is used in the reported
               model) -- see module docstring.
    """

    threshold: float = 0.8
    attention_reweight_enabled: bool = False
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (-1.0 <= self.threshold <= 1.0):
            raise AdjacencyConfigError(f"AdjacencyConfig.threshold must be in [-1, 1] (cosine sim range), got {self.threshold!r}.")


def _validate_node_features(X: torch.Tensor, fn_name: str) -> None:
    if X.dim() not in (2, 3):
        raise AdjacencyShapeError(f"{fn_name}: expected X of shape (N, D) or (B, N, D), got {tuple(X.shape)}.")


def cosine_similarity_matrix(X: torch.Tensor) -> torch.Tensor:
    """Eq. 14: pairwise cosine similarity. `X`: `(..., N, D)` -> `(..., N, N)`."""
    _validate_node_features(X, "cosine_similarity_matrix")
    x_norm = torch.nn.functional.normalize(X, p=2, dim=-1, eps=_EPS)
    return x_norm @ x_norm.transpose(-2, -1)


def threshold_binarize(sim_matrix: torch.Tensor, threshold: float) -> torch.Tensor:
    """Eq. 15: strict `>` binarization. Same shape/dtype as `sim_matrix` (0.0/1.0 float)."""
    return (sim_matrix > threshold).to(sim_matrix.dtype)


def symmetrize(A: torch.Tensor) -> torch.Tensor:
    """Eq. 16: `Z = (A + A^T) / 2`."""
    return (A + A.transpose(-2, -1)) / 2.0


def normalize_adjacency(Z: torch.Tensor) -> torch.Tensor:
    """Eq. 17: `A~ = D^(-1/2) . Z . D^(-1/2)`.

    `eps` added to node degree before the inverse-sqrt for isolated (degree-0) nodes --
    IMPLEMENTATION-CHOICE numerical-stability step, see module docstring.
    """
    degree = Z.sum(dim=-1)  # (..., N)
    d_inv_sqrt = torch.pow(degree + _EPS, -0.5)
    D_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # (..., N, N)
    return D_inv_sqrt @ Z @ D_inv_sqrt


def attention_reweight(A_norm: torch.Tensor, sim_matrix: torch.Tensor) -> torch.Tensor:
    """Eq. 18 (optional): `A~ = A~ . attention`, `attention = softmax(sim_matrix)` (row-wise)."""
    attention = torch.softmax(sim_matrix, dim=-1)
    return A_norm * attention


def build_adjacency(X: torch.Tensor, config: AdjacencyConfig) -> torch.Tensor:
    """Full Eqs. 14-18 pipeline: node features `X` `(..., N, D)` -> adjacency `(..., N, N)`."""
    _validate_node_features(X, "build_adjacency")
    sim = cosine_similarity_matrix(X)
    A0 = threshold_binarize(sim, config.threshold)
    Z = symmetrize(A0)
    A_norm = normalize_adjacency(Z)
    if config.attention_reweight_enabled:
        A_norm = attention_reweight(A_norm, sim)
    return A_norm


def build_graph(
    prompt_tensor: torch.Tensor,
    mlp_bridge: PromptToNodeMLP,
    node_config: NodeBuilderConfig,
    adjacency_config: AdjacencyConfig,
) -> Graph:
    """End-to-end: prompt tensor -> MLP bridge -> node features -> adjacency -> `Graph`.

    This is the Stage 4 "smoke test" path: `prompt/embedding -> MLP -> graph`. Does not touch
    GIN/ACGA/HGN-EC (later stages) -- returns the clean `(X, A)` interface those stages consume.
    """
    X = build_nodes(prompt_tensor, mlp_bridge, node_config)  # (N=L, D)
    A = build_adjacency(X, adjacency_config)  # (N, N)
    return Graph(X=X, A=A, N=X.shape[-2], feature_dim=X.shape[-1], modality=node_config.modality)

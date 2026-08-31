"""
models/graph/graph_data.py
=============================

Stage 4 — graph data contract.

Not itself a blueprint-named file (mirrors `models/prompts/_common.py`'s precedent: shared
plumbing factored out of `node_builder.py`/`adjacency.py` rather than duplicated between
them). Defines the `Graph` container the task description names directly
(`Graph(X=..., A=..., modality=...)`) that Stage 5's GIN layer will consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

_VALID_MODALITIES = {"text", "vision"}


class GraphShapeError(Exception):
    """Raised when a `Graph`'s `X`/`A` tensors are internally inconsistent."""


@dataclass
class Graph:
    """Node features + adjacency for one modality's graph (Blocker 1: `N = L`).

    X:        node features, `(..., N, D)` -- `(N, D)` unbatched (the normal case: `G`/`GV`
              are model parameters, not per-sample data, so there is one graph per modality,
              shared across a batch) or `(B, N, D)` if a caller batches graph construction.
    A:        normalized adjacency, `(..., N, N)`, matching `X`'s leading (batch) dims.
    N:        node count. Frozen decision (Blocker 1): `N == num_layers` (`L`) per modality.
    feature_dim: `D`, node feature dimension (Fig.1 MLP bridge output width).
    modality: `"text"` or `"vision"`.
    """

    X: torch.Tensor
    A: torch.Tensor
    N: int
    feature_dim: int
    modality: str

    def __post_init__(self) -> None:
        if self.modality not in _VALID_MODALITIES:
            raise GraphShapeError(f"Graph.modality must be one of {_VALID_MODALITIES}, got {self.modality!r}.")
        if self.N <= 0:
            raise GraphShapeError(f"Graph.N must be a positive int, got {self.N!r}.")
        if self.feature_dim <= 0:
            raise GraphShapeError(f"Graph.feature_dim must be a positive int, got {self.feature_dim!r}.")

        if self.X.dim() not in (2, 3):
            raise GraphShapeError(f"Graph.X must have shape (N, D) or (B, N, D), got {tuple(self.X.shape)}.")
        if self.A.dim() not in (2, 3):
            raise GraphShapeError(f"Graph.A must have shape (N, N) or (B, N, N), got {tuple(self.A.shape)}.")
        if self.X.dim() != self.A.dim():
            raise GraphShapeError(
                f"Graph.X and Graph.A must have matching batch-ness: "
                f"X.dim()={self.X.dim()} vs A.dim()={self.A.dim()}."
            )

        if self.X.shape[-2] != self.N:
            raise GraphShapeError(f"Graph.X node dim {self.X.shape[-2]} != Graph.N {self.N}.")
        if self.X.shape[-1] != self.feature_dim:
            raise GraphShapeError(f"Graph.X feature dim {self.X.shape[-1]} != Graph.feature_dim {self.feature_dim}.")
        if tuple(self.A.shape[-2:]) != (self.N, self.N):
            raise GraphShapeError(f"Graph.A shape {tuple(self.A.shape[-2:])} != (N, N) = ({self.N}, {self.N}).")
        if self.X.dim() == 3 and self.X.shape[0] != self.A.shape[0]:
            raise GraphShapeError(
                f"Graph.X batch size {self.X.shape[0]} != Graph.A batch size {self.A.shape[0]}."
            )

        if self.X.device != self.A.device:
            raise GraphShapeError(f"Graph.X device {self.X.device} != Graph.A device {self.A.device}.")
        if self.X.dtype != self.A.dtype:
            raise GraphShapeError(f"Graph.X dtype {self.X.dtype} != Graph.A dtype {self.A.dtype}.")

    @property
    def batch_size(self) -> Optional[int]:
        """`None` for an unbatched `(N, D)` graph, else the leading batch size."""
        return self.X.shape[0] if self.X.dim() == 3 else None

    @property
    def device(self) -> torch.device:
        return self.X.device

    @property
    def dtype(self) -> torch.dtype:
        return self.X.dtype

    def to(self, *args, **kwargs) -> "Graph":
        """Return a copy with `X`/`A` moved via `torch.Tensor.to(...)` (device/dtype)."""
        return Graph(
            X=self.X.to(*args, **kwargs),
            A=self.A.to(*args, **kwargs),
            N=self.N,
            feature_dim=self.feature_dim,
            modality=self.modality,
        )

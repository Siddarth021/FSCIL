"""
models/hgn_ec/hamiltonian.py
================================

Stage 7 — HGN-EC Hamiltonian energy function (Eq. 29) and Hamilton's equations via
autodiff (Eq. 30).

--------------------------------------------------------------------------------------------
REFERENCE TRACEABILITY
--------------------------------------------------------------------------------------------

Decision: `H = H_net(cat(q, p))`, with `H_net` composed of GIN layer(s) (reusing the frozen
    Stage-5 `models.gnn.gin_layer.GIN`, NOT a reimplementation) followed by a final linear
    "readout" projection to width 1 and a mean-pool over nodes, to realize
    `hamiltonian_output_shape = "scalar_per_graph"`.
Source: Eq. 29: `H = H_net(cat(q, p))`; Section IV.D.4, "H_net is an MLP consisting of GIN
    layers and activation functions" (`configs/model/hgn_ec.yaml:
    hamiltonian_net_composition = "gin_layers_plus_activations"`, PAPER-FACT — the GIN-layers
    part; the final linear+mean-pool readout is this module's own addition, see below).
Evidence type: PAPER-FACT (GIN-layers-plus-activations composition) for the GIN stack
    itself; IMPLEMENTATION-CHOICE for the readout layer.
Confidence: Medium — GIN's own `MLP^(k)` (Eq. 21) already supplies "activation functions"
    (GELU) internally, so no separate activation is added between `H_net`'s GIN layers and
    the readout. The readout linear+mean-pool is NOT itself a GIN layer or an activation
    function; Eq. 29 does not literally describe it. It exists because `configs/model/
    hgn_ec.yaml: hamiltonian_output_shape = "scalar_per_graph"` (IMPLEMENTATION-CHOICE,
    `FINAL_RESEARCH_DECISIONS.md` Issue 19 — the paper never states whether `H` is
    scalar-per-graph or per-node) requires *some* node-count-independent reduction, and a
    linear-then-mean-pool readout is the simplest one that does not silently reinterpret
    Eq. 29 as producing a per-node vector. Flagged, not claimed as a paper fact.

Decision: `hnet_gin_layers` (how many stacked `GINLayer`s make up `H_net`'s GIN portion)
    defaults to `1`.
Source: `configs/model/hgn_ec.yaml: hnet_gin_layers = 1` (IMPLEMENTATION-CHOICE, "Fig. 1
    depicts a single 'Hamiltonian GIN Layer' block (singular)").
Evidence type: IMPLEMENTATION-CHOICE (weak figure-based justified inference per
    `FINAL_RESEARCH_DECISIONS.md` Issue 20, not a confirmed paper fact).
Confidence: Low.

Decision: `q_dot = dH/dp`, `p_dot = -dH/dq`, computed via `torch.autograd.grad(H.sum(),
    (q, p), create_graph=True, retain_graph=True)` (summing `H` over the batch dimension,
    since `torch.autograd.grad` requires a scalar or a `grad_outputs` — the sum-then-grad
    pattern gives per-graph gradients identical to differentiating each batch element's own
    scalar `H` independently, since gradients do not mix across independent batch elements
    for this graph).
Source: Eq. 30: `q_dot = dH/dp, p_dot = -dH/dq`; `configs/model/hgn_ec.yaml:
    gradient_computation = "autograd"` (PAPER-FACT).
Evidence type: PAPER-FACT (the equations themselves) + IMPLEMENTATION-CHOICE (the
    sum-for-batched-autograd mechanism, standard PyTorch practice for this exact pattern).
Confidence: High.

Decision: `create_graph=True` — Hamilton's-equation gradients are themselves differentiated
    through the rest of the update (Symplectic Euler + restoration + `L_energy`) during
    backpropagation of the total training loss, so the graph connecting `H` to `q`/`p` must
    survive the `grad()` call itself.
Source: no direct paper statement (the paper does not discuss autograd internals at this
    level); necessary for `L_total`'s gradient (Eq. 34) to reach `H_net`'s and the
    compressor's parameters at all, since Eq. 33's `L_energy` is itself computed from `H`
    values that depend on `q_dot`/`p_dot`.
Evidence type: IMPLEMENTATION-CHOICE (standard PyTorch requirement for a differentiable
    "gradient of a gradient" pipeline), not a paper fact.
Confidence: High (there is no alternative that keeps the pipeline trainable end-to-end).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.gnn.gin_layer import GIN, GINConfig


class HamiltonianNetConfigError(Exception):
    """Raised when a `HamiltonianNetConfig`'s own fields are invalid."""


class HamiltonianShapeError(Exception):
    """Raised when input to `HamiltonianNet`/`hamiltonian_gradients` has an invalid shape."""


@dataclass
class HamiltonianNetConfig:
    """Config for `HamiltonianNet` (Eq. 29).

    compressed_dim: `Dc`, `q`'s (and `p`'s) feature dim; `H_net`'s input is
                    `cat(q, p)`, width `2*Dc`.
    gin_hidden_dim: hidden width of `H_net`'s internal GIN stack. UNRESOLVED in the paper
                    (Section IV.D.4 gives no width); defaults to `compressed_dim` when not
                    supplied (IMPLEMENTATION-CHOICE, mirrors `DiscriminatorConfig.hidden_dim`
                    's precedent of defaulting an unspecified width to an already-fixed
                    surrounding dimension).
    num_gin_layers: forwarded to `GINConfig.num_layers`; PAPER-FACT-adjacent default `1`
                    (`configs/model/hgn_ec.yaml: hnet_gin_layers`, see module docstring).
    mlp_hidden_dim: forwarded to `GINConfig.mlp_hidden_dim` (each `GINLayer`'s internal MLP
                    width; `None` -> defaults to `gin_hidden_dim` per `GINLayerConfig`).
    eps_init:       forwarded to `GINConfig.eps_init`.
    provenance:     free-form dict.
    """

    compressed_dim: int
    gin_hidden_dim: Optional[int] = None
    num_gin_layers: int = 1
    mlp_hidden_dim: Optional[int] = None
    eps_init: float = 0.0
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.compressed_dim, int) or self.compressed_dim <= 0:
            raise HamiltonianNetConfigError(
                f"HamiltonianNetConfig.compressed_dim must be a positive int, got {self.compressed_dim!r}."
            )
        if self.gin_hidden_dim is None:
            self.gin_hidden_dim = self.compressed_dim
        elif not isinstance(self.gin_hidden_dim, int) or self.gin_hidden_dim <= 0:
            raise HamiltonianNetConfigError(
                f"HamiltonianNetConfig.gin_hidden_dim must be a positive int or None, got {self.gin_hidden_dim!r}."
            )
        if not isinstance(self.num_gin_layers, int) or self.num_gin_layers <= 0:
            raise HamiltonianNetConfigError(
                f"HamiltonianNetConfig.num_gin_layers must be a positive int, got {self.num_gin_layers!r}."
            )


class HamiltonianNet(nn.Module):
    """Eq. 29: `H = H_net(cat(q, p))`.

    `H_net` = GIN layer(s) (reusing frozen Stage-5 `GIN.forward_tensors`, consistent with
    `configs/model/hgn_ec.yaml: hamiltonian_net_composition`) + a linear readout to width 1
    + mean-pool over nodes, realizing `hamiltonian_output_shape = "scalar_per_graph"` (see
    module docstring's "Reference traceability").
    """

    def __init__(self, config: HamiltonianNetConfig):
        super().__init__()
        self.config = config
        self.gin = GIN(
            GINConfig(
                input_dim=2 * config.compressed_dim,
                num_layers=config.num_gin_layers,
                hidden_dim=config.gin_hidden_dim,
                mlp_hidden_dim=config.mlp_hidden_dim,
                eps_init=config.eps_init,
            )
        )
        self.readout = nn.Linear(config.gin_hidden_dim, 1)

    def forward(self, q: torch.Tensor, p: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """`q`, `p`: `(..., N, Dc)`, `A`: `(..., N, N)` -> `H`: `(...,)` (scalar per graph).

        `(...)` is empty (a 0-dim scalar tensor) for an unbatched `(N, Dc)` input, or `(B,)`
        for a batched `(B, N, Dc)` input.
        """
        if q.dim() not in (2, 3) or p.dim() not in (2, 3):
            raise HamiltonianShapeError(
                f"HamiltonianNet: expected q, p of shape (N, Dc) or (B, N, Dc); got q={tuple(q.shape)}, p={tuple(p.shape)}."
            )
        if q.shape != p.shape:
            raise HamiltonianShapeError(f"HamiltonianNet: q and p must have identical shape; got {tuple(q.shape)} vs {tuple(p.shape)}.")
        if q.shape[-1] != self.config.compressed_dim:
            raise HamiltonianShapeError(
                f"HamiltonianNet: q/p last dim {q.shape[-1]} != config.compressed_dim {self.config.compressed_dim}."
            )
        if A.dim() not in (2, 3):
            raise HamiltonianShapeError(f"HamiltonianNet: expected A of shape (N, N) or (B, N, N), got {tuple(A.shape)}.")
        if A.shape[-1] != q.shape[-2] or A.shape[-1] != A.shape[-2]:
            raise HamiltonianShapeError(
                f"HamiltonianNet: A must be square (N, N) matching q/p's node dim; got node dim "
                f"{q.shape[-2]}, A shape {tuple(A.shape[-2:])}."
            )

        combined = torch.cat([q, p], dim=-1)  # Eq. 29: cat(q, p), (..., N, 2*Dc)
        node_features = self.gin.forward_tensors(combined, A)  # (..., N, gin_hidden_dim)
        node_energy = self.readout(node_features).squeeze(-1)  # (..., N)
        H = node_energy.mean(dim=-1)  # (...,) — scalar per graph
        return H

    def extra_repr(self) -> str:
        return f"compressed_dim={self.config.compressed_dim}, gin_hidden_dim={self.config.gin_hidden_dim}"


def hamiltonian_gradients(
    H: torch.Tensor,
    q: torch.Tensor,
    p: torch.Tensor,
    *,
    create_graph: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. 30: `q_dot = dH/dp`, `p_dot = -dH/dq`, via `torch.autograd.grad`.

    `H`: `(...,)` (scalar per graph, or a single 0-dim scalar). `q`, `p`: `(..., N, Dc)`.
    Returns `(q_dot, p_dot)`, each matching `q`'s/`p`'s shape.

    `H` is summed before calling `torch.autograd.grad` so a batched `H` (`(B,)`) can be
    differentiated in one call; because each batch element's `H` depends only on that same
    batch element's `q`/`p` (no cross-batch mixing anywhere in `HamiltonianNet`), summing
    first and differentiating once gives per-batch-element gradients identical to
    differentiating each element's own scalar `H` separately (see module docstring).
    """
    if not q.requires_grad:
        raise HamiltonianShapeError("hamiltonian_gradients: q must have requires_grad=True (autodiff w.r.t. q is required by Eq. 30).")
    if not p.requires_grad:
        raise HamiltonianShapeError("hamiltonian_gradients: p must have requires_grad=True (autodiff w.r.t. p is required by Eq. 30).")

    dHdq, dHdp = torch.autograd.grad(
        outputs=H.sum(),
        inputs=(q, p),
        create_graph=create_graph,
        retain_graph=True,
    )
    q_dot = dHdp  # Eq. 30: q_dot = dH/dp
    p_dot = -dHdq  # Eq. 30: p_dot = -dH/dq
    return q_dot, p_dot

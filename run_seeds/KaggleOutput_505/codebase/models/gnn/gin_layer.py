"""
models/gnn/gin_layer.py
==========================

Stage 5 — Graph Isomorphism Network (GIN), Eqs. 13, 19-21.

Consumes the Stage-4 `Graph(X, A, N, feature_dim, modality)` contract
(`models/graph/graph_data.py`) exactly as produced by `models/graph/adjacency.py`.
Does NOT reconstruct, threshold, symmetrize, or otherwise modify `A` -- that pipeline is
frozen (Stage 4, Eqs. 14-18) and out of scope here.

--------------------------------------------------------------------------------------------
REFERENCE TRACEABILITY (nontrivial GIN design decisions)
--------------------------------------------------------------------------------------------

Decision: Eq. 19's neighbor aggregation `agg_v = sum_{u in N(v)} h_u^(k-1)` is implemented as
    a dense matmul `agg = A @ X`.
Source: Eq. 19; the Stage-4 adjacency (Eqs. 14-18) is the only adjacency this module is
    permitted to consume ("Use the Stage-4 adjacency exactly as provided" — Stage 5 task
    spec).
Evidence type: PAPER-FACT (the aggregation *operation*: a matrix multiply against the
    adjacency realizes "sum over neighbors" for any adjacency matrix, weighted or 0/1).
Confidence: High.

Decision: `A` is used exactly as received (including its diagonal), with NO self-loop
    masking/zeroing performed inside `GINLayer`.
Source: Stage 5 task spec, "ADJACENCY" section: "Do NOT reconstruct or modify adjacency
    inside GIN." `models/graph/adjacency.py`'s cosine-similarity construction (Eq. 14) gives
    every node `cos(x_i, x_i) = 1 > 0.8`, so `A`'s pre-normalization diagonal is always 1
    before Eq. 17's degree normalization rescales it — i.e. the Stage-4 adjacency this module
    receives is not guaranteed self-loop-free.
Evidence type: IMPLEMENTATION-CHOICE (compliance with an explicit Stage-5 task constraint,
    not a paper fact). This creates a known tension with standard GIN (Xu et al., 2018 —
    `configs/model/gin.yaml`'s own reference), where `N(v)` conventionally excludes `v`
    itself and the self-contribution is handled *only* via the separate `(1+epsilon)*h_v`
    term (Eq. 20) — under that convention, a self-loop already present in `A` would double
    count the self-feature. The paper's own Eqs. 14-18/19 never address this, and no
    self-loop-handling sentence exists anywhere in the source text.
Confidence: Low. See "Known limitations" in `docs/implementation_progress.md`'s Stage 5
    section — this must be revisited if author code or errata surface. Not silently
    "fixed" here by masking the diagonal, since that would itself be an undocumented
    adjacency modification the task spec forbids.

Decision: Self-feature contribution `(1+epsilon^(k))*h_v^(k-1)` and neighbor aggregation
    `agg_v` are combined by elementwise addition before the MLP (Eq. 20), exactly as written.
Source: Eq. 20.
Evidence type: PAPER-FACT.
Confidence: High.

Decision: `epsilon` is one learnable scalar `nn.Parameter` per `GINLayer` instance (not
    shared across the 4 stacked layers).
Source: `configs/model/gin.yaml: epsilon_shared_across_layers = false`; Eq. 13's
    `epsilon^(k)` is superscripted by layer index `k`.
Evidence type: PAPER-FACT.
Confidence: High.

Decision: `epsilon`'s initial value is `0.0`.
Source: not stated anywhere in the paper (Eq. 13 only says `epsilon^(k)` is learnable).
Evidence type: IMPLEMENTATION-CHOICE. `0.0` reduces Eq. 20 to `combined_v = h_v + agg_v`
    at initialization (the original GIN paper's own common initialization), a neutral,
    commonly-used default -- never claimed as a paper fact. Exposed as `GINLayerConfig.
    eps_init`, overridable.
Confidence: Low (no paper evidence either way).

Decision: `MLP^(k)` (Eq. 21) = `Linear(in, mlp_hidden) -> BatchNorm -> GELU -> Linear(
    mlp_hidden, out)`.
Source: `configs/model/gin.yaml: mlp_composition = "linear_batchnorm_gelu"`
    (JUSTIFIED_INFERENCE, reusing Eq. 22's ACGA-encoder MLP composition -- the paper's only
    fully-specified MLP recipe -- as the best-evidence default for the "main" GIN's own
    unspecified MLP internals; `FINAL_RESEARCH_DECISIONS.md` Issue 9).
Evidence type: JUSTIFIED-INFERENCE (inherited from Stage-1/4's own already-recorded
    inference; not re-derived here).
Confidence: Medium (per Issue 9's own stated confidence).

Decision: `mlp_num_linear_layers` (how many `Linear` layers make up `MLP^(k)`) defaults to
    exactly 2 (one before, one after the BatchNorm+GELU), matching Eq. 22's literal
    Linear+BatchNorm+GELU phrasing read as a single Linear/BN/GELU block plus the necessary
    output-projecting Linear.
Source: `configs/model/gin.yaml: mlp_num_linear_layers = null (UNRESOLVED)` -- the paper
    never states this number for the main GIN or for Eq. 22's ACGA-encoder MLP.
Evidence type: IMPLEMENTATION-CHOICE (a runnable default is required; the value is NEVER
    claimed as paper-derived). `GINLayerConfig.mlp_num_linear_layers` is configurable and
    documented as such; `mlp_num_linear_layers != 2` raises `NotImplementedError` rather
    than silently guessing a different composition.
Confidence: Low.

Decision: No activation/normalization is applied *between* stacked `GINLayer` instances
    beyond what each layer's own `MLP^(k)` already applies internally (Eq. 21's `MLP^(k)`
    IS the complete per-layer transform).
Source: Eqs. 13/19-21 describe one layer's complete update rule; nothing in Section IV.B or
    the equation block describes an inter-layer operation.
Evidence type: PAPER-FACT (absence of any stated inter-layer step).
Confidence: High.

Decision: Layer-1 maps `input_dim = D` (the Stage-4 `Graph.feature_dim`, itself UNRESOLVED
    in `configs/model/graph.yaml: node_feature_dim_D` and never defaulted here) to
    `hidden_dim = 16`; layers 2-4 map `16 -> 16`.
Source: `configs/model/gin.yaml: num_layers = 4, hidden_dim = 16` (Section V.B, PAPER-FACT,
    "the number of neurons in the hidden layer being 16"); Part 2 module-mapping table of
    `FINAL_IMPLEMENTATION_BLUEPRINT.md`: `GINLayer` output shape `(N, D_hidden=16)`.
Evidence type: PAPER-FACT (layer count, hidden width) + JUSTIFIED-INFERENCE (that "hidden
    dim 16" names every layer's *output* width, including the first layer's, since the
    blueprint's own table states the GIN's overall output shape as `(N, D_hidden=16)` with
    no separate first-layer width given).
Confidence: Medium.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.graph.graph_data import Graph, GraphShapeError

_SUPPORTED_MLP_COMPOSITIONS = {"linear_batchnorm_gelu"}


class GINLayerConfigError(Exception):
    """Raised when a `GINLayerConfig`'s own fields are invalid."""


class GINLayerShapeError(Exception):
    """Raised when input to `GINLayer`/`GIN` has an invalid shape."""


class GINConfigError(Exception):
    """Raised when a `GINConfig`'s own fields are invalid."""


@dataclass
class GINLayerConfig:
    """Config for a single `GINLayer` (Eqs. 13, 19-21).

    input_dim:  node feature dim consumed by this layer (`h^(k-1)`'s last dim).
    output_dim: node feature dim produced by this layer (`h^(k)`'s last dim).
    mlp_hidden_dim: internal width of `MLP^(k)`'s hidden layer. UNRESOLVED in the paper for
                the main GIN (only ACGA's encoder MLP composition -- not width -- is given);
                defaults to `output_dim` when not supplied (IMPLEMENTATION-CHOICE, documented
                in the module docstring, never silently claimed as a paper fact).
    mlp_composition: locked to `"linear_batchnorm_gelu"` -- the only implemented composition,
                matching `configs/model/gin.yaml: mlp_composition` (JUSTIFIED_INFERENCE).
    mlp_num_linear_layers: fixed at 2 (`Linear -> BatchNorm -> GELU -> Linear`) -- see module
                docstring's "Reference traceability" entry. Kept as an explicit field (not a
                bare constant) so a future revisit of `configs/model/gin.yaml:
                mlp_num_linear_layers` (currently UNRESOLVED) only touches this module.
    eps_init:   initial value of the learnable `epsilon^(k)` scalar (Eq. 13). Not paper-
                specified; defaults to `0.0` (IMPLEMENTATION-CHOICE, see module docstring).
    provenance: free-form dict, mirrors `MLPBridgeConfig.provenance` / `AdjacencyConfig.
                provenance`.
    """

    input_dim: int
    output_dim: int
    mlp_hidden_dim: Optional[int] = None
    mlp_composition: str = "linear_batchnorm_gelu"
    mlp_num_linear_layers: int = 2
    eps_init: float = 0.0
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("input_dim", "output_dim"):
            val = getattr(self, name)
            if not isinstance(val, int) or val <= 0:
                raise GINLayerConfigError(f"GINLayerConfig.{name} must be a positive int, got {val!r}.")
        if self.mlp_hidden_dim is None:
            self.mlp_hidden_dim = self.output_dim
        elif not isinstance(self.mlp_hidden_dim, int) or self.mlp_hidden_dim <= 0:
            raise GINLayerConfigError(
                f"GINLayerConfig.mlp_hidden_dim must be a positive int or None, got {self.mlp_hidden_dim!r}."
            )
        if self.mlp_composition not in _SUPPORTED_MLP_COMPOSITIONS:
            raise GINLayerConfigError(
                f"GINLayerConfig.mlp_composition must be one of {_SUPPORTED_MLP_COMPOSITIONS} "
                f"(configs/model/gin.yaml frozen default); got {self.mlp_composition!r}."
            )
        if self.mlp_num_linear_layers != 2:
            raise NotImplementedError(
                "GINLayerConfig.mlp_num_linear_layers != 2 is not implemented -- "
                "configs/model/gin.yaml: mlp_num_linear_layers is UNRESOLVED in the paper; "
                "only the 2-linear-layer (Linear->BatchNorm->GELU->Linear) default documented "
                "in gin_layer.py's module docstring is implemented. Do not silently guess a "
                "different composition."
            )


class GINLayer(nn.Module):
    """A single GIN layer (Eqs. 13, 19-21).

    ``h_v^(k) = MLP^(k)( (1 + epsilon^(k)) * h_v^(k-1) + sum_{u in N(v)} h_u^(k-1) )``

    Consumes node features `X` and an adjacency `A` exactly as produced by Stage 4
    (`models/graph/adjacency.py`); performs no adjacency reconstruction/modification.
    """

    def __init__(self, config: GINLayerConfig):
        super().__init__()
        self.config = config
        # Eq. 13: one learnable epsilon^(k) per layer instance (not shared across layers,
        # per configs/model/gin.yaml: epsilon_shared_across_layers = false).
        self.epsilon = nn.Parameter(torch.tensor(float(config.eps_init)))
        self.mlp = nn.Sequential(
            nn.Linear(config.input_dim, config.mlp_hidden_dim),
            nn.BatchNorm1d(config.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(config.mlp_hidden_dim, config.output_dim),
        )

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """`X`: `(..., N, input_dim)`, `A`: `(..., N, N)` -> `(..., N, output_dim)`.

        Accepts an unbatched `(N, D)`/`(N, N)` pair or a batched `(B, N, D)`/`(B, N, N)` pair,
        matching `Graph.X`/`Graph.A`'s own batch-optionality (`graph_data.py`).
        """
        if X.dim() not in (2, 3):
            raise GINLayerShapeError(f"GINLayer: expected X of shape (N, D) or (B, N, D), got {tuple(X.shape)}.")
        if A.dim() not in (2, 3):
            raise GINLayerShapeError(f"GINLayer: expected A of shape (N, N) or (B, N, N), got {tuple(A.shape)}.")
        if X.dim() != A.dim():
            raise GINLayerShapeError(
                f"GINLayer: X and A must have matching batch-ness: X.dim()={X.dim()} vs A.dim()={A.dim()}."
            )
        if X.shape[-1] != self.config.input_dim:
            raise GINLayerShapeError(
                f"GINLayer: X last dim {X.shape[-1]} does not match config.input_dim {self.config.input_dim}."
            )
        if X.shape[-2] != A.shape[-1] or A.shape[-1] != A.shape[-2]:
            raise GINLayerShapeError(
                f"GINLayer: A must be square (N, N) matching X's node dim; got X node dim "
                f"{X.shape[-2]}, A shape {tuple(A.shape[-2:])}."
            )
        if X.dim() == 3 and X.shape[0] != A.shape[0]:
            raise GINLayerShapeError(f"GINLayer: X batch size {X.shape[0]} != A batch size {A.shape[0]}.")

        # Eq. 19: neighbor aggregation, realized as a matmul against the Stage-4 adjacency
        # (used exactly as provided -- no self-loop masking, see module docstring).
        agg = A @ X  # (..., N, input_dim)

        # Eq. 20: self-feature contribution + neighbor aggregation.
        combined = (1.0 + self.epsilon) * X + agg  # (..., N, input_dim)

        # Eq. 21: MLP transform. BatchNorm1d expects (batch, C), so flatten any leading dims.
        leading_shape = combined.shape[:-1]
        flat = combined.reshape(-1, combined.shape[-1])
        out_flat = self.mlp(flat)
        out = out_flat.reshape(*leading_shape, self.config.output_dim)
        return out

    def extra_repr(self) -> str:
        return f"input_dim={self.config.input_dim}, output_dim={self.config.output_dim}"


@dataclass
class GINConfig:
    """Config for the `GIN` stack (4 layers, hidden dim 16 -- Section V.B, PAPER-FACT).

    input_dim:  `D`, the incoming `Graph.feature_dim` (Stage-4 output). UNRESOLVED in the
                paper (`configs/model/graph.yaml: node_feature_dim_D`); caller-supplied,
                never defaulted here -- matching `MLPBridgeConfig`'s convention for
                UNRESOLVED dims.
    num_layers: PAPER-FACT default `4` (Section V.B).
    hidden_dim: PAPER-FACT default `16` (Section V.B). Every layer's output width, including
                the first layer's (see module docstring's "Reference traceability").
    mlp_hidden_dim: forwarded to every `GINLayerConfig`; `None` -> defaults to `hidden_dim`
                per-layer (see `GINLayerConfig`).
    eps_init:   forwarded to every `GINLayerConfig`.
    provenance: free-form dict.
    """

    input_dim: int
    num_layers: int = 4
    hidden_dim: int = 16
    mlp_hidden_dim: Optional[int] = None
    eps_init: float = 0.0
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.input_dim, int) or self.input_dim <= 0:
            raise GINConfigError(f"GINConfig.input_dim must be a positive int, got {self.input_dim!r}.")
        if not isinstance(self.num_layers, int) or self.num_layers <= 0:
            raise GINConfigError(f"GINConfig.num_layers must be a positive int, got {self.num_layers!r}.")
        if not isinstance(self.hidden_dim, int) or self.hidden_dim <= 0:
            raise GINConfigError(f"GINConfig.hidden_dim must be a positive int, got {self.hidden_dim!r}.")


class GIN(nn.Module):
    """The full `num_layers`-deep GIN stack (Section V.B: 4 layers, hidden dim 16).

    Consumes/produces the Stage-4 `Graph` contract directly (`models/graph/graph_data.py`):
    `GIN.forward(graph)` returns a new `Graph` with the same `A`, `N`, and `modality`, and
    `X` replaced by the stack's final node features (`feature_dim = hidden_dim`).

    Layer 1 maps `config.input_dim -> config.hidden_dim`; layers 2..num_layers map
    `hidden_dim -> hidden_dim` (see module docstring's "Reference traceability").
    """

    def __init__(self, config: GINConfig):
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.num_layers):
            in_dim = config.input_dim if i == 0 else config.hidden_dim
            layer_config = GINLayerConfig(
                input_dim=in_dim,
                output_dim=config.hidden_dim,
                mlp_hidden_dim=config.mlp_hidden_dim,
                eps_init=config.eps_init,
            )
            layers.append(GINLayer(layer_config))
        self.layers = nn.ModuleList(layers)

    def forward_tensors(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Tensor-level entry point (used by ACGA's encoder / HGN-EC's H_net, which reuse
        `GINLayer`/`GIN` per `configs/model/gin.yaml`'s module docstring on equations
        appearing twice) -- `X`: `(..., N, input_dim)`, `A`: `(..., N, N)` ->
        `(..., N, hidden_dim)`.
        """
        out = X
        for layer in self.layers:
            out = layer(out, A)
        return out

    def forward(self, graph: Graph) -> Graph:
        """Consume the Stage-4 `Graph` contract; return an updated `Graph`.

        Preserves: batch dimension (if any, via `Graph.X`'s own optional leading dim), node
        count `N`, and `modality`. `A` is passed through unchanged (GIN never modifies
        adjacency). `feature_dim` on the returned `Graph` is `self.config.hidden_dim`,
        explicitly exposing the resulting node feature dimension.
        """
        if graph.feature_dim != self.config.input_dim:
            raise GINLayerShapeError(
                f"GIN: graph.feature_dim {graph.feature_dim} does not match "
                f"config.input_dim {self.config.input_dim}."
            )
        X_out = self.forward_tensors(graph.X, graph.A)
        return Graph(
            X=X_out,
            A=graph.A,
            N=graph.N,
            feature_dim=self.config.hidden_dim,
            modality=graph.modality,
        )

    def extra_repr(self) -> str:
        return f"num_layers={self.config.num_layers}, hidden_dim={self.config.hidden_dim}"

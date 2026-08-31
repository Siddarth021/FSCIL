"""
models/graph/node_builder.py
===============================

Stage 4 — graph node construction (Blocker 1: N = L).

Behavior: Graph node count `N` per modality equals `L` (the number of Transformer/prompt
    layers), NOT `M` (number of prompt tokens per layer) or `L*M`. Node features `X_node` are
    obtained by decomposing the `(L, 1, d)` prompt tensor along its `L` axis into `L` per-layer
    prompt vectors, then projecting each through that modality's `PromptToNodeMLP`
    (`models/prompts/mlp_bridge.py`) into `R^D`.
Source: Section IV.B: "node feature matrices X in R^{N x D}"; Section V.D.4 confirms M=1.
    No paper sentence equates N to M, L, or L*M.
Evidence type: IMPLEMENTATION-CHOICE (FINAL_IMPLEMENTATION_BLUEPRINT.md Blocker 1;
    `configs/model/graph.yaml: num_nodes_mode = per_layer`). Rejection of the degenerate
    `N=M=1` alternative is JUSTIFIED-INFERENCE; the specific choice `N=L` beyond
    "non-degenerate" is not otherwise evidenced. Confidence: Low. This must be revisited if
    author code or errata surface (do not present as PAPER-FACT).
Implementation note: `build_nodes` requires `prompt_tensor.shape[1] == 1` (i.e. `M == 1`) and
    raises `NodeBuilderError` otherwise -- the `N=L` decomposition is only well-defined for the
    reported `M=1` configuration (see Blocker 1's own stated rationale); it does not silently
    guess a decomposition for `M > 1`. `num_nodes_mode` is kept as an explicit config field
    (not a hard-coded constant) so a future revisit of Blocker 1 changes one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch

from models.prompts.mlp_bridge import PromptToNodeMLP

SUPPORTED_NUM_NODES_MODES = {"per_layer"}  # only mode implemented; Blocker 1 frozen decision.


class NodeBuilderConfigError(Exception):
    """Raised when a `NodeBuilderConfig`'s own fields are invalid."""


class NodeBuilderShapeError(Exception):
    """Raised when the prompt tensor passed to `build_nodes` has an invalid shape."""


@dataclass
class NodeBuilderConfig:
    """Config for `build_nodes`.

    num_nodes_mode: only `"per_layer"` (N=L) is implemented -- the frozen Blocker 1 decision.
                    Kept as an explicit field (not a bare constant) so a future revisit only
                    touches this module, not every call site.
    modality:       `"text"` or `"vision"`, forwarded to the resulting node-feature metadata
                    (the `Graph` container itself is built downstream in `adjacency.py`, but
                    keeping the tag here makes `build_nodes`'s own errors self-describing).
    provenance:     free-form dict, mirrors `MLPBridgeConfig.provenance`.
    """

    num_nodes_mode: str = "per_layer"
    modality: str = "text"
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_nodes_mode not in SUPPORTED_NUM_NODES_MODES:
            raise NodeBuilderConfigError(
                f"NodeBuilderConfig.num_nodes_mode must be one of {SUPPORTED_NUM_NODES_MODES} "
                f"(Blocker 1 frozen decision); got {self.num_nodes_mode!r}."
            )
        if self.modality not in {"text", "vision"}:
            raise NodeBuilderConfigError(f"NodeBuilderConfig.modality must be 'text' or 'vision', got {self.modality!r}.")


def build_nodes(prompt_tensor: torch.Tensor, mlp_bridge: PromptToNodeMLP, config: NodeBuilderConfig) -> torch.Tensor:
    """Build `X_node in R^{N=L, D}` from a `(L, M, d)` prompt tensor (Blocker 1: N=L, M=1).

    Args:
        prompt_tensor: `(L, M, d)` learnable prompt parameter (`G` or `GV`). `M` must be 1 --
            see module docstring.
        mlp_bridge: this modality's `PromptToNodeMLP` (`R^d -> R^D`).
        config: `NodeBuilderConfig` (`num_nodes_mode` must be `"per_layer"`).

    Returns:
        `X_node`: `(L, D)` node feature tensor, `L == prompt_tensor.shape[0] == N`.
    """
    if prompt_tensor.dim() != 3:
        raise NodeBuilderShapeError(
            f"build_nodes: expected prompt_tensor of shape (L, M, d), got {tuple(prompt_tensor.shape)}."
        )
    L, M, d = prompt_tensor.shape
    if M != 1:
        raise NodeBuilderShapeError(
            f"build_nodes: num_nodes_mode='per_layer' (Blocker 1) requires M=1 for a "
            f"well-defined L-node decomposition of the prompt tensor; got M={M}. "
            f"(This is the reported final-model M -- see configs/model/prompts.yaml.)"
        )
    if d != mlp_bridge.config.input_dim:
        raise NodeBuilderShapeError(
            f"build_nodes: prompt_tensor last dim {d} does not match mlp_bridge.input_dim "
            f"{mlp_bridge.config.input_dim}."
        )

    per_layer_prompts = prompt_tensor.squeeze(1)  # (L, M=1, d) -> (L, d)
    x_node = mlp_bridge(per_layer_prompts)  # (L, D)
    return x_node

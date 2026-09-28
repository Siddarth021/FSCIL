"""
models/prompts/mlp_bridge.py
===============================

Stage 4 — Fig.1 MLP bridge (prompt space -> graph node-feature space).

Behavior: Two unlabeled "MLP" boxes in Fig. 1 sit between "Learnable Prompts" and the GIN
    modules, one per modality, projecting each layer's prompt vector into the node-feature
    space consumed by graph construction (Eq. 13's `X`).
Source: Fig. 1; no sentence in Section IV names or describes them.
Evidence type: IMPLEMENTATION-CHOICE (FINAL_IMPLEMENTATION_BLUEPRINT.md Blocker 3).
Implementation note: `PromptToNodeMLP` implements exactly this: `R^d -> R^D`, two linear
    layers with a GELU in between, mirroring Eq. 22's ACGA-encoder MLP composition (the
    paper's only fully-specified MLP recipe) per Blocker 3's stated concrete default. This is
    NOT a paper fact -- confidence Low, per `configs/model/mlp_bridge.yaml`.

Behavior: One MLP instance per modality (text, vision), not a single shared/cross-modal
    bridge.
Source: Fig. 1 shows two separate MLP boxes.
Evidence type: IMPLEMENTATION-CHOICE (`configs/model/mlp_bridge.yaml: shared_across_modalities
    = false`).
Implementation note: this module makes no attempt to share weights between the two calling
    sites -- callers construct two independent `PromptToNodeMLP` instances (see
    `models/graph/node_builder.py`).

Behavior: hidden width of the MLP bridge and its output dimension `D` (node feature dim).
Source: not specified anywhere in the paper (figure-only existence, per Blocker 3).
Evidence type: UNRESOLVED (paper). This module never silently invents a number for either:
    `MLPBridgeConfig.hidden_dim` and `.output_dim` are required constructor arguments with no
    default, so a caller must supply them explicitly (from `configs/model/mlp_bridge.yaml` /
    `configs/model/graph.yaml`, both of which record these as UNRESOLVED with `value: null`),
    exactly as Stage 2's `CLIPConfig` refuses to silently pick UNRESOLVED CLIP dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn


class MLPBridgeConfigError(Exception):
    """Raised when an `MLPBridgeConfig`'s own fields are invalid."""


class MLPBridgeShapeError(Exception):
    """Raised when input to `PromptToNodeMLP` has an invalid shape."""


@dataclass
class MLPBridgeConfig:
    """Config for one modality's `PromptToNodeMLP`.

    input_dim:  `d` -- prompt embedding dimension (must match the corresponding
                `TextPromptConfig`/`VisionPromptConfig.prompt_dim`).
    hidden_dim: internal width. UNRESOLVED in the paper (`configs/model/mlp_bridge.yaml`) --
                caller-supplied, never defaulted here.
    output_dim: `D` -- node feature dimension. UNRESOLVED in the paper
                (`configs/model/graph.yaml: node_feature_dim_D`) -- caller-supplied.
    dropout:    not paper-specified; defaults to 0 (no dropout), overridable.
    provenance: free-form dict recording where `hidden_dim`/`output_dim` actually came from
                (e.g. `{"hidden_dim": "TEST_OVERRIDE", "output_dim": "TEST_OVERRIDE"}`), so a
                caller reading a real config can distinguish a resolved paper/choice value
                from a synthetic test value -- mirrors `CLIPConfig.dim_provenance`.
    """

    input_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float = 0.0
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("input_dim", "hidden_dim", "output_dim"):
            val = getattr(self, name)
            if not isinstance(val, int) or val <= 0:
                raise MLPBridgeConfigError(f"MLPBridgeConfig.{name} must be a positive int, got {val!r}.")
        if not (0.0 <= self.dropout < 1.0):
            raise MLPBridgeConfigError(f"MLPBridgeConfig.dropout must be in [0, 1), got {self.dropout!r}.")


class PromptToNodeMLP(nn.Module):
    """`R^d -> R^D` projection: `Linear(d, hidden) -> GELU -> Linear(hidden, D)`.

    One instance = one modality's Fig.1 MLP block (Blocker 3). Construct two independent
    instances (one per modality) rather than sharing weights, per
    `configs/model/mlp_bridge.yaml: shared_across_modalities = false`.
    """

    def __init__(self, config: MLPBridgeConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(..., input_dim)` -> `(..., output_dim)`. Any number of leading dims."""
        if x.shape[-1] != self.config.input_dim:
            raise MLPBridgeShapeError(
                f"PromptToNodeMLP: last dim {x.shape[-1]} does not match input_dim "
                f"{self.config.input_dim}."
            )
        return self.net(x)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.config.input_dim}, hidden_dim={self.config.hidden_dim}, "
            f"output_dim={self.config.output_dim}"
        )

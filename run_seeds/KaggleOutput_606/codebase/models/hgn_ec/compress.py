"""
models/hgn_ec/compress.py
=============================

Stage 7 — HGN-EC feature compression (Eq. 28).

--------------------------------------------------------------------------------------------
REFERENCE TRACEABILITY
--------------------------------------------------------------------------------------------

Decision: `compressed = W_compress . state + b_compress` implemented as a single
    `nn.Linear(state_dim, compressed_dim)`.
Source: Eq. 28; `configs/model/hgn_ec.yaml: compression_num_layers = 1` (PAPER-FACT, "a
    single linear layer").
Evidence type: PAPER-FACT.
Confidence: High.

Decision: `compressed_dim` (`Dc`) has NO default and must be supplied by the caller.
Source: `configs/model/hgn_ec.yaml: compressed_dim_Dc = UNRESOLVED` — "Eq. 28 gives the
    linear-layer form but no numeric output width."
Evidence type: UNRESOLVED (paper gap) — the resulting code-level requirement (no silent
    default) is an IMPLEMENTATION-CHOICE mirroring `ACGAEncoderConfig.latent_dim`'s and
    `GINConfig.input_dim`'s established precedent for UNRESOLVED dims elsewhere in this
    codebase.
Confidence: N/A (this is a required-argument policy, not a numeric guess).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn


class CompressorConfigError(Exception):
    """Raised when a `FeatureCompressorConfig`'s own fields are invalid."""


class CompressorShapeError(Exception):
    """Raised when input to `FeatureCompressor` has an invalid shape."""


@dataclass
class FeatureCompressorConfig:
    """Config for `FeatureCompressor` (Eq. 28).

    state_dim:      `2*D`, the incoming `state`'s feature dim (Eq. 27's output width).
                    Caller-supplied (derived from the Stage-5 GIN's `D`), never defaulted.
    compressed_dim: `Dc`, UNRESOLVED in the paper (`configs/model/hgn_ec.yaml:
                    compressed_dim_Dc`). Required, no default — see module docstring.
    provenance:     free-form dict.
    """

    state_dim: int
    compressed_dim: int
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("state_dim", "compressed_dim"):
            val = getattr(self, name)
            if not isinstance(val, int) or val <= 0:
                raise CompressorConfigError(f"FeatureCompressorConfig.{name} must be a positive int, got {val!r}.")


class FeatureCompressor(nn.Module):
    """Eq. 28: `compressed = W_compress . state + b_compress` — a single linear layer."""

    def __init__(self, config: FeatureCompressorConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.state_dim, config.compressed_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """`state`: `(..., N, state_dim)` -> `(..., N, compressed_dim)`."""
        if state.dim() not in (2, 3):
            raise CompressorShapeError(f"FeatureCompressor: expected state of shape (N, D) or (B, N, D), got {tuple(state.shape)}.")
        if state.shape[-1] != self.config.state_dim:
            raise CompressorShapeError(
                f"FeatureCompressor: state last dim {state.shape[-1]} != config.state_dim {self.config.state_dim}."
            )
        return self.linear(state)

    def extra_repr(self) -> str:
        return f"state_dim={self.config.state_dim}, compressed_dim={self.config.compressed_dim}"

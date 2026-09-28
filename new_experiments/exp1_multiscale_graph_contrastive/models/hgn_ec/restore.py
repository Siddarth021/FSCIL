"""
models/hgn_ec/restore.py
============================

Stage 7 — HGN-EC state restoration (Eq. 32).

--------------------------------------------------------------------------------------------
REFERENCE TRACEABILITY
--------------------------------------------------------------------------------------------

Decision: `q_final = W_restore . q_new + b_restore` implemented as a single
    `nn.Linear(compressed_dim, restored_dim)`.
Source: Eq. 32; `configs/model/hgn_ec.yaml: restoration_num_layers = 1` (PAPER-FACT, "a
    single linear layer, 'inverse in spirit' to Eq. 28's compression").
Evidence type: PAPER-FACT.
Confidence: High.

Decision: `restored_dim` defaults to targeting the PRE-COMPRESSION `state` dimensionality
    (`2*D`, Eq. 27's output width) rather than the original per-layer prompt dimensionality
    `d`.
Source: `configs/model/hgn_ec.yaml: restoration_target_dim = "state_dim"` — "Eq. 32's
    'restored to the original dimensionality' is ambiguous between (a) the pre-compression
    state dimensionality (2*D) and (b) the original per-layer prompt dimensionality d needed
    to feed back into G/GV" (`FINAL_RESEARCH_DECISIONS.md` Issue 25).
Evidence type: IMPLEMENTATION-CHOICE (one of two documented readings; the OTHER reading —
    reshaping `q_final` back into a `(L, 1, d)` prompt tensor — is explicitly named a
    separate TRUE BLOCKER (Issue 26 / `feedback_reshape_path`) and is OUT OF SCOPE for this
    module and this stage: `FeatureRestorer` only implements Eq. 32's linear layer itself,
    never an implicit reshape into prompt-tensor form).
Confidence: Low (per Issue 25's own stated ambiguity) — `restored_dim` is therefore a
    REQUIRED constructor argument (no numeric default), so a caller must explicitly choose
    which reading it wants (commonly `2*D`, i.e. `state_dim`, to round-trip Eq. 27's output
    width) rather than this module silently picking one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn


class RestorerConfigError(Exception):
    """Raised when a `FeatureRestorerConfig`'s own fields are invalid."""


class RestorerShapeError(Exception):
    """Raised when input to `FeatureRestorer` has an invalid shape."""


@dataclass
class FeatureRestorerConfig:
    """Config for `FeatureRestorer` (Eq. 32).

    compressed_dim: `Dc`, the incoming `q_new`'s feature dim.
    restored_dim:   output width. UNRESOLVED-ambiguous in the paper (see module docstring);
                    no default — caller must supply explicitly (commonly `2*D` = `state_dim`,
                    per `configs/model/hgn_ec.yaml: restoration_target_dim = "state_dim"`).
    provenance:     free-form dict.
    """

    compressed_dim: int
    restored_dim: int
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("compressed_dim", "restored_dim"):
            val = getattr(self, name)
            if not isinstance(val, int) or val <= 0:
                raise RestorerConfigError(f"FeatureRestorerConfig.{name} must be a positive int, got {val!r}.")


class FeatureRestorer(nn.Module):
    """Eq. 32: `q_final = W_restore . q_new + b_restore` — a single linear layer."""

    def __init__(self, config: FeatureRestorerConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.compressed_dim, config.restored_dim)

    def forward(self, q_new: torch.Tensor) -> torch.Tensor:
        """`q_new`: `(..., N, compressed_dim)` -> `(..., N, restored_dim)`."""
        if q_new.dim() not in (2, 3):
            raise RestorerShapeError(f"FeatureRestorer: expected q_new of shape (N, D) or (B, N, D), got {tuple(q_new.shape)}.")
        if q_new.shape[-1] != self.config.compressed_dim:
            raise RestorerShapeError(
                f"FeatureRestorer: q_new last dim {q_new.shape[-1]} != config.compressed_dim {self.config.compressed_dim}."
            )
        return self.linear(q_new)

    def extra_repr(self) -> str:
        return f"compressed_dim={self.config.compressed_dim}, restored_dim={self.config.restored_dim}"

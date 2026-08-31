"""
models/acga/discriminator.py
==============================

Stage 6 — ACGA adversarial discriminator (Eq. 25).

    D(z) = Sigmoid(W2 . GELU(W1 z + b1) + b2)                                       (Eq. 25)

Section IV.C.3: "A discriminator D: R^K -> [0,1] is introduced to force the latent
representation Z to match the prior distribution p_z = N(0, I). The discriminator is
composed of two fully-connected layers..."

--------------------------------------------------------------------------------------------
REFERENCE TRACEABILITY
--------------------------------------------------------------------------------------------

Component: Discriminator layer count and activations.
Equation/reference: Eq. 25.
Source: Section IV.C.3, "composed of two fully-connected layers"; Eq. 25's literal
    `Sigmoid(W2 . GELU(W1 z + b1) + b2)` form.
Evidence type: PAPER-FACT. Matches `configs/model/acga.yaml: discriminator_num_layers=2,
    discriminator_hidden_activation=gelu, discriminator_output_activation=sigmoid`.
Confidence: High.

Component: Discriminator output is sigmoid-bounded to `[0, 1]` despite Eq. 26 labeling the
    loss built from it a "Wasserstein distance".
Equation/reference: Eq. 25 (bounded output) vs. Eq. 26 ("Wasserstein-style" loss label).
Source: `configs/model/acga.yaml: adversarial_loss_form` note; `FINAL_RESEARCH_DECISIONS.md`
    Issue 14: "a sigmoid-bounded critic is atypical for a Wasserstein-distance loss; the paper
    gives no acknowledgment of this tension; not resolved, implemented literally as written,
    flagged."
Evidence type: PAPER-FACT (Eq. 25's literal form is implemented as written; the tension with
    conventional WGAN critics is a documented, not silently resolved, inconsistency -- project
    rule #8, "do not silently correct inconsistencies").
Confidence: High (for "implement literally"); the underlying paper inconsistency itself is
    unresolved by design.

Component: Discriminator hidden width.
Equation/reference: Eq. 25 gives the two-layer FC + GELU + Sigmoid *form* but no numeric
    hidden width.
Source: `configs/model/acga.yaml: discriminator_hidden_dim` = UNRESOLVED (null).
Evidence type: UNRESOLVED. `DiscriminatorConfig.hidden_dim` defaults to `None`, which resolves
    locally (code-level, never YAML-tracked as a paper fact) to `latent_dim` -- mirroring
    `GINLayerConfig.mlp_hidden_dim`'s established "default to the other known dim" convention
    in this codebase. This default is an IMPLEMENTATION-CHOICE, exposed and overridable, never
    claimed as paper-derived.
Confidence: Low (no paper evidence either way for the numeric default; the *shape* of the
    default choice mirrors an established codebase convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn


class DiscriminatorConfigError(Exception):
    """Raised when a `DiscriminatorConfig`'s own fields are invalid."""


class DiscriminatorShapeError(Exception):
    """Raised when `Discriminator` receives an invalid input shape."""


@dataclass
class DiscriminatorConfig:
    """Config for `Discriminator` (Eq. 25).

    latent_dim: `K`, must match `ACGAEncoderConfig.latent_dim` for a given `ACGA` instance.
    hidden_dim: internal FC width. UNRESOLVED in the paper; `None` -> defaults to `latent_dim`
                (IMPLEMENTATION-CHOICE, see module docstring).
    provenance: free-form dict.
    """

    latent_dim: int
    hidden_dim: Optional[int] = None
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.latent_dim, int) or self.latent_dim <= 0:
            raise DiscriminatorConfigError(
                f"DiscriminatorConfig.latent_dim must be a positive int, got {self.latent_dim!r}."
            )
        if self.hidden_dim is None:
            self.hidden_dim = self.latent_dim
        elif not isinstance(self.hidden_dim, int) or self.hidden_dim <= 0:
            raise DiscriminatorConfigError(
                f"DiscriminatorConfig.hidden_dim must be a positive int or None, got {self.hidden_dim!r}."
            )


class Discriminator(nn.Module):
    """ACGA's adversarial discriminator (Eq. 25): `z in R^K -> D(z) in [0, 1]`.

    Two fully-connected layers: `Linear(K, hidden) -> GELU -> Linear(hidden, 1) -> Sigmoid`,
    implemented exactly as Eq. 25 is written (see module docstring re: the sigmoid-bounded /
    "Wasserstein" tension -- implemented literally, not silently corrected).
    """

    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.latent_dim, config.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.hidden_dim, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(..., N, K)` (or `(..., K)`) -> `D(z)`: same leading shape, last dim squeezed.

        Never mutates `z` in place.
        """
        if z.shape[-1] != self.config.latent_dim:
            raise DiscriminatorShapeError(
                f"Discriminator: z last dim {z.shape[-1]} does not match "
                f"config.latent_dim {self.config.latent_dim}."
            )
        h = self.act(self.fc1(z))
        logits = self.fc2(h)
        d = self.out_act(logits)
        return d.squeeze(-1)

    def extra_repr(self) -> str:
        return f"latent_dim={self.config.latent_dim}, hidden_dim={self.config.hidden_dim}"

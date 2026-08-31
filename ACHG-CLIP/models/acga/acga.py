"""
models/acga/acga.py
======================

Stage 6 — top-level ACGA module: wires `ACGAEncoder` (Eq. 22), `InnerProductDecoder`
(Eq. 23), and `Discriminator` (Eq. 25) together, and exposes both loss terms
(`losses/acga_losses.py`: Eqs. 24, 26) as separate components for Stage 8's training loop to
compose (Stage 6 task, "LOSSES" section: "Return individual loss components separately... so
Stage 8/training can compose them later.").

--------------------------------------------------------------------------------------------
CRITICAL: ACGA is a PARALLEL AUXILIARY HEAD, not a transformation stage HGN-EC consumes
--------------------------------------------------------------------------------------------

    GIN output (X, A)
       |--> ACGA  -> Z, A_hat  (used ONLY for L_recon, L_adv)
       |
       `--> (later Stage 7) HGN-EC, consuming the SAME (X, A) unchanged

Source: `configs/model/acga.yaml: acga_hgnec_data_wiring` (IMPLEMENTATION-CHOICE, Blocker 4);
`FINAL_IMPLEMENTATION_BLUEPRINT.md` Blocker 4 resolution, quoted in full: "ACGA computes Z and
A_hat only for its own loss terms (L_recon, L_adv); the (X, A) that GIN produced pass through
to HGN-EC unchanged. ACGA acts as a parallel auxiliary/regularization head off the same
(X, A), not as a transformation stage HGN-EC consumes downstream of."

`ACGA.forward` therefore:
  * takes `(X, A)` (or a Stage-4 `Graph`) as input,
  * returns an `ACGAOutput` bundle (`Z`, `A_hat`, discriminator outputs, loss components),
  * never returns a `Graph`, never mutates its `X`/`A` inputs, and provides no code path by
    which its `Z`/`A_hat` could silently overwrite the GIN `X`/`A` that a later HGN-EC stage
    (not implemented here, per this Stage-6 task's explicit "Do NOT implement HGN-EC" /
    "STRICT STOP" instructions) would consume.

--------------------------------------------------------------------------------------------
REFERENCE TRACEABILITY
--------------------------------------------------------------------------------------------

Component: Per-modality invocation (ACGA run once for the text graph, once for the vision
    graph, with SHARED weights).
Equation/reference: Section IV.B ("respectively") vs. Section IV.C's singular `X, A, Z, H`.
Source: `configs/model/acga.yaml: modality_scope` = `independent_shared_weights`
    (IMPLEMENTATION-CHOICE, Blocker 5). `FINAL_IMPLEMENTATION_BLUEPRINT.md` Blocker 5
    resolution: "one set of ACGA parameters... invoked twice per training step (once on the
    text graph, once on the vision graph), producing two independent L_recon/L_adv... summed
    before... lambda weighting."
Evidence type: IMPLEMENTATION-CHOICE.
Confidence: Low-Medium (per the blueprint's own stated confidence).
Note: this Stage-6 module implements ONE `ACGA` instance with shared weights, callable on
    either modality's `(X, A)`; the "call it twice and sum" composition itself belongs to
    Stage 8's total-loss wiring (out of scope here, per the Stage-6 task's explicit "Do not
    implement... the training loop").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.acga.discriminator import Discriminator, DiscriminatorConfig
from models.acga.decoder import InnerProductDecoder
from models.acga.encoder import ACGAEncoder, ACGAEncoderConfig
from losses.acga_losses import adversarial_loss, reconstruction_loss
from models.graph.graph_data import Graph, GraphShapeError


class ACGAConfigError(Exception):
    """Raised when an `ACGAConfig`'s own fields are invalid."""


@dataclass
class ACGAConfig:
    """Config for the top-level `ACGA` module.

    input_dim:  `D`, node feature dim consumed by the encoder (see `ACGAEncoderConfig`).
    latent_dim: `K`, the latent space dimension (UNRESOLVED in the paper; required, no
                default -- see `ACGAEncoderConfig`).
    encoder_num_layers: forwarded to `ACGAEncoderConfig.num_layers` (IMPLEMENTATION-CHOICE
                default `1`, see `encoder.py`).
    encoder_mlp_hidden_dim: forwarded to `ACGAEncoderConfig.mlp_hidden_dim`.
    encoder_eps_init: forwarded to `ACGAEncoderConfig.eps_init`.
    discriminator_hidden_dim: forwarded to `DiscriminatorConfig.hidden_dim` (UNRESOLVED in
                the paper; `None` -> defaults to `latent_dim`, see `discriminator.py`).
    prior_distribution: fixed to `"standard_normal"` -- the only implemented prior
                (`configs/model/acga.yaml: prior_distribution`, PAPER-FACT, Eq. 26:
                `z ~ p_z = N(0, I)`). Kept as an explicit field (not a bare constant) so a
                future non-standard-normal prior only touches this module.
    negative_sampling_ratio: forwarded to `reconstruction_loss` (UNRESOLVED in the paper,
                `configs/model/acga.yaml: negative_sampling_ratio`; `None` -> the dense
                full-matrix default documented in `losses/acga_losses.py`).
    reduction:  forwarded to `reconstruction_loss` (`"mean"` or `"sum"`; IMPLEMENTATION-CHOICE
                default `"mean"`, see `losses/acga_losses.py`).
    provenance: free-form dict.
    """

    input_dim: int
    latent_dim: int
    encoder_num_layers: int = 1
    encoder_mlp_hidden_dim: Optional[int] = None
    encoder_eps_init: float = 0.0
    discriminator_hidden_dim: Optional[int] = None
    prior_distribution: str = "standard_normal"
    negative_sampling_ratio: Optional[float] = None
    reduction: str = "mean"
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prior_distribution != "standard_normal":
            raise ACGAConfigError(
                "ACGAConfig.prior_distribution: only 'standard_normal' is implemented "
                "(configs/model/acga.yaml: prior_distribution, PAPER-FACT, Eq. 26); "
                f"got {self.prior_distribution!r}."
            )
        if self.reduction not in ("mean", "sum"):
            raise ACGAConfigError(f"ACGAConfig.reduction must be 'mean' or 'sum', got {self.reduction!r}.")


@dataclass
class ACGAOutput:
    """Bundle of everything a single `ACGA.forward` call produces.

    Z:          `(..., N, K)` -- latent representation (Eq. 22). Exposed separately from
                every other field so callers can distinguish it from the input `X`/`A`
                (Stage 6 task, "Z/reconstruction are exposed separately").
    A_hat:      `(..., N, N)` -- reconstructed adjacency (Eq. 23). Exposed separately from
                the input `A` for the same reason; NEVER the same tensor object as the input
                `A` (see `ACGA.forward`'s immutability guarantee).
    d_real:     discriminator output on prior samples `z ~ N(0, I)`, `(..., N)` (Eq. 26's
                `E_{z~p_z}[D(z)]` term, per-sample).
    d_fake:     discriminator output on encoded `Z`, `(..., N)` (Eq. 26's
                `E_{z~q(Z|X,A)}[D(z)]` term, per-sample).
    reconstruction_loss: scalar (Eq. 24). Returned separately, NOT pre-summed with
                `adversarial_loss` -- Stage 8 composes `L_total` (Eq. 34).
    adversarial_loss: scalar (Eq. 26). Returned separately (see above).
    """

    Z: torch.Tensor
    A_hat: torch.Tensor
    d_real: torch.Tensor
    d_fake: torch.Tensor
    reconstruction_loss: torch.Tensor
    adversarial_loss: torch.Tensor


class ACGA(nn.Module):
    """Top-level Adversarially Constrained Graph Autoencoder (Eqs. 22-26).

    A PARALLEL AUXILIARY HEAD (see module docstring): consumes `(X, A)` (or a Stage-4
    `Graph`), produces `Z`/`A_hat`/discriminator outputs/loss components, and never mutates
    or overwrites its inputs. Does not implement HGN-EC, the training loop, or the top-level
    `L_total` composition (Eq. 34) -- all out of scope for Stage 6.
    """

    def __init__(self, config: ACGAConfig):
        super().__init__()
        self.config = config
        self.encoder = ACGAEncoder(
            ACGAEncoderConfig(
                input_dim=config.input_dim,
                latent_dim=config.latent_dim,
                num_layers=config.encoder_num_layers,
                mlp_hidden_dim=config.encoder_mlp_hidden_dim,
                eps_init=config.encoder_eps_init,
            )
        )
        self.decoder = InnerProductDecoder()
        self.discriminator = Discriminator(
            DiscriminatorConfig(
                latent_dim=config.latent_dim,
                hidden_dim=config.discriminator_hidden_dim,
            )
        )

    def forward_tensors(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> ACGAOutput:
        """Tensor-level entry point. `X`: `(..., N, D)`, `A`: `(..., N, N)`.

        Does not mutate `X`/`A` in place: every op below (`ACGAEncoder`, `InnerProductDecoder`,
        `Discriminator`, prior sampling) is purely functional over its inputs, never an
        in-place tensor op on `X` or `A` themselves.
        """
        Z = self.encoder(X, A)  # (..., N, K), Eq. 22.
        A_hat = self.decoder(Z)  # (..., N, N), Eq. 23.

        # Eq. 26: z ~ p_z = N(0, I), same shape as Z (one prior sample per node/graph-slot).
        if generator is not None:
            prior_z = torch.empty_like(Z).normal_(mean=0.0, std=1.0, generator=generator)
        else:
            prior_z = torch.randn_like(Z)

        d_real = self.discriminator(prior_z)  # (..., N) -- E_{z~p_z}[D(z)] terms.
        d_fake = self.discriminator(Z)  # (..., N) -- E_{z~q(Z|X,A)}[D(z)] terms.

        recon_loss = reconstruction_loss(
            A,
            A_hat,
            negative_sampling_ratio=self.config.negative_sampling_ratio,
            reduction=self.config.reduction,
        )
        adv_loss = adversarial_loss(d_real, d_fake)

        return ACGAOutput(
            Z=Z,
            A_hat=A_hat,
            d_real=d_real,
            d_fake=d_fake,
            reconstruction_loss=recon_loss,
            adversarial_loss=adv_loss,
        )

    def forward(self, graph: Graph, *, generator: Optional[torch.Generator] = None) -> ACGAOutput:
        """Consume the Stage-4 `Graph` contract. Returns `ACGAOutput`, NOT a `Graph` --

        this is the explicit "parallel auxiliary head" contract (module docstring): ACGA
        never returns a `(X, A)`-shaped object that could be mistaken for -- or accidentally
        substituted into -- the GIN `Graph` a later HGN-EC stage consumes.
        """
        if graph.feature_dim != self.config.input_dim:
            raise GraphShapeError(
                f"ACGA: graph.feature_dim {graph.feature_dim} does not match "
                f"config.input_dim {self.config.input_dim}."
            )
        return self.forward_tensors(graph.X, graph.A, generator=generator)

    def extra_repr(self) -> str:
        return f"input_dim={self.config.input_dim}, latent_dim={self.config.latent_dim}"

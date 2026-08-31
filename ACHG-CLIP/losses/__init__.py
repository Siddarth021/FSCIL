"""
losses/
========

Stage 6 populates only `losses/acga_losses.py` (ACGA-specific loss terms, Eqs. 24, 26).
`losses/hgn_ec_losses.py` (Eq. 33) and `losses/total_loss.py` (Eq. 34) are later stages,
per `docs/FINAL_IMPLEMENTATION_BLUEPRINT.md` Part 9's stage table -- not implemented here.
"""

from __future__ import annotations

from losses.acga_losses import (
    ACGALossError,
    adversarial_loss,
    clip_discriminator_weights,
    gradient_penalty,
    reconstruction_loss,
)

__all__ = [
    "ACGALossError",
    "reconstruction_loss",
    "adversarial_loss",
    "clip_discriminator_weights",
    "gradient_penalty",
]

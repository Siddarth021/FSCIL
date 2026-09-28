"""
models/acga/
==============

Stage 6 — Adversarially Constrained Graph Autoencoder (Eqs. 22-26).

ACGA is a PARALLEL AUXILIARY HEAD off the Stage-5 GIN's `(X, A)` output -- see
`models/acga/acga.py`'s module docstring. Does not implement HGN-EC (Stage 7) or any
top-level `ACHG-CLIP` wiring (Stage 8).
"""

from __future__ import annotations

from models.acga.acga import ACGA, ACGAConfig, ACGAConfigError, ACGAOutput
from models.acga.decoder import DecoderShapeError, InnerProductDecoder
from models.acga.discriminator import (
    Discriminator,
    DiscriminatorConfig,
    DiscriminatorConfigError,
    DiscriminatorShapeError,
)
from models.acga.encoder import ACGAEncoder, ACGAEncoderConfig, ACGAEncoderConfigError

__all__ = [
    "ACGA",
    "ACGAConfig",
    "ACGAConfigError",
    "ACGAOutput",
    "ACGAEncoder",
    "ACGAEncoderConfig",
    "ACGAEncoderConfigError",
    "InnerProductDecoder",
    "DecoderShapeError",
    "Discriminator",
    "DiscriminatorConfig",
    "DiscriminatorConfigError",
    "DiscriminatorShapeError",
]

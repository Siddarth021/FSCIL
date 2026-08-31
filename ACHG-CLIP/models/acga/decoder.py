"""
models/acga/decoder.py
========================

Stage 6 — ACGA structural reconstruction decoder (Eq. 23).

    A_hat = sigmoid(Z Z^T)                                                          (Eq. 23)

Section IV.C.2: "The decoder reconstructs the adjacency matrix through the inner product of
latent representations and models the probability of edge existence using the Sigmoid
function."

--------------------------------------------------------------------------------------------
REFERENCE TRACEABILITY
--------------------------------------------------------------------------------------------

Component: Decoder architecture (inner-product + sigmoid).
Equation/reference: Eq. 23.
Source: Section IV.C.2, verbatim.
Evidence type: PAPER-FACT.
Confidence: High.

Component: Decoder has no learnable parameters.
Equation/reference: Eq. 23.
Source: `A_hat = sigmoid(Z Z^T)` contains no weight matrix -- purely a function of `Z`.
Evidence type: PAPER-FACT (absence of any stated weight in the formula).
Confidence: High.

Component: `Do not invent an alternate decoder merely because it is common in ARGA
    implementations.` (Stage 6 task constraint.)
Source: Stage 6 task spec, "GRAPH RECONSTRUCTION" section.
Evidence type: IMPLEMENTATION-CHOICE (compliance) -- no alternate decoder (e.g. an MLP
    decoder, common in some ARGA variants) is implemented here; only the literal inner-product
    form of Eq. 23.
Confidence: High (direct task instruction).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DecoderShapeError(Exception):
    """Raised when `InnerProductDecoder` receives an invalid `Z` shape."""


class InnerProductDecoder(nn.Module):
    """ACGA's structural reconstruction decoder (Eq. 23): `Z -> A_hat in [0,1]^{N x N}`.

    Stateless (no learnable parameters, see module docstring). Does not mutate its `Z` input.
    """

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """`Z`: `(..., N, K)` -> `A_hat`: `(..., N, N)`, symmetric, entries in `[0, 1]`."""
        if Z.dim() not in (2, 3):
            raise DecoderShapeError(
                f"InnerProductDecoder: expected Z of shape (N, K) or (B, N, K), got {tuple(Z.shape)}."
            )
        Z_t = Z.transpose(-2, -1)
        logits = torch.matmul(Z, Z_t)  # (..., N, N); symmetric by construction (Z Z^T).
        A_hat = torch.sigmoid(logits)
        return A_hat

    def extra_repr(self) -> str:
        return "inner_product_sigmoid (Eq. 23, no learnable parameters)"

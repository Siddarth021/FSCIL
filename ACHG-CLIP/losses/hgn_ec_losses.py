"""
losses/hgn_ec_losses.py
===========================

Stage 7 — HGN-EC energy conservation loss (Eq. 33).

--------------------------------------------------------------------------------------------
REFERENCE TRACEABILITY
--------------------------------------------------------------------------------------------

Decision: `L_energy = MSE(H_initial, H_final) = (1/n) * sum_{i=1}^{n} (H_initial,i -
    H_final,i)^2`, implemented via `torch.nn.functional.mse_loss` over whatever leading
    (batch) dimension `H_initial`/`H_final` carry — `n` = that dimension's size (batch size,
    under this module's `hamiltonian_output_shape = "scalar_per_graph"` reading; see
    `models/hgn_ec/hamiltonian.py`'s module docstring).
Source: Eq. 33; `configs/model/hgn_ec.yaml: energy_loss_form =
    "mse_between_initial_and_final_H"` (PAPER-FACT).
Evidence type: PAPER-FACT (the MSE form itself).
Confidence: High.

Decision: `n`'s numeric referent (batch size vs. node count `N`) is left to whatever shape
    `H_initial`/`H_final` actually have when passed in — this function does not itself
    resolve `configs/model/hgn_ec.yaml: energy_loss_n_referent` (UNRESOLVED); it simply
    computes the mean of squared differences over every element of its inputs, which is
    equivalent to Eq. 33's literal per-graph-`H` reading when `H_initial`/`H_final` are
    `(B,)`-shaped (this project's `hamiltonian_output_shape = "scalar_per_graph"` default,
    `FINAL_RESEARCH_DECISIONS.md` Issue 19) and would be equally correct under a per-node `H`
    reading if a future revision passes `(B, N)`-shaped tensors instead — this function's
    contract is deliberately shape-agnostic so it does not need to change if Issue 19 is
    later resolved differently.
Source: `configs/model/hgn_ec.yaml: energy_loss_n_referent = UNRESOLVED`.
Evidence type: UNRESOLVED (paper gap), handled by shape-agnostic design rather than a guess.
Confidence: N/A.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class HGNECLossShapeError(Exception):
    """Raised when inputs to `energy_conservation_loss` have mismatched shapes."""


def energy_conservation_loss(H_initial: torch.Tensor, H_final: torch.Tensor) -> torch.Tensor:
    """Eq. 33: `L_energy = MSE(H_initial, H_final)`.

    `H_initial`, `H_final`: any matching shape (typically `(...,)`, one scalar energy value
    per graph/batch-element — see module docstring). Returns a 0-dim scalar tensor.
    """
    if H_initial.shape != H_final.shape:
        raise HGNECLossShapeError(
            f"energy_conservation_loss: H_initial shape {tuple(H_initial.shape)} != H_final shape {tuple(H_final.shape)}."
        )
    return F.mse_loss(H_final, H_initial)

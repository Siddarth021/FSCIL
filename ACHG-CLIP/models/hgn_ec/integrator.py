"""
models/hgn_ec/integrator.py
===============================

Stage 7 — HGN-EC Symplectic Euler integration step (Eq. 31).

--------------------------------------------------------------------------------------------
REFERENCE TRACEABILITY
--------------------------------------------------------------------------------------------

Decision: `p_new = p + dt * p_dot`, `q_new = q + dt * q_dot` — implemented EXACTLY as
    written, in this order (`p` updated using the CURRENT `q_dot`/`p_dot` values passed in,
    not re-derived mid-step). No ordinary Euler, RK4, Verlet, or generic ODE solver is
    substituted (explicit Stage 7 task constraint).
Source: Eq. 31.
Evidence type: PAPER-FACT.
Confidence: High.

Decision: `dt` has NO default and must be supplied by the caller at every call site (never a
    module-level constant).
Source: `configs/model/hgn_ec.yaml: dt = UNRESOLVED` — "No numeric value, default, or
    search range anywhere in the paper." Explicit Stage 7 task constraint: "Do NOT invent a
    paper value" / "The paper's dt ... must remain configurable and provenance-tracked."
Evidence type: UNRESOLVED (paper gap) — the resulting "no silent default, always an explicit
    required argument" policy is this module's own IMPLEMENTATION-CHOICE.
Confidence: N/A (required-argument policy, not a numeric guess).

Decision: number of integration steps is a caller-supplied, explicit `num_steps` parameter
    on `HGNEC` (see `models/hgn_ec/hgn_ec.py`), NOT hard-coded inside this module — this
    module implements exactly ONE Symplectic Euler step; looping is the caller's
    responsibility (`configs/model/hgn_ec.yaml: integration_steps = 1`,
    IMPLEMENTATION-CHOICE default, but exposed as a configurable knob, never assumed fixed
    inside `symplectic_euler_step` itself).
Source: `FINAL_RESEARCH_DECISIONS.md` Issue 23; Stage 7 task, "TESTS" section item 8
    ("multiple integration steps").
Evidence type: IMPLEMENTATION-CHOICE.
Confidence: Low (paper gives no iteration language either way — see Issue 23).
"""

from __future__ import annotations

from typing import Tuple

import torch


class IntegratorShapeError(Exception):
    """Raised when inputs to `symplectic_euler_step` have an invalid/mismatched shape."""


def symplectic_euler_step(
    q: torch.Tensor,
    p: torch.Tensor,
    q_dot: torch.Tensor,
    p_dot: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. 31: one Symplectic Euler step.

    ``p_new = p + dt * p_dot``
    ``q_new = q + dt * q_dot``

    All four tensors (`q`, `p`, `q_dot`, `p_dot`) must have identical shape. `dt` is a plain
    Python float (or a 0-dim tensor), always explicitly supplied by the caller — see module
    docstring ("dt has NO default").

    Returns `(q_new, p_new)`. Does not mutate `q`/`p`/`q_dot`/`p_dot` in place (pure
    out-of-place arithmetic), so the pre-step tensors remain valid for anything the caller
    still needs them for (e.g. recomputing `H_initial`).
    """
    if not (q.shape == p.shape == q_dot.shape == p_dot.shape):
        raise IntegratorShapeError(
            "symplectic_euler_step: q, p, q_dot, p_dot must all share the same shape; got "
            f"q={tuple(q.shape)}, p={tuple(p.shape)}, q_dot={tuple(q_dot.shape)}, p_dot={tuple(p_dot.shape)}."
        )
    if isinstance(dt, torch.Tensor):
        if dt.dim() != 0:
            raise IntegratorShapeError(f"symplectic_euler_step: dt tensor must be 0-dim (scalar), got shape {tuple(dt.shape)}.")
    elif not isinstance(dt, (int, float)):
        raise IntegratorShapeError(f"symplectic_euler_step: dt must be a Python int/float or a 0-dim tensor, got {type(dt)!r}.")

    p_new = p + dt * p_dot  # Eq. 31, first line, exactly as written
    q_new = q + dt * q_dot  # Eq. 31, second line, exactly as written
    return q_new, p_new

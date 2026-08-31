"""
models/hgn_ec/state_init.py
===============================

Stage 7 — HGN-EC initial state formation (Eq. 27) and q/p initialization
(Section IV.D.3, unnumbered).

--------------------------------------------------------------------------------------------
REFERENCE TRACEABILITY
--------------------------------------------------------------------------------------------

Decision: `aggregated = A . X` realized as a dense matmul against the adjacency; `state =
    concat([X, aggregated], dim=-1)` (concatenation along the feature axis, doubling the
    feature dimension from `D` to `2*D`).
Source: Eq. 27: `aggregated = A · X, state = [X, aggregated]`; `configs/model/hgn_ec.yaml:
    state_init_form = "concat_X_and_AX"` (PAPER-FACT).
Evidence type: PAPER-FACT.
Confidence: High.

Decision: `A` is the GIN's own output adjacency (from the SAME `(X, A)` pair the calling
    module's Stage-5 GIN produced), used exactly as received — no reconstruction, no
    self-loop masking, matching `GINLayer`'s own established convention
    (`models/gnn/gin_layer.py`).
Source: `FINAL_IMPLEMENTATION_BLUEPRINT.md` Blocker 4: "the (X, A) that GIN produced pass
    through to HGN-EC unchanged."
Evidence type: IMPLEMENTATION-CHOICE (compliance with the frozen Blocker-4 wiring decision).
Confidence: High.

Decision: `q` and `p` are BOTH initialized to the (post-compression) `compressed` vector —
    i.e. `q = compressed`, `p = compressed` — via two independent tensor views of the same
    values (not the same Python object; see `init_q_p`'s docstring on why this matters for
    autograd).
Source: Section IV.D.3 ("HGN-EC assigns the compressed feature vector `compressed` to both
    `q` and `p`"); `configs/model/hgn_ec.yaml: qp_init_rule = "q_and_p_both_equal_compressed"`
    (PAPER-FACT).
Evidence type: PAPER-FACT.
Confidence: High.

Note: `init_q_p` takes `compressed` (Eq. 28's output), NOT the raw `state` (Eq. 27's output)
    — `build_initial_state` and `init_q_p` are two separate, independently testable steps,
    matching the paper's own step numbering (Section IV.D.1 vs. .3) and the Stage 7 task's
    explicit test-category split ("initialization" is distinct from "input/output shapes").
    `models/hgn_ec/compress.py`'s `FeatureCompressor` sits between the two.
"""

from __future__ import annotations

from typing import Tuple

import torch


class StateInitShapeError(Exception):
    """Raised when inputs to `build_initial_state`/`init_q_p` have an invalid shape."""


def build_initial_state(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Eq. 27: `aggregated = A . X`, `state = [X, aggregated]` (concat along feature axis).

    `X`: `(..., N, D)`, `A`: `(..., N, N)` -> `state`: `(..., N, 2*D)`.

    Does not mutate `X`/`A` (pure functional composition of `torch.matmul` + `torch.cat`).
    """
    if X.dim() not in (2, 3):
        raise StateInitShapeError(f"build_initial_state: expected X of shape (N, D) or (B, N, D), got {tuple(X.shape)}.")
    if A.dim() not in (2, 3):
        raise StateInitShapeError(f"build_initial_state: expected A of shape (N, N) or (B, N, N), got {tuple(A.shape)}.")
    if X.dim() != A.dim():
        raise StateInitShapeError(
            f"build_initial_state: X and A must have matching batch-ness: X.dim()={X.dim()} vs A.dim()={A.dim()}."
        )
    if X.shape[-2] != A.shape[-1] or A.shape[-1] != A.shape[-2]:
        raise StateInitShapeError(
            f"build_initial_state: A must be square (N, N) matching X's node dim; got X node dim "
            f"{X.shape[-2]}, A shape {tuple(A.shape[-2:])}."
        )
    if X.dim() == 3 and X.shape[0] != A.shape[0]:
        raise StateInitShapeError(f"build_initial_state: X batch size {X.shape[0]} != A batch size {A.shape[0]}.")

    aggregated = A @ X  # Eq. 27: A . X, (..., N, D)
    state = torch.cat([X, aggregated], dim=-1)  # (..., N, 2*D)
    return state


def init_q_p(compressed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Section IV.D.3: assign `compressed` (Eq. 28's output) to both `q` and `p`.

    Returns two tensors with identical VALUES but independent identities
    (`compressed.clone()` for one of the two), so that `torch.autograd.grad(H, (q, p), ...)`
    can later differentiate `H` with respect to `q` and `p` as distinguishable inputs — using
    the exact same Python object for both would still work numerically (autograd tracks
    values, not aliasing, for this call), but keeping them as distinct tensors avoids any
    possibility of an in-place op on one silently mutating the other, and makes `q is p`
    explicitly `False` (asserted in `tests/test_hgn_ec.py`) rather than relying on that
    implicit guarantee.
    """
    if compressed.dim() not in (2, 3):
        raise StateInitShapeError(
            f"init_q_p: expected compressed of shape (N, Dc) or (B, N, Dc), got {tuple(compressed.shape)}."
        )
    q = compressed
    p = compressed.clone()
    return q, p

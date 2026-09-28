"""
models/hgn_ec/
==================

Stage 7 — Hamiltonian Graph Network with Energy Conservation (Eqs. 27-33).

HGN-EC consumes the Stage-5 GIN's `(X, A)` output DIRECTLY (Blocker 4 — the same `(X, A)`
ACGA consumes as a parallel auxiliary head; NOT ACGA's `Z`/`A_hat`). See `models/hgn_ec/
hgn_ec.py`'s module docstring for the full data-flow diagram and reference traceability.

Does not implement top-level `ACHG-CLIP` wiring (Stage 8), the training loop, datasets,
incremental sessions, evaluation, or any loss outside Eq. 33's energy-conservation term.
"""

from __future__ import annotations

from models.hgn_ec.state_init import (
    build_initial_state,
    init_q_p,
    StateInitShapeError,
)
from models.hgn_ec.compress import FeatureCompressor, FeatureCompressorConfig, CompressorConfigError
from models.hgn_ec.hamiltonian import (
    HamiltonianNet,
    HamiltonianNetConfig,
    HamiltonianNetConfigError,
    HamiltonianShapeError,
    hamiltonian_gradients,
)
from models.hgn_ec.integrator import symplectic_euler_step, IntegratorShapeError
from models.hgn_ec.restore import FeatureRestorer, FeatureRestorerConfig, RestorerConfigError
from models.hgn_ec.hgn_ec import HGNEC, HGNECConfig, HGNECConfigError, HGNECOutput

__all__ = [
    "build_initial_state",
    "init_q_p",
    "StateInitShapeError",
    "FeatureCompressor",
    "FeatureCompressorConfig",
    "CompressorConfigError",
    "HamiltonianNet",
    "HamiltonianNetConfig",
    "HamiltonianNetConfigError",
    "HamiltonianShapeError",
    "hamiltonian_gradients",
    "symplectic_euler_step",
    "IntegratorShapeError",
    "FeatureRestorer",
    "FeatureRestorerConfig",
    "RestorerConfigError",
    "HGNEC",
    "HGNECConfig",
    "HGNECConfigError",
    "HGNECOutput",
]

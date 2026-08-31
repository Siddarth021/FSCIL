"""
models/hgn_ec/hgn_ec.py
===========================

Stage 7 — top-level HGN-EC module: wires initial-state formation (Eq. 27), q/p
initialization, feature compression (Eq. 28), the Hamiltonian energy function (Eq. 29) and
its gradients (Eq. 30), Symplectic Euler integration (Eq. 31, one or more configurable
steps), state restoration (Eq. 32), and the energy-conservation loss (Eq. 33) into a single
`nn.Module`.

--------------------------------------------------------------------------------------------
CRITICAL: HGN-EC consumes the GIN's OWN (X, A), never ACGA's (Z, A_hat)
--------------------------------------------------------------------------------------------

    GIN output (X, A)
       |--> ACGA    -> Z, A_hat   (Stage 6, used ONLY for L_recon, L_adv)
       |
       `--> HGN-EC  -> q_final, L_energy   (THIS module, consumes the SAME (X, A) unchanged)

Source: `configs/model/hgn_ec.yaml: acga_hgnec_data_wiring` (mirrors `configs/model/
acga.yaml`'s own entry); `FINAL_IMPLEMENTATION_BLUEPRINT.md` Blocker 4: "ACGA acts as a
parallel auxiliary/regularization head off the same (X, A), not as a transformation stage
HGN-EC consumes downstream of." `HGNEC.forward_tensors`/`HGNEC.forward` therefore accept
`(X, A)` (or a Stage-4 `Graph`) DIRECTLY — there is no code path anywhere in this module that
imports or references `models.acga.*`, and no parameter named `Z`/`A_hat` appears in this
module's signature.

--------------------------------------------------------------------------------------------
REFERENCE TRACEABILITY (top-level composition)
--------------------------------------------------------------------------------------------

Decision: modality scope — ONE `HGNEC` instance with shared weights, callable on either
    modality's `(X, A)` (text or vision), exactly mirroring `ACGA`'s own Stage-6 precedent.
Source: `configs/model/hgn_ec.yaml: modality_scope = "independent_shared_weights"`
    (IMPLEMENTATION-CHOICE, mirror of `configs/model/acga.yaml`'s own entry, Blocker 5).
Evidence type: IMPLEMENTATION-CHOICE.
Confidence: Low-Medium (per the blueprint's own stated confidence for Blocker 5).
Note: "call it twice (text, vision) and sum/compose the resulting `L_energy` terms" is
    Stage 8's total-loss wiring, out of scope here — same split `ACGA`/Stage 6 established.

Decision: the full per-call sequence is: `build_initial_state` -> `FeatureCompressor` ->
    `init_q_p` -> [loop over `num_steps`: `HamiltonianNet` -> `hamiltonian_gradients` ->
    `symplectic_euler_step`] -> `FeatureRestorer`; `H_initial` is the FIRST loop iteration's
    `H` (computed from the initial `q, p` straight out of `init_q_p`, before any update
    step), and `H_final` is the LAST loop iteration's `H` (computed from the `q, p` values
    that go into the final update step) — i.e. `L_energy` measures energy drift across the
    complete integration, not just across a single step, whenever `num_steps > 1`.
Source: Section IV.D.1-.8's own step ordering; Eq. 33's `H_initial`/`H_final` naming (no
    per-step subscript is given, consistent with measuring drift end-to-end).
Evidence type: IMPLEMENTATION-CHOICE (the specific "first-iteration H vs. last-iteration H"
    definition of `H_initial`/`H_final` when `num_steps > 1`, since the paper's own Section
    IV.D.1-.8 narrative describes a single pass with no iteration language at all — see
    `configs/model/hgn_ec.yaml: integration_steps` note, `FINAL_RESEARCH_DECISIONS.md` Issue
    23).
Confidence: Low for the multi-step generalization specifically (the single-step case,
    `num_steps=1`, is the paper's own best-evidence reading and is the default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.hgn_ec.state_init import build_initial_state, init_q_p
from models.hgn_ec.compress import FeatureCompressor, FeatureCompressorConfig
from models.hgn_ec.hamiltonian import HamiltonianNet, HamiltonianNetConfig, hamiltonian_gradients
from models.hgn_ec.integrator import symplectic_euler_step
from models.hgn_ec.restore import FeatureRestorer, FeatureRestorerConfig
from losses.hgn_ec_losses import energy_conservation_loss
from models.graph.graph_data import Graph, GraphShapeError


class HGNECConfigError(Exception):
    """Raised when an `HGNECConfig`'s own fields are invalid."""


@dataclass
class HGNECConfig:
    """Config for the top-level `HGNEC` module.

    input_dim:       `D`, node feature dim of the incoming GIN `(X, A)` (Eq. 27's input
                     width). Caller-supplied, never defaulted.
    compressed_dim:  `Dc` (`configs/model/hgn_ec.yaml: compressed_dim_Dc`, UNRESOLVED;
                     required, no default — see `FeatureCompressorConfig`).
    restored_dim:    output width of `FeatureRestorer` (Eq. 32). `None` -> defaults to
                     `2 * input_dim` (`state_dim`, per `restoration_target_dim = "state_dim"`
                     — see `restore.py`'s module docstring).
    hnet_gin_hidden_dim: forwarded to `HamiltonianNetConfig.gin_hidden_dim` (`None` ->
                     defaults to `compressed_dim`, see `hamiltonian.py`).
    hnet_num_gin_layers: forwarded to `HamiltonianNetConfig.num_gin_layers`
                     (IMPLEMENTATION-CHOICE default `1`, `configs/model/hgn_ec.yaml:
                     hnet_gin_layers`).
    hnet_mlp_hidden_dim: forwarded to `HamiltonianNetConfig.mlp_hidden_dim`.
    hnet_eps_init:   forwarded to `HamiltonianNetConfig.eps_init`.
    dt:              Symplectic Euler step size (Eq. 31). UNRESOLVED in the paper
                     (`configs/model/hgn_ec.yaml: dt`); REQUIRED, no default — every call
                     must supply it explicitly (see `integrator.py`'s module docstring).
                     Kept as a per-call `forward`/`forward_tensors` argument (NOT frozen into
                     `HGNECConfig` at construction time) so its provenance stays visibly
                     "caller decides every time," never silently baked into the module.
    num_steps:       number of Symplectic Euler steps (`configs/model/hgn_ec.yaml:
                     integration_steps`, IMPLEMENTATION-CHOICE default `1`).
    provenance:      free-form dict.
    """

    input_dim: int
    compressed_dim: int
    restored_dim: Optional[int] = None
    hnet_gin_hidden_dim: Optional[int] = None
    hnet_num_gin_layers: int = 1
    hnet_mlp_hidden_dim: Optional[int] = None
    hnet_eps_init: float = 0.0
    num_steps: int = 1
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.input_dim, int) or self.input_dim <= 0:
            raise HGNECConfigError(f"HGNECConfig.input_dim must be a positive int, got {self.input_dim!r}.")
        if not isinstance(self.compressed_dim, int) or self.compressed_dim <= 0:
            raise HGNECConfigError(f"HGNECConfig.compressed_dim must be a positive int, got {self.compressed_dim!r}.")
        if self.restored_dim is None:
            self.restored_dim = 2 * self.input_dim  # "state_dim" reading, see restore.py
        elif not isinstance(self.restored_dim, int) or self.restored_dim <= 0:
            raise HGNECConfigError(f"HGNECConfig.restored_dim must be a positive int or None, got {self.restored_dim!r}.")
        if not isinstance(self.num_steps, int) or self.num_steps <= 0:
            raise HGNECConfigError(f"HGNECConfig.num_steps must be a positive int, got {self.num_steps!r}.")


@dataclass
class HGNECOutput:
    """Bundle of everything a single `HGNEC.forward(_tensors)` call produces.

    q_final:    `(..., N, restored_dim)` — Eq. 32's output, "the updated learnable prompt"
                (Section IV.D.8). Exposed separately from every other field.
    H_initial:  `(...,)` — Eq. 29's `H` evaluated at the FIRST loop iteration's `q, p` (see
                module docstring on multi-step `H_initial`/`H_final` semantics).
    H_final:    `(...,)` — Eq. 29's `H` evaluated at the LAST loop iteration's `q, p`.
    energy_loss: scalar (Eq. 33). Returned separately, NOT pre-summed with any other loss —
                Stage 8 composes `L_total` (Eq. 34).
    q_trajectory / p_trajectory: list of every intermediate `q`/`p` (length `num_steps + 1`,
                including the initial and final values) — exposed for testing/debugging
                (e.g. verifying the Symplectic Euler update step-by-step); not part of the
                paper's own output contract.
    """

    q_final: torch.Tensor
    H_initial: torch.Tensor
    H_final: torch.Tensor
    energy_loss: torch.Tensor
    q_trajectory: list
    p_trajectory: list


class HGNEC(nn.Module):
    """Top-level Hamiltonian Graph Network with Energy Conservation (Eqs. 27-33).

    Consumes `(X, A)` (or a Stage-4 `Graph`) DIRECTLY — the GIN's own output, never ACGA's
    `Z`/`A_hat` (see module docstring). Does not implement the top-level `ACHG-CLIP` wiring,
    the training loop, datasets, incremental sessions, evaluation, or any loss outside
    Eq. 33 — all out of scope for Stage 7.
    """

    def __init__(self, config: HGNECConfig):
        super().__init__()
        self.config = config
        state_dim = 2 * config.input_dim  # Eq. 27: state = [X, aggregated], width 2*D
        self.compressor = FeatureCompressor(
            FeatureCompressorConfig(state_dim=state_dim, compressed_dim=config.compressed_dim)
        )
        self.hamiltonian_net = HamiltonianNet(
            HamiltonianNetConfig(
                compressed_dim=config.compressed_dim,
                gin_hidden_dim=config.hnet_gin_hidden_dim,
                num_gin_layers=config.hnet_num_gin_layers,
                mlp_hidden_dim=config.hnet_mlp_hidden_dim,
                eps_init=config.hnet_eps_init,
            )
        )
        self.restorer = FeatureRestorer(
            FeatureRestorerConfig(compressed_dim=config.compressed_dim, restored_dim=config.restored_dim)
        )

    def forward_tensors(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        *,
        dt: float,
        num_steps: Optional[int] = None,
    ) -> HGNECOutput:
        """Tensor-level entry point. `X`: `(..., N, D)`, `A`: `(..., N, N)`.

        `dt` is REQUIRED at every call (no default anywhere in this module — see
        `HGNECConfig.dt`'s docstring / `integrator.py`'s module docstring: `dt` is
        UNRESOLVED in the paper and must never be silently invented). `num_steps` defaults
        to `self.config.num_steps` when not overridden per-call.

        Does not mutate `X`/`A` in place; `X`/`A` are never passed to a step that could
        alter their storage (`build_initial_state`, `HamiltonianNet.forward`, and the
        integrator are all purely functional over their inputs).
        """
        n_steps = num_steps if num_steps is not None else self.config.num_steps
        if not isinstance(n_steps, int) or n_steps <= 0:
            raise HGNECConfigError(f"HGNEC.forward_tensors: num_steps must be a positive int, got {n_steps!r}.")

        # Eq. 27: initial state formation.
        state = build_initial_state(X, A)  # (..., N, 2*D)

        # Eq. 28: feature compression.
        compressed = self.compressor(state)  # (..., N, Dc)

        # Section IV.D.3: q, p both initialized to `compressed`.
        q, p = init_q_p(compressed)
        q = q.requires_grad_(True) if not q.requires_grad else q
        p = p.requires_grad_(True) if not p.requires_grad else p

        q_trajectory = [q]
        p_trajectory = [p]
        H_initial: Optional[torch.Tensor] = None
        H_final: Optional[torch.Tensor] = None

        with torch.enable_grad():
            for step in range(n_steps):
                # Eq. 29: Hamiltonian energy function.
                H = self.hamiltonian_net(q, p, A)
                if step == 0:
                    H_initial = H
                H_final = H
    
                # Eq. 30: Hamilton's equations via autodiff.
                q_dot, p_dot = hamiltonian_gradients(H, q, p, create_graph=True)
    
                # Eq. 31: Symplectic Euler update.
                q, p = symplectic_euler_step(q, p, q_dot, p_dot, dt)
                q_trajectory.append(q)
                p_trajectory.append(p)
    
            # Eq. 32: state restoration.
            # We detach q_final if we don't actually need gradients flowing further outside during eval,
            # but usually the restorer expects a tensor.
        q_final = self.restorer(q)

        # Eq. 33: energy conservation loss.
        energy_loss = energy_conservation_loss(H_initial, H_final)

        return HGNECOutput(
            q_final=q_final,
            H_initial=H_initial,
            H_final=H_final,
            energy_loss=energy_loss,
            q_trajectory=q_trajectory,
            p_trajectory=p_trajectory,
        )

    def forward(self, graph: Graph, *, dt: float, num_steps: Optional[int] = None) -> HGNECOutput:
        """Consume the Stage-4 `Graph` contract directly (the GIN's own output — see module
        docstring on the ACGA/HGN-EC parallel-branch boundary)."""
        if graph.feature_dim != self.config.input_dim:
            raise HGNECConfigError(
                f"HGNEC: graph.feature_dim {graph.feature_dim} does not match config.input_dim {self.config.input_dim}."
            )
        return self.forward_tensors(graph.X, graph.A, dt=dt, num_steps=num_steps)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.config.input_dim}, compressed_dim={self.config.compressed_dim}, "
            f"restored_dim={self.config.restored_dim}, num_steps={self.config.num_steps}"
        )

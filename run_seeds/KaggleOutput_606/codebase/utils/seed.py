"""
utils/seed.py
==============

Stage 1 reproducibility infrastructure: seeding for `random`, `numpy`, and (optionally)
`torch`.

Per reproduction_protocol.md Section 13 and FINAL_RESEARCH_DECISIONS.md, the paper specifies
NO random seed, no seed-averaging protocol, and no statement of how many runs its reported
numbers represent. The `seed` value used here therefore always comes from
`configs/experiment.yaml: seed`, which is explicitly tagged IMPLEMENTATION_CHOICE -- this
module never invents its own default independent of that config.

`torch` is an optional dependency at this stage (Stage 1 does not implement any model code),
so torch-seeding is skipped gracefully with a note if torch is not installed, rather than
failing the whole utility.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only in environments without torch
    _HAS_TORCH = False


@dataclass
class SeedReport:
    """Records exactly what was seeded, for logging/checkpoint metadata."""

    seed: int
    deterministic: bool
    torch_available: bool
    torch_seeded: bool


def set_seed(seed: int, deterministic: bool = True) -> SeedReport:
    """Seed all relevant RNGs.

    Args:
        seed: integer seed value. Must be supplied explicitly by the caller (normally read
            from `configs/experiment.yaml: seed` via `utils.config_tracking`) -- this
            function does not choose a seed on its own.
        deterministic: if True and torch is available, request deterministic algorithms
            (`torch.use_deterministic_algorithms(True)`, `cudnn.benchmark=False`). This can
            reduce throughput; the trade-off is intentional and documented
            (FINAL_IMPLEMENTATION_BLUEPRINT.md Part 8).

    Returns:
        A SeedReport describing what was actually seeded (useful for checkpoint metadata
        and structured logs).
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed)} ({seed!r}).")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_seeded = False
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Not all torch builds/ops support full determinism; degrade gracefully
                # rather than crashing seed-setting itself.
                pass
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        torch_seeded = True

    return SeedReport(
        seed=seed,
        deterministic=deterministic,
        torch_available=_HAS_TORCH,
        torch_seeded=torch_seeded,
    )


def get_python_random_state():
    """Return the current stdlib `random` state (for checkpoint RNG bookkeeping)."""
    return random.getstate()


def get_numpy_random_state():
    """Return the current numpy legacy-RNG state (for checkpoint RNG bookkeeping)."""
    return np.random.get_state()


def set_python_random_state(state) -> None:
    random.setstate(state)


def set_numpy_random_state(state) -> None:
    np.random.set_state(state)


def seed_from_config(resolved_config, allow_unresolved: bool = False) -> SeedReport:
    """Convenience wrapper: pull `experiment.seed` / `experiment.deterministic` out of a
    `utils.config_tracking.ResolvedConfig` and seed everything in one call.

    This is the path real scripts should use, so the seed used for an experiment is always
    exactly the one recorded in that experiment's saved config.
    """
    seed = resolved_config.get("experiment.seed", allow_unresolved=allow_unresolved)
    deterministic = resolved_config.get(
        "experiment.deterministic", allow_unresolved=allow_unresolved, default=True
    )
    return set_seed(seed=seed, deterministic=bool(deterministic))

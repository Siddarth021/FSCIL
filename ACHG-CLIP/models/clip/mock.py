"""
models/clip/mock.py
=====================

Stage 2 — deterministic mock CLIP backbone for tests.

`configs/model/clip_backbone.yaml` leaves `variant`, `d`, `d_e`, `d_k`, and `L` UNRESOLVED
(no numeric value is given anywhere in the paper, and no named CLIP checkpoint exists to
download for a variant the paper never identifies -- see `docs/implementation_progress.md`
Stage 2 "Dependency / environment issues"). There is therefore no real pretrained backbone
this project can currently load.

This module builds a small, randomly-initialized (not pretrained), fully deterministic
`CLIPWrapper` using tiny synthetic dimensions, exactly matching the philosophy of
`FINAL_IMPLEMENTATION_BLUEPRINT.md` Part 10's synthetic smoke test ("a randomly-initialized
(not pretrained) tiny Transformer stand-in for CLIP, since the smoke test goal is pipeline
shape/gradient-flow correctness, not accuracy"). It is used by Stage 2's tests
(`tests/test_clip_wrapper.py`) and will be reused by the real `scripts/smoke_test.py` once
later stages exist.

Every dimension here is explicitly `MOCK_SYNTHETIC` in `CLIPConfig.dim_provenance` --
never `PAPER_FACT`, never `IMPLEMENTATION_CHOICE` -- so nothing downstream can mistake a
mock-backbone run for a real reproduction attempt (`FINAL_IMPLEMENTATION_BLUEPRINT.md`
Part 11's "Explicit non-claim").
"""

from __future__ import annotations

from typing import Optional

from models.clip.clip_wrapper import CLIPConfig, CLIPWrapper

MOCK_SYNTHETIC = "MOCK_SYNTHETIC"

# Tiny synthetic defaults. Values are deliberately small and arbitrary (chosen only for fast,
# legible tests), not derived from any paper evidence.
MOCK_D_MODEL = 8
MOCK_D_E = 4
MOCK_D_K = 4
MOCK_NUM_HEADS = 2
MOCK_NUM_LAYERS = 3
MOCK_VOCAB_SIZE = 50
MOCK_MAX_TEXT_LEN = 16
MOCK_PATCH_DIM = 12
MOCK_MAX_PATCHES = 9


def build_mock_clip_config(
    d_model: int = MOCK_D_MODEL,
    d_e: int = MOCK_D_E,
    d_k: int = MOCK_D_K,
    num_heads: int = MOCK_NUM_HEADS,
    num_layers: int = MOCK_NUM_LAYERS,
    vocab_size: int = MOCK_VOCAB_SIZE,
    max_text_len: int = MOCK_MAX_TEXT_LEN,
    patch_dim: int = MOCK_PATCH_DIM,
    max_patches: int = MOCK_MAX_PATCHES,
    frozen: bool = True,
    dropout: float = 0.0,
    ffn_hidden_dim: Optional[int] = None,
) -> CLIPConfig:
    """Build a small synthetic `CLIPConfig` for shape/gradient-flow tests only."""
    dim_provenance = {
        "variant": MOCK_SYNTHETIC,
        "d_model": MOCK_SYNTHETIC,
        "d_e": MOCK_SYNTHETIC,
        "d_k": MOCK_SYNTHETIC,
        "num_heads": MOCK_SYNTHETIC,
        "num_layers": MOCK_SYNTHETIC,
        "vocab_size": MOCK_SYNTHETIC,
        "max_text_len": MOCK_SYNTHETIC,
        "patch_dim": MOCK_SYNTHETIC,
        "max_patches": MOCK_SYNTHETIC,
        "frozen": MOCK_SYNTHETIC,
        "ffn_hidden_dim": MOCK_SYNTHETIC,
        "dropout": MOCK_SYNTHETIC,
    }
    return CLIPConfig(
        d_model=d_model,
        d_e=d_e,
        d_k=d_k,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
        max_text_len=max_text_len,
        patch_dim=patch_dim,
        max_patches=max_patches,
        ffn_hidden_dim=ffn_hidden_dim,
        dropout=dropout,
        variant="mock-synthetic-clip",
        frozen=frozen,
        dim_provenance=dim_provenance,
    )


def build_mock_clip_wrapper(**config_kwargs) -> CLIPWrapper:
    """Build a `CLIPWrapper` around `build_mock_clip_config(**config_kwargs)`.

    Determinism across calls is governed by whatever RNG state is active when this is
    called (e.g. `utils.seed.set_seed`) -- this function does not seed anything itself, so
    tests that need bit-for-bit reproducibility must call `set_seed` first.
    """
    return CLIPWrapper(build_mock_clip_config(**config_kwargs))

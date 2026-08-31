"""
models/prompts/_common.py
===========================

Stage 3 — shared plumbing for `text_prompt.py` and `vision_prompt.py`.

Not part of the blueprint's named file list on its own; exists only to avoid duplicating the
identical init/validation/provenance logic the two modules share (both build a
`(num_layers, num_prompts, prompt_dim)` learnable parameter and both concatenate it into a
per-layer sequence, per FINAL_IMPLEMENTATION_BLUEPRINT.md Blocker 2). Nothing here is
paper-derived; it is IMPLEMENTATION-CHOICE plumbing only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

INSERTION_MODE = "concatenate"  # frozen decision, Blocker 2 -- do not change silently.


class PromptShapeError(Exception):
    """Raised when a prompt tensor or an insertion target has an invalid/mismatched shape."""


class PromptConfigError(Exception):
    """Raised when a prompt config's own fields are invalid (non-positive dims, etc.)."""


@dataclass
class _BasePromptConfig:
    """Shared fields for `TextPromptConfig` / `VisionPromptConfig`.

    num_layers:  L -- number of Transformer layers the prompt tensor covers (Eq. 9/10's `l`
                 index range). Must match the CLIP tower's `num_layers` for the injector to be
                 usable with a given `CLIPWrapper`, but this module never reads a
                 `CLIPWrapper`/`CLIPConfig` directly (kept decoupled per the "do not hard-code
                 prompt behavior into the CLIP encoder itself" constraint) -- callers are
                 responsible for passing a matching `num_layers`.
    num_prompts: M -- number of learnable prompt tokens per layer. PAPER_FACT default is 1
                 (configs/model/prompts.yaml: num_learnable_prompts_M), but this dataclass
                 does not hard-code that value; callers read it from config.
    prompt_dim:  d -- embedding dimension prompts live in (must match the CLIP tower's token
                 dimension `d_model` for concatenation to be shape-valid).
    insertion_mode: always "concatenate" here (Blocker 2 IMPLEMENTATION-CHOICE). Kept as an
                 explicit field (not a bare constant) so the mechanism stays swappable later
                 without changing every call site -- see module docstrings below.
    init_std:    stddev of the Normal(0, init_std) initializer. Paper gives no initialization
                 recipe for Eq. 9/10's prompts (UNRESOLVED) -- IMPLEMENTATION-CHOICE default,
                 matching common learnable-prompt practice (e.g. VPT/CoOp-style small-std init).
    seed:        optional int; when given, the prompt parameter is initialized deterministically
                 (Requirement: "Deterministic initialization when seed is fixed").
    """

    num_layers: int
    num_prompts: int
    prompt_dim: int
    insertion_mode: str = INSERTION_MODE
    init_std: float = 0.02
    seed: Optional[int] = None
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_prompts", "prompt_dim"):
            val = getattr(self, name)
            if not isinstance(val, int) or val <= 0:
                raise PromptConfigError(f"{type(self).__name__}.{name} must be a positive int, got {val!r}.")
        if self.insertion_mode != INSERTION_MODE:
            raise PromptConfigError(
                f"{type(self).__name__}.insertion_mode must be {INSERTION_MODE!r} "
                f"(FINAL_IMPLEMENTATION_BLUEPRINT.md Blocker 2 frozen decision); got {self.insertion_mode!r}."
            )
        if self.init_std < 0:
            raise PromptConfigError(f"{type(self).__name__}.init_std must be >= 0, got {self.init_std!r}.")


def _init_prompt_param(cfg: _BasePromptConfig) -> nn.Parameter:
    """Build the learnable `(L, M, d)` prompt parameter, seeded if `cfg.seed` is set."""
    shape = (cfg.num_layers, cfg.num_prompts, cfg.prompt_dim)
    if cfg.seed is not None:
        gen = torch.Generator().manual_seed(cfg.seed)
        data = torch.empty(shape).normal_(mean=0.0, std=cfg.init_std, generator=gen)
    else:
        data = torch.empty(shape).normal_(mean=0.0, std=cfg.init_std)
    return nn.Parameter(data, requires_grad=True)


def _validate_layer_index(l: int, num_layers: int, cls_name: str) -> None:
    if not (0 <= l < num_layers):
        raise PromptShapeError(f"{cls_name}: layer index {l} out of range for num_layers={num_layers}.")


def _validate_sequence_input(x: torch.Tensor, expected_dim: int, cls_name: str) -> None:
    if x.dim() != 3:
        raise PromptShapeError(f"{cls_name}: expected input of shape (B, seq, d), got {tuple(x.shape)}.")
    if x.shape[-1] != expected_dim:
        raise PromptShapeError(
            f"{cls_name}: input last dim {x.shape[-1]} does not match prompt_dim {expected_dim}."
        )


def _concat_prompt(x: torch.Tensor, prompt_l: torch.Tensor, insert_after: int, cls_name: str) -> torch.Tensor:
    """Concatenate an `(M, d)` per-layer prompt slice into `(B, seq, d)` `x` at `insert_after`.

    `insert_after` is the number of leading tokens (e.g. 1 for `[CLS]`) the prompt is inserted
    immediately behind, matching Eq. 9 (`[X_CLS, g^(l), X_tokens]`) / Eq. 10
    (`[X_CLS, gV^(l), X_patches]`) token ordering.
    """
    if prompt_l.dim() != 2 or prompt_l.shape[-1] != x.shape[-1]:
        raise PromptShapeError(
            f"{cls_name}: prompt slice shape {tuple(prompt_l.shape)} incompatible with input dim {x.shape[-1]}."
        )
    batch = x.shape[0]
    prompt_b = prompt_l.unsqueeze(0).expand(batch, -1, -1)
    head, tail = x[:, :insert_after, :], x[:, insert_after:, :]
    return torch.cat([head, prompt_b, tail], dim=1)

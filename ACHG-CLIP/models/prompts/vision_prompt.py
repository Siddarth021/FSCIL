"""
models/prompts/vision_prompt.py
==================================

Stage 3 — learnable vision prompts (Eq. 10).

Behavior: Vision prompts are generated from a learnable parameter GV in R^{L x M x d} and
    inserted into the vision patch sequence at each Transformer layer as
    [X_CLS, gV^(l), X_patches].
Source: Section IV.A.3 "Learnable Prompts" / Eq. 10.
Evidence type: PAPER-FACT for the tensor shape and Eq. 10's literal concatenation notation.

Behavior: Vision prompt insertion mechanism is CONCATENATION (not the literal token
    replacement the paper's prose sentence immediately after Eq. 10 suggests: "Prompts
    directly replace the input of each layer").
Source: Eq. 10 notation vs. the contradicting prose right after it
    (FINAL_RESEARCH_DECISIONS.md Issue 5, TRUE BLOCKER;
    FINAL_IMPLEMENTATION_BLUEPRINT.md Blocker 2).
Evidence type: IMPLEMENTATION-CHOICE. This is the frozen project decision, NOT a paper fact:
    concatenation was adopted because (a) Eq. 10 uses identical notation to Eq. 9's undisputed
    text-side concatenation, and (b) CPE-CLIP (the closest reference the paper draws prompt
    mechanics from) assigns a "replace"-like behavior to its *language* branch, not vision --
    suggesting the paper's "replace" sentence is a likely mislabeling rather than an intended
    vision-specific design. Confidence: Medium (per configs/model/prompts.yaml /
    FINAL_IMPLEMENTATION_BLUEPRINT.md Blocker 2). Do NOT silently change this to a
    token-replacement path -- see "Architectural constraint" note below.
Implementation note: identical `_concat_prompt` mechanism to `text_prompt.py`, kept in a
    separate module (mirroring `configs/model/prompts.yaml`'s per-modality
    `vision_prompt_insertion_mode` key) so a future revisit of Blocker 2 (e.g. if author code
    or errata surface) only has to change this file, not `text_prompt.py`. The alternative
    "replacement" mechanism is intentionally NOT implemented anywhere in this module -- adding
    it would require re-opening Blocker 2, which is out of scope for Stage 3 (see task
    constraints: "preserve the frozen implementation choice; do not present it as a
    PAPER-FACT; keep the insertion mechanism modular").

Behavior: Prompt insertion point is immediately after the [CLS] token, before the image patch
    tokens.
Source: Eq. 10: X = [X_CLS, gV^(l), X_patches].
Evidence type: PAPER-FACT (position only; the surrounding insertion *mechanism* is the
    IMPLEMENTATION-CHOICE above).
Implementation note: encoded as `insert_after=1` in `_common._concat_prompt`, matching Eq. 7's
    vision sequence layout (`[x_CLS; patch embeddings]`).

Behavior: Prompt initialization distribution/std.
Source: not specified anywhere in the paper.
Evidence type: UNRESOLVED (paper) / IMPLEMENTATION-CHOICE (this module): Normal(0, 0.02),
    same convention as `text_prompt.py` for consistency across modalities.
Implementation note: `VisionPromptConfig.init_std`, overridable per instantiation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.prompts._common import (
    INSERTION_MODE,
    PromptShapeError,
    _BasePromptConfig,
    _concat_prompt,
    _init_prompt_param,
    _validate_layer_index,
    _validate_sequence_input,
)

#: Number of leading tokens (just [CLS]) the vision prompt is inserted behind (Eq. 10).
VISION_INSERT_AFTER = 1


@dataclass
class VisionPromptConfig(_BasePromptConfig):
    """Config for `VisionPromptInjector`. See module docstring for field provenance."""


class VisionPromptInjector(nn.Module):
    """Learnable vision prompts (Eq. 10): parameter `GV in R^{L x M x d}`, concatenation
    insertion (Blocker 2 IMPLEMENTATION-CHOICE -- see module docstring; NOT a paper fact).

    Usage (standalone, decoupled from `VisionEncoder`/`CLIPWrapper`):

        cfg = VisionPromptConfig(num_layers=L, num_prompts=M, prompt_dim=d)
        injector = VisionPromptInjector(cfg)
        x_with_prompt = injector(x, layer=l)  # x: (B, 1+m, d) -> (B, 1+M+m, d)
    """

    def __init__(self, config: VisionPromptConfig):
        super().__init__()
        self.config = config
        self.prompts = _init_prompt_param(config)  # GV: (L, M, d), requires_grad=True

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    @property
    def num_prompts(self) -> int:
        return self.config.num_prompts

    @property
    def prompt_dim(self) -> int:
        return self.config.prompt_dim

    def prompt_for_layer(self, l: int) -> torch.Tensor:
        """Return `gV^(l)`: `(M, d)` slice of `GV` for layer `l`."""
        _validate_layer_index(l, self.config.num_layers, type(self).__name__)
        return self.prompts[l]

    def forward(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """Insert `gV^(l)` into `x` per Eq. 10 (concatenation; Blocker 2): `[X_CLS, gV^(l), X_patches]`.

        Args:
            x: `(B, m', d)` vision token sequence for layer `layer` (already includes `[CLS]`
               as the first token, per Eq. 7/10's `X_CLS` term).
            layer: layer index `l` in `[0, num_layers)`.

        Returns:
            `(B, m' + M, d)` sequence with the prompt concatenated in.
        """
        _validate_sequence_input(x, self.config.prompt_dim, type(self).__name__)
        if x.shape[1] < VISION_INSERT_AFTER:
            raise PromptShapeError(
                f"{type(self).__name__}: input sequence length {x.shape[1]} too short to contain "
                f"a [CLS] token (need >= {VISION_INSERT_AFTER})."
            )
        gv_l = self.prompt_for_layer(layer)
        return _concat_prompt(x, gv_l, insert_after=VISION_INSERT_AFTER, cls_name=type(self).__name__)

    def extra_repr(self) -> str:
        return (
            f"num_layers={self.config.num_layers}, num_prompts={self.config.num_prompts}, "
            f"prompt_dim={self.config.prompt_dim}, insertion_mode={INSERTION_MODE!r} "
            f"(Blocker 2 IMPLEMENTATION-CHOICE, not PAPER-FACT)"
        )

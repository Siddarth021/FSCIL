"""
models/prompts/text_prompt.py
================================

Stage 3 — learnable text prompts (Eq. 9).

Behavior: Text prompts are generated from a learnable parameter G in R^{L x M x d} and
    inserted into the text token sequence at each Transformer layer as
    [X_CLS, g^(l), X_tokens].
Source: Section IV.A.3 "Learnable Prompts" / Eq. 9.
Evidence type: PAPER-FACT (tensor shape G in R^{L x M x d}, insertion point, and the
    concatenation notation itself); the *insertion mechanism* being genuine concatenation
    (rather than the vision branch's ambiguous "replace" wording) is corroborated as
    PAPER-FACT for the text branch specifically -- FINAL_IMPLEMENTATION_BLUEPRINT.md Blocker 2
    notes Eq. 9's text-side concatenation is "undisputed" (no contradicting prose exists for
    text, unlike vision).
Implementation note: `TextPromptInjector` is a standalone `nn.Module` operating on plain
    `(B, n, d)` token tensors -- it does not import or modify `models/clip/text_encoder.py`,
    per the "do not hard-code prompt behavior into the CLIP encoder itself" constraint. Wiring
    it into the actual per-layer forward pass of `TextEncoder` is left to a later stage; this
    module only guarantees the insertion op itself is correct, shape-tested, and swappable.

Behavior: Prompt insertion point is immediately after the [CLS] token, before the remaining
    text tokens.
Source: Eq. 9: X = [X_CLS, g^(l), X_tokens].
Evidence type: PAPER-FACT.
Implementation note: encoded as `insert_after=1` in `_common._concat_prompt`.

Behavior: M (num_learnable_prompts_M) = 1 is the reported final-model configuration.
Source: Section V.D.4 / Fig. 2(b), configs/model/prompts.yaml.
Evidence type: PAPER-FACT (as a reported *result*, not an architectural constraint -- this
    module supports any M >= 1; it does not hard-code M=1).
Implementation note: `TextPromptConfig.num_prompts` is caller-supplied, sourced from
    `configs/model/prompts.yaml` by the caller, not hard-coded here.

Behavior: Prompt initialization distribution/std.
Source: not specified anywhere in the paper.
Evidence type: UNRESOLVED (paper) / IMPLEMENTATION-CHOICE (this module): Normal(0, 0.02),
    matching common learnable soft-prompt practice.
Implementation note: `TextPromptConfig.init_std`, overridable per instantiation; never
    presented as a paper fact (see `configs/model/prompts.yaml`'s own provenance tags, which
    this module does not duplicate but is consistent with).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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

#: Number of leading tokens (just [CLS]) the text prompt is inserted behind (Eq. 9).
TEXT_INSERT_AFTER = 1


@dataclass
class TextPromptConfig(_BasePromptConfig):
    """Config for `TextPromptInjector`. See module docstring for field provenance."""


class TextPromptInjector(nn.Module):
    """Learnable text prompts (Eq. 9): parameter `G in R^{L x M x d}`, concatenation insertion.

    Usage (standalone, decoupled from `TextEncoder`/`CLIPWrapper` -- see module docstring):

        cfg = TextPromptConfig(num_layers=L, num_prompts=M, prompt_dim=d)
        injector = TextPromptInjector(cfg)
        x_with_prompt = injector(x, layer=l)   # x: (B, 1 + n_rest, d) -> (B, 1 + M + n_rest, d)
    """

    def __init__(self, config: TextPromptConfig):
        super().__init__()
        self.config = config
        self.prompts = _init_prompt_param(config)  # G: (L, M, d), requires_grad=True

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
        """Return `g^(l)`: `(M, d)` slice of `G` for layer `l`."""
        _validate_layer_index(l, self.config.num_layers, type(self).__name__)
        return self.prompts[l]

    def forward(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """Insert `g^(l)` into `x` per Eq. 9: `[X_CLS, g^(l), X_tokens]`.

        Args:
            x: `(B, n, d)` text token sequence for layer `layer` (already includes `[CLS]` as
               the first token, per Eq. 9's `X_CLS` term).
            layer: layer index `l` in `[0, num_layers)`.

        Returns:
            `(B, n + M, d)` sequence with the prompt concatenated in.
        """
        _validate_sequence_input(x, self.config.prompt_dim, type(self).__name__)
        if x.shape[1] < TEXT_INSERT_AFTER:
            raise PromptShapeError(
                f"{type(self).__name__}: input sequence length {x.shape[1]} too short to contain "
                f"a [CLS] token (need >= {TEXT_INSERT_AFTER})."
            )
        g_l = self.prompt_for_layer(layer)
        return _concat_prompt(x, g_l, insert_after=TEXT_INSERT_AFTER, cls_name=type(self).__name__)

    def extra_repr(self) -> str:
        return (
            f"num_layers={self.config.num_layers}, num_prompts={self.config.num_prompts}, "
            f"prompt_dim={self.config.prompt_dim}, insertion_mode={INSERTION_MODE!r}"
        )

"""
models/prompts/
=================

Stage 3 — learnable prompt subsystem (Eqs. 9-10).

Exposes `TextPromptInjector` and `VisionPromptInjector`. See each module's docstring for
paper traceability. Nothing here reaches into `models/clip/*` internals or CLIPWrapper --
these modules operate on plain `(B, seq, d)` tensors so they can be unit-tested in isolation
and wired in by a later stage without hard-coding prompt behavior into the CLIP encoder
itself (per project rule, see docs/implementation_progress.md Stage 3).
"""

from __future__ import annotations

from models.prompts.text_prompt import TextPromptInjector, TextPromptConfig
from models.prompts.vision_prompt import VisionPromptInjector, VisionPromptConfig
from models.prompts.mlp_bridge import PromptToNodeMLP, MLPBridgeConfig

__all__ = [
    "TextPromptInjector",
    "TextPromptConfig",
    "VisionPromptInjector",
    "VisionPromptConfig",
    "PromptToNodeMLP",
    "MLPBridgeConfig",
]

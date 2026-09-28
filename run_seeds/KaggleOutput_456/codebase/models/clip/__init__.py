"""
models/clip/
=============

Stage 2 — frozen CLIP backbone (Eqs. 1-8).

Public interface: import `CLIPConfig` / `CLIPWrapper` from this package rather than reaching
into `text_encoder.py` / `vision_encoder.py` / `attention.py` / `transformer_block.py`
directly. `mock.py`'s `build_mock_clip_wrapper` is the supported way to get a small
deterministic backbone for tests (see its module docstring for why no real pretrained
backbone is available yet).
"""

from models.clip.clip_wrapper import CLIPConfig, CLIPConfigError, CLIPWrapper
from models.clip.mock import build_mock_clip_config, build_mock_clip_wrapper

__all__ = [
    "CLIPConfig",
    "CLIPConfigError",
    "CLIPWrapper",
    "build_mock_clip_config",
    "build_mock_clip_wrapper",
]

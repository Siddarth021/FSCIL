"""
tests/test_prompt_insertion.py
=================================

Stage 3 tests: learnable text/vision prompt subsystem (Eqs. 9-10).
"""

from __future__ import annotations

import unittest

import torch

from models.prompts._common import PromptConfigError, PromptShapeError
from models.prompts.text_prompt import TextPromptConfig, TextPromptInjector
from models.prompts.vision_prompt import VisionPromptConfig, VisionPromptInjector

L, M, D = 3, 2, 8  # small synthetic dims, mirrors models/clip/mock.py style


def _make_text(seed=None):
    cfg = TextPromptConfig(num_layers=L, num_prompts=M, prompt_dim=D, seed=seed)
    return TextPromptInjector(cfg)


def _make_vision(seed=None):
    cfg = VisionPromptConfig(num_layers=L, num_prompts=M, prompt_dim=D, seed=seed)
    return VisionPromptInjector(cfg)


class TestTextPromptInit(unittest.TestCase):
    def test_shape(self):
        inj = _make_text()
        self.assertEqual(tuple(inj.prompts.shape), (L, M, D))

    def test_requires_grad(self):
        inj = _make_text()
        self.assertTrue(inj.prompts.requires_grad)

    def test_invalid_dims_rejected(self):
        with self.assertRaises(PromptConfigError):
            TextPromptConfig(num_layers=0, num_prompts=M, prompt_dim=D)
        with self.assertRaises(PromptConfigError):
            TextPromptConfig(num_layers=L, num_prompts=-1, prompt_dim=D)


class TestVisionPromptInit(unittest.TestCase):
    def test_shape(self):
        inj = _make_vision()
        self.assertEqual(tuple(inj.prompts.shape), (L, M, D))

    def test_requires_grad(self):
        inj = _make_vision()
        self.assertTrue(inj.prompts.requires_grad)


class TestDeterministicInit(unittest.TestCase):
    def test_same_seed_same_values(self):
        a = _make_text(seed=123)
        b = _make_text(seed=123)
        self.assertTrue(torch.allclose(a.prompts, b.prompts))

    def test_different_seed_different_values(self):
        a = _make_text(seed=1)
        b = _make_text(seed=2)
        self.assertFalse(torch.allclose(a.prompts, b.prompts))

    def test_vision_same_seed_same_values(self):
        a = _make_vision(seed=42)
        b = _make_vision(seed=42)
        self.assertTrue(torch.allclose(a.prompts, b.prompts))


class TestTextPromptInsertion(unittest.TestCase):
    def test_concatenation_grows_sequence_by_M(self):
        inj = _make_text()
        B, n = 4, 5  # includes [CLS] as token 0
        x = torch.randn(B, n, D)
        out = inj(x, layer=0)
        self.assertEqual(out.shape, (B, n + M, D))

    def test_cls_token_preserved_at_position_0(self):
        inj = _make_text()
        x = torch.randn(2, 4, D)
        out = inj(x, layer=1)
        self.assertTrue(torch.allclose(out[:, 0, :], x[:, 0, :]))

    def test_prompt_values_inserted_immediately_after_cls(self):
        inj = _make_text()
        x = torch.randn(2, 4, D)
        out = inj(x, layer=1)
        g1 = inj.prompt_for_layer(1)
        for b in range(2):
            self.assertTrue(torch.allclose(out[b, 1 : 1 + M, :], g1))

    def test_remaining_tokens_preserved_after_prompt(self):
        inj = _make_text()
        x = torch.randn(2, 4, D)
        out = inj(x, layer=0)
        self.assertTrue(torch.allclose(out[:, 1 + M :, :], x[:, 1:, :]))

    def test_not_a_replacement(self):
        """Explicitly verify concatenation, not replacement: output length > input length,
        and every original token value is still present somewhere in the output."""
        inj = _make_text()
        x = torch.randn(1, 3, D)
        out = inj(x, layer=0)
        self.assertGreater(out.shape[1], x.shape[1])
        self.assertEqual(out.shape[1], x.shape[1] + M)
        # original CLS + tokens both fully preserved (concatenation keeps everything)
        self.assertTrue(torch.allclose(out[:, 0, :], x[:, 0, :]))
        self.assertTrue(torch.allclose(out[:, 1 + M :, :], x[:, 1:, :]))


class TestVisionPromptInsertion(unittest.TestCase):
    def test_concatenation_grows_sequence_by_M(self):
        inj = _make_vision()
        B, seq = 3, 6  # [CLS] + 5 patches
        x = torch.randn(B, seq, D)
        out = inj(x, layer=0)
        self.assertEqual(out.shape, (B, seq + M, D))

    def test_not_a_replacement(self):
        inj = _make_vision()
        x = torch.randn(2, 5, D)
        out = inj(x, layer=2)
        self.assertEqual(out.shape[1], x.shape[1] + M)
        self.assertTrue(torch.allclose(out[:, 0, :], x[:, 0, :]))
        self.assertTrue(torch.allclose(out[:, 1 + M :, :], x[:, 1:, :]))

    def test_cls_and_patches_preserved(self):
        inj = _make_vision()
        x = torch.randn(2, 5, D)
        out = inj(x, layer=0)
        gv0 = inj.prompt_for_layer(0)
        self.assertTrue(torch.allclose(out[:, 0, :], x[:, 0, :]))
        for b in range(2):
            self.assertTrue(torch.allclose(out[b, 1 : 1 + M, :], gv0))
        self.assertTrue(torch.allclose(out[:, 1 + M :, :], x[:, 1:, :]))


class TestBatchAndSeqHandling(unittest.TestCase):
    def test_various_batch_sizes(self):
        inj = _make_text()
        for B in (1, 2, 7):
            x = torch.randn(B, 5, D)
            out = inj(x, layer=0)
            self.assertEqual(out.shape[0], B)

    def test_various_sequence_lengths(self):
        inj = _make_vision()
        for seq in (1, 4, 10):
            x = torch.randn(2, seq, D)
            out = inj(x, layer=0)
            self.assertEqual(out.shape[1], seq + M)


class TestMultiplePromptTokens(unittest.TestCase):
    def test_M_greater_than_one(self):
        for m in (1, 2, 5):
            cfg = TextPromptConfig(num_layers=L, num_prompts=m, prompt_dim=D)
            inj = TextPromptInjector(cfg)
            x = torch.randn(2, 4, D)
            out = inj(x, layer=0)
            self.assertEqual(out.shape[1], 4 + m)
            self.assertEqual(inj.prompts.shape, (L, m, D))


class TestInvalidInputHandling(unittest.TestCase):
    def test_wrong_last_dim_rejected(self):
        inj = _make_text()
        x = torch.randn(2, 4, D + 1)
        with self.assertRaises(PromptShapeError):
            inj(x, layer=0)

    def test_non_3d_input_rejected(self):
        inj = _make_text()
        x = torch.randn(4, D)
        with self.assertRaises(PromptShapeError):
            inj(x, layer=0)

    def test_layer_out_of_range_rejected(self):
        inj = _make_text()
        x = torch.randn(2, 4, D)
        with self.assertRaises(PromptShapeError):
            inj(x, layer=L)
        with self.assertRaises(PromptShapeError):
            inj(x, layer=-1)

    def test_empty_batch_rejected_by_downstream_shape(self):
        # Zero-length sequence (no room for [CLS]) must be rejected explicitly.
        inj = _make_text()
        x = torch.randn(2, 0, D)
        with self.assertRaises(PromptShapeError):
            inj(x, layer=0)


class TestConfigValidation(unittest.TestCase):
    def test_insertion_mode_locked_to_concatenate(self):
        with self.assertRaises(PromptConfigError):
            TextPromptConfig(num_layers=L, num_prompts=M, prompt_dim=D, insertion_mode="replace")
        with self.assertRaises(PromptConfigError):
            VisionPromptConfig(num_layers=L, num_prompts=M, prompt_dim=D, insertion_mode="replace")

    def test_negative_init_std_rejected(self):
        with self.assertRaises(PromptConfigError):
            TextPromptConfig(num_layers=L, num_prompts=M, prompt_dim=D, init_std=-0.1)


class TestStage2Compatibility(unittest.TestCase):
    """Prompt injectors must accept the exact (B, seq, d_model) shapes Stage 2's
    TextEncoder/VisionEncoder (via mock.py's synthetic dims) produce, without importing
    CLIPWrapper internals (decoupled per the architectural constraint)."""

    def test_matches_mock_clip_token_dim(self):
        from models.clip.mock import MOCK_D_MODEL, MOCK_NUM_LAYERS

        cfg = TextPromptConfig(num_layers=MOCK_NUM_LAYERS, num_prompts=1, prompt_dim=MOCK_D_MODEL)
        inj = TextPromptInjector(cfg)
        x = torch.randn(2, 6, MOCK_D_MODEL)  # e.g. [CLS] + 5 tokens
        out = inj(x, layer=0)
        self.assertEqual(out.shape, (2, 7, MOCK_D_MODEL))

    def test_vision_matches_mock_clip_token_dim(self):
        from models.clip.mock import MOCK_D_MODEL, MOCK_NUM_LAYERS

        cfg = VisionPromptConfig(num_layers=MOCK_NUM_LAYERS, num_prompts=1, prompt_dim=MOCK_D_MODEL)
        inj = VisionPromptInjector(cfg)
        x = torch.randn(2, 10, MOCK_D_MODEL)  # [CLS] + 9 patches
        out = inj(x, layer=MOCK_NUM_LAYERS - 1)
        self.assertEqual(out.shape, (2, 11, MOCK_D_MODEL))


if __name__ == "__main__":
    unittest.main()

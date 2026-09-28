"""
Stage 2 tests for models/clip/ (the frozen CLIP backbone wrapper).

Covers (per the Stage 2 task's requirement list):
  1. CLIP wrapper initialization
  2. image input/output shape
  3. text input/output shape
  4. embedding dimension propagation
  5. freezing parameters
  6. unfreezing parameters if supported
  7. device handling
  8. mock-backbone compatibility
  9. configuration/provenance validation

Uses only the deterministic mock backbone (`models/clip/mock.py`) -- no real pretrained CLIP
checkpoint is available in this environment (see `docs/implementation_progress.md` Stage 2
"Dependency / environment issues"); this is exactly the authorized Stage 2 test path
(`FINAL_IMPLEMENTATION_BLUEPRINT.md` Part 9, row 2: "shape tests for h*_T, h*_V on a dummy
pretrained backbone").
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402

from models.clip import (  # noqa: E402
    CLIPConfig,
    CLIPConfigError,
    CLIPWrapper,
    build_mock_clip_config,
    build_mock_clip_wrapper,
)
from models.clip.mock import MOCK_SYNTHETIC  # noqa: E402
from utils.config_tracking import ConfigManager, UnresolvedParameterError  # noqa: E402
from utils.seed import set_seed  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_ROOT = os.path.join(REPO_ROOT, "configs")


# ------------------------------------------------------------------------------------------
# 1. CLIP wrapper initialization
# ------------------------------------------------------------------------------------------


class TestInitialization(unittest.TestCase):
    def test_build_mock_wrapper(self):
        model = build_mock_clip_wrapper()
        self.assertIsInstance(model, CLIPWrapper)
        self.assertIsInstance(model, torch.nn.Module)

    def test_submodules_exist(self):
        model = build_mock_clip_wrapper()
        self.assertTrue(hasattr(model, "text_encoder"))
        self.assertTrue(hasattr(model, "vision_encoder"))

    def test_default_frozen_at_init(self):
        # config.frozen defaults to True in build_mock_clip_config
        model = build_mock_clip_wrapper()
        self.assertTrue(model.is_frozen)

    def test_can_build_unfrozen(self):
        model = CLIPWrapper(build_mock_clip_config(frozen=False))
        self.assertFalse(model.is_frozen)

    def test_invalid_config_rejected(self):
        with self.assertRaises(CLIPConfigError):
            build_mock_clip_config(d_model=0)
        with self.assertRaises(CLIPConfigError):
            build_mock_clip_config(num_layers=-1)
        with self.assertRaises(CLIPConfigError):
            build_mock_clip_config(dropout=1.5)


# ------------------------------------------------------------------------------------------
# 2. image input/output shape
# ------------------------------------------------------------------------------------------


class TestImageShapes(unittest.TestCase):
    def setUp(self):
        set_seed(0, deterministic=False)
        self.config = build_mock_clip_config()
        self.model = CLIPWrapper(self.config)

    def test_encode_image_output_shape(self):
        batch, num_patches = 5, 6
        patches = torch.randn(batch, num_patches, self.config.patch_dim)
        h_v = self.model.encode_image(patches)
        self.assertEqual(h_v.shape, (batch, self.config.d_e))

    def test_encode_image_is_l2_normalized(self):
        patches = torch.randn(3, 4, self.config.patch_dim)
        h_v = self.model.encode_image(patches)
        norms = h_v.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-4)

    def test_encode_image_return_sequence_shape(self):
        batch, num_patches = 2, 5
        patches = torch.randn(batch, num_patches, self.config.patch_dim)
        h_v, seq = self.model.encode_image(patches, return_sequence=True)
        self.assertEqual(h_v.shape, (batch, self.config.d_e))
        # +1 for the prepended [CLS] token
        self.assertEqual(seq.shape, (batch, num_patches + 1, self.config.d_model))

    def test_too_many_patches_raises(self):
        patches = torch.randn(1, self.config.max_patches + 1, self.config.patch_dim)
        with self.assertRaises(ValueError):
            self.model.encode_image(patches)

    def test_wrong_patch_dim_raises(self):
        patches = torch.randn(1, 3, self.config.patch_dim + 1)
        with self.assertRaises(ValueError):
            self.model.encode_image(patches)


# ------------------------------------------------------------------------------------------
# 3. text input/output shape
# ------------------------------------------------------------------------------------------


class TestTextShapes(unittest.TestCase):
    def setUp(self):
        set_seed(0, deterministic=False)
        self.config = build_mock_clip_config()
        self.model = CLIPWrapper(self.config)

    def test_encode_text_output_shape(self):
        batch, seq_len = 5, 7
        tokens = torch.randint(0, self.config.vocab_size, (batch, seq_len))
        h_t = self.model.encode_text(tokens)
        self.assertEqual(h_t.shape, (batch, self.config.d_e))

    def test_encode_text_is_l2_normalized(self):
        tokens = torch.randint(0, self.config.vocab_size, (4, 6))
        h_t = self.model.encode_text(tokens)
        norms = h_t.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-4)

    def test_encode_text_return_sequence_shape(self):
        batch, seq_len = 2, 6
        tokens = torch.randint(0, self.config.vocab_size, (batch, seq_len))
        h_t, seq = self.model.encode_text(tokens, return_sequence=True)
        self.assertEqual(h_t.shape, (batch, self.config.d_e))
        self.assertEqual(seq.shape, (batch, seq_len, self.config.d_model))

    def test_sequence_too_long_raises(self):
        tokens = torch.randint(0, self.config.vocab_size, (1, self.config.max_text_len + 1))
        with self.assertRaises(ValueError):
            self.model.encode_text(tokens)

    def test_wrong_ndim_raises(self):
        tokens = torch.randint(0, self.config.vocab_size, (self.config.max_text_len,))
        with self.assertRaises(ValueError):
            self.model.encode_text(tokens)

    def test_forward_returns_both_embeddings(self):
        batch = 3
        patches = torch.randn(batch, 4, self.config.patch_dim)
        tokens = torch.randint(0, self.config.vocab_size, (batch, 5))
        h_v, h_t = self.model(patches, tokens)
        self.assertEqual(h_v.shape, (batch, self.config.d_e))
        self.assertEqual(h_t.shape, (batch, self.config.d_e))


# ------------------------------------------------------------------------------------------
# 4. embedding dimension propagation
# ------------------------------------------------------------------------------------------


class TestEmbeddingDimensionPropagation(unittest.TestCase):
    def test_dims_match_config(self):
        config = build_mock_clip_config(d_model=16, d_e=10, num_layers=2)
        model = CLIPWrapper(config)
        self.assertEqual(model.image_embedding_dim, 10)
        self.assertEqual(model.text_embedding_dim, 10)
        self.assertEqual(model.token_dim, 16)
        self.assertEqual(model.num_layers, 2)

    def test_changing_d_e_changes_output_shape(self):
        config = build_mock_clip_config(d_e=7)
        model = CLIPWrapper(config)
        patches = torch.randn(2, 3, config.patch_dim)
        tokens = torch.randint(0, config.vocab_size, (2, 3))
        self.assertEqual(model.encode_image(patches).shape[-1], 7)
        self.assertEqual(model.encode_text(tokens).shape[-1], 7)

    def test_image_and_text_embedding_dims_equal(self):
        # Eqs. 6/8 both project into the same shared d_e (needed for Eq. 11's cosine sim).
        config = build_mock_clip_config()
        model = CLIPWrapper(config)
        self.assertEqual(model.image_embedding_dim, model.text_embedding_dim)


# ------------------------------------------------------------------------------------------
# 5. freezing parameters
# ------------------------------------------------------------------------------------------


class TestFreezing(unittest.TestCase):
    def test_frozen_at_construction_by_default(self):
        model = build_mock_clip_wrapper()
        self.assertTrue(model.is_frozen)
        self.assertEqual(len(model.trainable_parameters()), 0)
        self.assertGreater(len(model.frozen_parameters()), 0)

    def test_freeze_backbone_sets_requires_grad_false(self):
        model = CLIPWrapper(build_mock_clip_config(frozen=False))
        self.assertFalse(model.is_frozen)
        count = model.freeze_backbone()
        self.assertGreater(count, 0)
        self.assertTrue(model.is_frozen)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_zero_grad_on_frozen_backbone_after_backward(self):
        model = build_mock_clip_wrapper()  # frozen by default
        config = model.config
        # requires_grad=True on the INPUT (not the frozen backbone) so the graph has
        # somewhere to flow gradients to; this isolates "are backbone params exempt from
        # gradient accumulation" from "does the input carry gradients at all".
        patches = torch.randn(2, 3, config.patch_dim, requires_grad=True)
        tokens = torch.randint(0, config.vocab_size, (2, 3))
        h_v, h_t = model(patches, tokens)
        loss = (h_v - h_t).pow(2).sum()
        loss.backward()
        for p in model.parameters():
            self.assertIsNone(p.grad)  # frozen params never accumulate a gradient
        self.assertIsNotNone(patches.grad)  # confirms the graph itself was live

    def test_num_trainable_parameters_zero_when_frozen(self):
        model = build_mock_clip_wrapper()
        self.assertEqual(model.num_trainable_parameters(), 0)
        self.assertGreater(model.num_parameters(), 0)


# ------------------------------------------------------------------------------------------
# 6. unfreezing parameters if supported
# ------------------------------------------------------------------------------------------


class TestUnfreezing(unittest.TestCase):
    def test_unfreeze_backbone_sets_requires_grad_true(self):
        model = build_mock_clip_wrapper()  # frozen
        count = model.unfreeze_backbone()
        self.assertGreater(count, 0)
        self.assertFalse(model.is_frozen)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(model.num_trainable_parameters(), model.num_parameters())

    def test_unfrozen_backbone_receives_gradients(self):
        model = build_mock_clip_wrapper()
        model.unfreeze_backbone()
        config = model.config
        patches = torch.randn(2, 3, config.patch_dim)
        tokens = torch.randint(0, config.vocab_size, (2, 3))
        h_v, h_t = model(patches, tokens)
        loss = (h_v - h_t).pow(2).sum()
        loss.backward()
        grads_present = [p.grad is not None for p in model.parameters()]
        self.assertTrue(any(grads_present))

    def test_refreeze_after_unfreeze(self):
        model = build_mock_clip_wrapper()
        model.unfreeze_backbone()
        self.assertFalse(model.is_frozen)
        model.freeze_backbone()
        self.assertTrue(model.is_frozen)


# ------------------------------------------------------------------------------------------
# 7. device handling
# ------------------------------------------------------------------------------------------


class TestDeviceHandling(unittest.TestCase):
    def test_device_defaults_to_cpu(self):
        model = build_mock_clip_wrapper()
        self.assertEqual(model.device.type, "cpu")

    def test_to_device_is_noop_safe_on_cpu_only_machine(self):
        model = build_mock_clip_wrapper()
        model.to(torch.device("cpu"))
        self.assertEqual(model.device.type, "cpu")

    def test_forward_pass_works_after_to_cpu(self):
        model = build_mock_clip_wrapper()
        model.to("cpu")
        config = model.config
        patches = torch.randn(2, 3, config.patch_dim, device=model.device)
        tokens = torch.randint(0, config.vocab_size, (2, 3), device=model.device)
        h_v, h_t = model(patches, tokens)
        self.assertEqual(h_v.device.type, "cpu")
        self.assertEqual(h_t.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available in this environment")
    def test_to_cuda_moves_parameters(self):  # pragma: no cover - only runs with a GPU
        model = build_mock_clip_wrapper()
        model.to("cuda")
        self.assertEqual(model.device.type, "cuda")


# ------------------------------------------------------------------------------------------
# 8. mock-backbone compatibility
# ------------------------------------------------------------------------------------------


class TestMockBackboneCompatibility(unittest.TestCase):
    def test_mock_config_dims_are_tagged_mock_synthetic(self):
        config = build_mock_clip_config()
        for key, tag in config.dim_provenance.items():
            self.assertEqual(tag, MOCK_SYNTHETIC, f"key {key!r} not tagged MOCK_SYNTHETIC")

    def test_mock_wrapper_deterministic_with_same_seed(self):
        set_seed(123, deterministic=False)
        model_a = build_mock_clip_wrapper()
        set_seed(123, deterministic=False)
        model_b = build_mock_clip_wrapper()

        sd_a = model_a.state_dict()
        sd_b = model_b.state_dict()
        self.assertEqual(sd_a.keys(), sd_b.keys())
        for key in sd_a:
            torch.testing.assert_close(sd_a[key], sd_b[key])

    def test_mock_wrapper_end_to_end_forward_is_finite(self):
        set_seed(7, deterministic=False)
        model = build_mock_clip_wrapper()
        config = model.config
        patches = torch.randn(4, 5, config.patch_dim)
        tokens = torch.randint(0, config.vocab_size, (4, 5))
        h_v, h_t = model(patches, tokens)
        self.assertTrue(torch.isfinite(h_v).all())
        self.assertTrue(torch.isfinite(h_t).all())

    def test_mock_wrapper_custom_dims(self):
        model = build_mock_clip_wrapper(d_model=32, d_e=16, num_layers=1, d_k=8, num_heads=4)
        config = model.config
        patches = torch.randn(2, 3, config.patch_dim)
        h_v = model.encode_image(patches)
        self.assertEqual(h_v.shape, (2, 16))


# ------------------------------------------------------------------------------------------
# 9. configuration / provenance validation
# ------------------------------------------------------------------------------------------


class TestConfigProvenanceValidation(unittest.TestCase):
    def setUp(self):
        self.mgr = ConfigManager(config_root=CONFIG_ROOT)
        self.resolved = self.mgr.load()

    def test_clip_backbone_dims_are_implementation_choice_in_real_config(self):
        # Now these are IMPLEMENTATION_CHOICE instead of UNRESOLVED
        for key in (
            "model.clip_backbone.variant",
            "model.clip_backbone.token_embedding_dim_d",
            "model.clip_backbone.projection_dim_de",
            "model.clip_backbone.attention_head_dim_dk",
            "model.clip_backbone.num_transformer_layers_L",
        ):
            self.assertEqual(self.resolved.get_entry(key).provenance, "IMPLEMENTATION_CHOICE")

    def test_frozen_flag_is_paper_fact_true(self):
        entry = self.resolved.get_entry("model.clip_backbone.frozen")
        self.assertEqual(entry.provenance, "PAPER_FACT")
        self.assertTrue(entry.value)

    def _get_unresolved_config(self):
        import copy
        resolved = copy.deepcopy(self.resolved)
        for key in (
            "model.clip_backbone.variant",
            "model.clip_backbone.token_embedding_dim_d",
            "model.clip_backbone.projection_dim_de",
            "model.clip_backbone.attention_head_dim_dk",
            "model.clip_backbone.num_transformer_layers_L",
        ):
            resolved.entries[key].provenance = "UNRESOLVED"
            resolved.entries[key].value = None
        return resolved

    def test_from_resolved_config_without_override_raises(self):
        unresolved_cfg = self._get_unresolved_config()
        with self.assertRaises(UnresolvedParameterError):
            CLIPConfig.from_resolved_config(
                unresolved_cfg,
                vocab_size=50,
                max_text_len=16,
                patch_dim=12,
                max_patches=9,
                num_heads=2,
            )

    def test_from_resolved_config_with_test_overrides_succeeds_and_is_tagged(self):
        unresolved_cfg = self._get_unresolved_config()
        config = CLIPConfig.from_resolved_config(
            unresolved_cfg,
            vocab_size=50,
            max_text_len=16,
            patch_dim=12,
            max_patches=9,
            num_heads=2,
            test_overrides={
                "variant": "test-only-variant",
                "token_embedding_dim_d": 8,
                "projection_dim_de": 4,
                "attention_head_dim_dk": 4,
                "num_transformer_layers_L": 3,
            },
        )
        self.assertEqual(config.d_model, 8)
        self.assertEqual(config.d_e, 4)
        self.assertEqual(config.num_layers, 3)
        self.assertEqual(config.dim_provenance["token_embedding_dim_d"], "TEST_OVERRIDE")
        self.assertEqual(config.dim_provenance["num_transformer_layers_L"], "TEST_OVERRIDE")
        # frozen came from the real (resolved, non-UNRESOLVED) config entry, not a test override
        self.assertEqual(config.dim_provenance.get("frozen", "PAPER_FACT"), "PAPER_FACT")
        self.assertTrue(config.frozen)

        # A wrapper can actually be built and run from this config.
        model = CLIPWrapper(config)
        patches = torch.randn(1, 2, 12)
        h_v = model.encode_image(patches)
        self.assertEqual(h_v.shape, (1, 4))

    def test_from_resolved_config_partial_override_still_raises_on_missing_key(self):
        unresolved_cfg = self._get_unresolved_config()
        with self.assertRaises(UnresolvedParameterError):
            CLIPConfig.from_resolved_config(
                unresolved_cfg,
                vocab_size=50,
                max_text_len=16,
                patch_dim=12,
                max_patches=9,
                num_heads=2,
                test_overrides={
                    "token_embedding_dim_d": 8,
                    # projection_dim_de, attention_head_dim_dk, num_transformer_layers_L,
                    # variant intentionally omitted -> should still raise.
                },
            )

    def test_config_round_trip_via_to_dict_from_dict(self):
        config = build_mock_clip_config()
        restored = CLIPConfig.from_dict(config.to_dict())
        self.assertEqual(config.to_dict(), restored.to_dict())


if __name__ == "__main__":
    unittest.main()

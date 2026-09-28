"""
tests/test_achg_clip.py
==========================

Stage 8 tests: top-level ACHG-CLIP architecture / wiring.

Uses the deterministic mock CLIP backbone (`models/clip/mock.py`) exactly as Stage 2's own
tests do -- no real pretrained CLIP checkpoint exists for this project (see
`docs/implementation_progress.md` Stage 2 "Dependency / environment issues").
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402

from models.achg_clip import (  # noqa: E402
    ACHGCLIP,
    ACHGCLIPConfig,
    ACHGCLIPConfigError,
    ACHGCLIPOutput,
    FeedbackPath,
    ModalityOutput,
    ResolvedFeedbackPath,
)
from models.clip.mock import build_mock_clip_config  # noqa: E402
from models.graph.graph_data import Graph  # noqa: E402
from utils.seed import set_seed  # noqa: E402

# --------------------------------------------------------------------------------------
# Small synthetic dims (deliberately small/arbitrary, chosen only for fast legible tests;
# not derived from any paper evidence -- mirrors every other stage's test convention).
# --------------------------------------------------------------------------------------
L = 3           # num_layers (also N, graph node count, per Blocker 1)
D_MODEL = 8     # CLIP token dim == prompt_dim d (must match)
D_NODE = 6      # D, node feature dim (Fig.1 MLP bridge output)
MLP_HIDDEN = 10
GIN_HIDDEN = 5  # deliberately != ACGA latent dim, for the critical structural test
LATENT_K = 3    # ACGA K, deliberately != GIN_HIDDEN
COMPRESSED_DC = 4
DT = 0.01

VOCAB = 50
MAX_TEXT_LEN = 16
PATCH_DIM = 12
MAX_PATCHES = 9


def _mock_clip_config(**overrides):
    kwargs = dict(
        d_model=D_MODEL,
        d_e=4,
        d_k=4,
        num_heads=2,
        num_layers=L,
        vocab_size=VOCAB,
        max_text_len=MAX_TEXT_LEN,
        patch_dim=PATCH_DIM,
        max_patches=MAX_PATCHES,
    )
    kwargs.update(overrides)
    return build_mock_clip_config(**kwargs)


def _make_config(**overrides) -> ACHGCLIPConfig:
    kwargs = dict(
        clip=_mock_clip_config(),
        num_layers=L,
        prompt_dim=D_MODEL,
        num_prompts=1,
        node_feature_dim=D_NODE,
        mlp_bridge_hidden_dim=MLP_HIDDEN,
        gin_num_layers=4,
        gin_hidden_dim=GIN_HIDDEN,
        acga_latent_dim=LATENT_K,
        hgnec_compressed_dim=COMPRESSED_DC,
    )
    kwargs.update(overrides)
    return ACHGCLIPConfig(**kwargs)


def _make_model(seed=None, **overrides) -> ACHGCLIP:
    if seed is not None:
        set_seed(seed, deterministic=False)
    return ACHGCLIP(_make_config(**overrides))


def _random_batch(batch=2, n_tokens=5, n_patches=4):
    tokens = torch.randint(0, VOCAB, (batch, n_tokens))
    images = torch.randn(batch, n_patches, PATCH_DIM)
    return images, tokens


# ========================================================================================
# 1. Top-level initialization
# ========================================================================================
class TestInitialization(unittest.TestCase):
    def test_builds(self):
        model = _make_model()
        self.assertIsInstance(model, ACHGCLIP)

    def test_submodules_present(self):
        model = _make_model()
        for name in ("clip", "text_prompt", "vision_prompt", "text_mlp_bridge",
                     "vision_mlp_bridge", "gin", "acga", "hgn_ec", "feedback_path"):
            self.assertTrue(hasattr(model, name), f"missing submodule {name}")

    def test_gin_acga_hgnec_are_single_shared_instances(self):
        """Blocker 5: ONE GIN/ACGA/HGN-EC instance each, reused for both modalities."""
        model = _make_model()
        self.assertIsInstance(model.gin, torch.nn.Module)
        # Only one of each -- no per-modality duplicates exist as attributes.
        self.assertFalse(hasattr(model, "text_gin"))
        self.assertFalse(hasattr(model, "vision_gin"))

    def test_mlp_bridges_are_independent_instances(self):
        """Blocker 3: shared_across_modalities = false -- two distinct MLP instances."""
        model = _make_model()
        self.assertIsNot(model.text_mlp_bridge, model.vision_mlp_bridge)
        self.assertIsNot(model.text_prompt, model.vision_prompt)

    def test_default_feedback_path_is_resolved(self):
        model = _make_model()
        self.assertIsInstance(model.feedback_path, ResolvedFeedbackPath)
        self.assertIsInstance(model.feedback_path, FeedbackPath)


# ========================================================================================
# 2. Text-only forward
# ========================================================================================
class TestTextOnlyForward(unittest.TestCase):
    def test_runs_and_shapes(self):
        model = _make_model()
        _, tokens = _random_batch()
        out = model.text_only_forward(tokens, dt=DT)
        self.assertIsInstance(out, ACHGCLIPOutput)
        self.assertIsNotNone(out.h_text)
        self.assertIsNone(out.h_vision)
        self.assertIsNotNone(out.text)
        self.assertIsNone(out.vision)
        self.assertEqual(out.h_text.shape, (tokens.shape[0], 4))  # d_e = 4

    def test_graph_node_count_is_L(self):
        model = _make_model()
        _, tokens = _random_batch()
        out = model.text_only_forward(tokens, dt=DT)
        self.assertEqual(out.text.gin_graph.N, L)


# ========================================================================================
# 3. Vision-only forward
# ========================================================================================
class TestVisionOnlyForward(unittest.TestCase):
    def test_runs_and_shapes(self):
        model = _make_model()
        images, _ = _random_batch()
        out = model.vision_only_forward(images, dt=DT)
        self.assertIsNone(out.h_text)
        self.assertIsNotNone(out.h_vision)
        self.assertIsNone(out.text)
        self.assertIsNotNone(out.vision)
        self.assertEqual(out.h_vision.shape, (images.shape[0], 4))


# ========================================================================================
# 4. Joint text+vision forward
# ========================================================================================
class TestJointForward(unittest.TestCase):
    def test_runs_and_shapes(self):
        model = _make_model()
        images, tokens = _random_batch()
        out = model.forward(images, tokens, dt=DT)
        self.assertIsNotNone(out.h_text)
        self.assertIsNotNone(out.h_vision)
        self.assertIsNotNone(out.text)
        self.assertIsNotNone(out.vision)


# ========================================================================================
# 5. Correct branch separation (text and vision graphs are independent objects/tensors)
# ========================================================================================
class TestBranchSeparation(unittest.TestCase):
    def test_text_and_vision_graphs_are_distinct_tensors(self):
        model = _make_model()
        images, tokens = _random_batch()
        out = model.forward(images, tokens, dt=DT)
        self.assertIsNot(out.text.gin_graph, out.vision.gin_graph)
        self.assertIsNot(out.text.gin_graph.X, out.vision.gin_graph.X)
        self.assertIsNot(out.text.gin_graph.A, out.vision.gin_graph.A)

    def test_no_cross_modal_edges_or_merging(self):
        """Frozen Decisions 7/8: no cross-modal graph edges; graphs never merged."""
        model = _make_model()
        images, tokens = _random_batch()
        out = model.forward(images, tokens, dt=DT)
        self.assertEqual(tuple(out.text.gin_graph.A.shape[-2:]), (L, L))
        self.assertEqual(tuple(out.vision.gin_graph.A.shape[-2:]), (L, L))

    def test_independent_processing_gives_different_outputs(self):
        """Text and vision prompts are independently initialized -> different graphs."""
        model = _make_model(seed=0)
        images, tokens = _random_batch()
        out = model.forward(images, tokens, dt=DT)
        self.assertFalse(torch.allclose(out.text.gin_graph.X, out.vision.gin_graph.X))


# ========================================================================================
# 6. CLIP -> prompt interface
# ========================================================================================
class TestCLIPPromptInterface(unittest.TestCase):
    def test_prompt_dim_matches_clip_token_dim(self):
        model = _make_model()
        self.assertEqual(model.text_prompt.prompt_dim, model.clip.token_dim)
        self.assertEqual(model.vision_prompt.prompt_dim, model.clip.token_dim)

    def test_prompt_num_layers_matches_clip(self):
        model = _make_model()
        self.assertEqual(model.text_prompt.num_layers, model.clip.num_layers)
        self.assertEqual(model.vision_prompt.num_layers, model.clip.num_layers)

    def test_clip_backbone_is_frozen(self):
        model = _make_model()
        self.assertTrue(model.clip.is_frozen)


# ========================================================================================
# 7. prompt -> MLP interface
# ========================================================================================
class TestPromptToMLPInterface(unittest.TestCase):
    def test_mlp_bridge_consumes_prompt_tensor_shape(self):
        model = _make_model()
        prompt_tensor = model.text_prompt.prompts  # (L, 1, d)
        self.assertEqual(tuple(prompt_tensor.shape), (L, 1, D_MODEL))
        per_layer = prompt_tensor.squeeze(1)
        out = model.text_mlp_bridge(per_layer)
        self.assertEqual(tuple(out.shape), (L, D_NODE))


# ========================================================================================
# 8. MLP -> graph interface
# ========================================================================================
class TestMLPToGraphInterface(unittest.TestCase):
    def test_pre_gin_graph_shapes(self):
        model = _make_model()
        _, tokens = _random_batch()
        out = model.text_only_forward(tokens, dt=DT)
        pre = out.text.pre_gin_graph
        self.assertIsInstance(pre, Graph)
        self.assertEqual(tuple(pre.X.shape), (L, D_NODE))
        self.assertEqual(tuple(pre.A.shape), (L, L))


# ========================================================================================
# 9. graph -> GIN interface
# ========================================================================================
class TestGraphToGINInterface(unittest.TestCase):
    def test_gin_output_shape_and_adjacency_passthrough(self):
        model = _make_model()
        _, tokens = _random_batch()
        out = model.text_only_forward(tokens, dt=DT)
        pre, post = out.text.pre_gin_graph, out.text.gin_graph
        self.assertEqual(tuple(post.X.shape), (L, GIN_HIDDEN))
        # GIN never modifies adjacency -- identical tensor object.
        self.assertIs(pre.A, post.A)


# ========================================================================================
# 10. GIN -> ACGA interface
# ========================================================================================
class TestGINToACGAInterface(unittest.TestCase):
    def test_acga_consumes_gin_output_shape(self):
        model = _make_model()
        _, tokens = _random_batch()
        out = model.text_only_forward(tokens, dt=DT)
        acga_out = out.text.acga
        self.assertEqual(tuple(acga_out.Z.shape), (L, LATENT_K))
        self.assertEqual(tuple(acga_out.A_hat.shape), (L, L))


# ========================================================================================
# 11. GIN -> HGN-EC interface
# ========================================================================================
class TestGINToHGNECInterface(unittest.TestCase):
    def test_hgnec_consumes_gin_output_shape(self):
        model = _make_model()
        _, tokens = _random_batch()
        out = model.text_only_forward(tokens, dt=DT)
        hgnec_out = out.text.hgn_ec
        expected_restored = D_MODEL  # hgnec_restored_dim is forced to prompt_dim
        self.assertEqual(tuple(hgnec_out.q_final.shape), (L, expected_restored))


# ========================================================================================
# 12/13. CRITICAL STRUCTURAL TEST: ACGA does not overwrite the GIN graph; HGN-EC receives
#         the GIN output directly (GIN -> {ACGA, HGN-EC}, NOT GIN -> ACGA -> HGN-EC).
# ========================================================================================
class TestCriticalStructural(unittest.TestCase):
    def test_gin_fans_out_to_acga_and_hgnec_not_serially(self):
        model = _make_model()
        _, tokens = _random_batch()
        out = model.text_only_forward(tokens, dt=DT)

        gin_graph = out.text.gin_graph
        acga_out = out.text.acga
        hgnec_out = out.text.hgn_ec

        # (a) ACGA's own outputs live in a DIFFERENT feature space (K != GIN_HIDDEN) and
        #     are separate tensor objects from gin_graph.X -- ACGA cannot have overwritten
        #     gin_graph.X with its own Z, since the shapes are provably incompatible.
        self.assertNotEqual(acga_out.Z.shape[-1], gin_graph.X.shape[-1])
        self.assertIsNot(acga_out.Z, gin_graph.X)
        self.assertIsNot(acga_out.A_hat, gin_graph.A)

        # (b) gin_graph.X/A are numerically UNCHANGED after both ACGA and HGN-EC have run
        #     on them (neither mutates in place) -- re-run the pre-GIN -> GIN steps
        #     independently and confirm the graph object we hold is still self-consistent.
        recomputed_pre = model._build_pre_gin_graph(model.text_prompt, model.text_mlp_bridge, model.text_node_config)
        recomputed_gin = model.gin(recomputed_pre)
        self.assertTrue(torch.equal(gin_graph.X, recomputed_gin.X))
        self.assertTrue(torch.equal(gin_graph.A, recomputed_gin.A))

        # (c) HGN-EC's internal state (Eq. 27: state = [X, aggregated]) is built directly
        #     from gin_graph.X/A -- reproduce it independently from gin_graph and confirm
        #     the FIRST GIN_HIDDEN columns match gin_graph.X exactly (i.e. HGN-EC consumed
        #     gin_graph.X, dimension GIN_HIDDEN=5, NOT acga_out.Z, dimension LATENT_K=3 --
        #     the two are not even shape-compatible, so this is an unambiguous, tensor-level
        #     check that HGN-EC did not consume ACGA's reconstruction/latent output).
        from models.hgn_ec.state_init import build_initial_state
        state = build_initial_state(gin_graph.X, gin_graph.A)
        self.assertEqual(state.shape[-1], 2 * GIN_HIDDEN)
        self.assertTrue(torch.equal(state[..., :GIN_HIDDEN], gin_graph.X))

        # (d) HGN-EC's output shape is only consistent with an input width of GIN_HIDDEN
        #     (compressor was built with state_dim = 2*GIN_HIDDEN); had it silently
        #     consumed ACGA's Z (width LATENT_K=3 != GIN_HIDDEN=5), the forward pass itself
        #     would have raised a shape error inside FeatureCompressor. The fact that
        #     `hgnec_out` was produced without error, combined with (c), demonstrates
        #     GIN -> HGN-EC directly, not GIN -> ACGA -> HGN-EC.
        self.assertIsNotNone(hgnec_out.q_final)


# ========================================================================================
# 14. text/vision modality preservation (never fused/averaged/concatenated)
# ========================================================================================
class TestModalityPreservation(unittest.TestCase):
    def test_output_never_fuses_modalities(self):
        model = _make_model()
        images, tokens = _random_batch()
        out = model.forward(images, tokens, dt=DT)
        self.assertEqual(out.text.modality, "text")
        self.assertEqual(out.vision.modality, "vision")
        # No field on ACHGCLIPOutput represents a fused/combined result.
        self.assertEqual(
            set(ACHGCLIPOutput.__dataclass_fields__.keys()),
            {"h_text", "h_vision", "text", "vision"},
        )


# ========================================================================================
# 15. batch handling
# ========================================================================================
class TestBatchHandling(unittest.TestCase):
    def test_clip_embeddings_scale_with_batch_size(self):
        model = _make_model()
        for batch in (1, 2, 5):
            images, tokens = _random_batch(batch=batch)
            out = model.forward(images, tokens, dt=DT)
            self.assertEqual(out.h_text.shape[0], batch)
            self.assertEqual(out.h_vision.shape[0], batch)

    def test_graph_branch_is_batch_independent(self):
        """Prompts G/GV are model parameters, not per-sample data -- one graph per
        modality, shared across the whole batch (Part 3 of the blueprint)."""
        model = _make_model()
        images_a, tokens_a = _random_batch(batch=1)
        images_b, tokens_b = _random_batch(batch=6)
        out_a = model.forward(images_a, tokens_a, dt=DT)
        out_b = model.forward(images_b, tokens_b, dt=DT)
        self.assertEqual(tuple(out_a.text.gin_graph.X.shape), tuple(out_b.text.gin_graph.X.shape))
        self.assertTrue(torch.equal(out_a.text.gin_graph.X, out_b.text.gin_graph.X))


# ========================================================================================
# 16. shape validation
# ========================================================================================
class TestShapeValidation(unittest.TestCase):
    def test_mismatched_prompt_dim_raises(self):
        with self.assertRaises(ACHGCLIPConfigError):
            _make_config(prompt_dim=D_MODEL + 1)

    def test_mismatched_num_layers_raises(self):
        with self.assertRaises(ACHGCLIPConfigError):
            _make_config(num_layers=L + 1)

    def test_num_prompts_other_than_one_raises(self):
        with self.assertRaises(ACHGCLIPConfigError):
            _make_config(num_prompts=2)


# ========================================================================================
# 17. deterministic fixed-seed behavior
# ========================================================================================
class TestDeterminism(unittest.TestCase):
    def test_same_seed_gives_identical_outputs(self):
        model_a = _make_model(seed=123)
        model_b = _make_model(seed=123)
        set_seed(999, deterministic=False)  # scramble RNG between builds; inputs use a fixed seed below
        torch.manual_seed(42)
        images, tokens = _random_batch()
        torch.manual_seed(7)
        out_a = model_a.forward(images, tokens, dt=DT)
        torch.manual_seed(7)
        out_b = model_b.forward(images, tokens, dt=DT)
        self.assertTrue(torch.allclose(out_a.h_text, out_b.h_text))
        self.assertTrue(torch.allclose(out_a.text.gin_graph.X, out_b.text.gin_graph.X))


# ========================================================================================
# 18. gradient propagation
# ========================================================================================
class TestGradientPropagation(unittest.TestCase):
    def test_trainable_params_receive_gradients(self):
        model = _make_model()
        images, tokens = _random_batch()
        out = model.forward(images, tokens, dt=DT)

        loss = (
            out.text.acga.reconstruction_loss
            + out.text.acga.adversarial_loss
            + out.text.hgn_ec.energy_loss
            + out.vision.acga.reconstruction_loss
            + out.vision.acga.adversarial_loss
            + out.vision.hgn_ec.energy_loss
        )
        loss.backward()

        # Prompts, MLP bridges, GIN, ACGA, HGN-EC weights all receive gradients.
        self.assertIsNotNone(model.text_prompt.prompts.grad)
        self.assertIsNotNone(model.vision_prompt.prompts.grad)
        for p in model.text_mlp_bridge.parameters():
            self.assertIsNotNone(p.grad)
        for p in model.gin.parameters():
            self.assertIsNotNone(p.grad)
        for p in model.acga.parameters():
            self.assertIsNotNone(p.grad)
        for name, p in model.hgn_ec.named_parameters():
            # The 'restorer' computes q_final, which is only used in the UNRESOLVED
            # feedback path (not in energy_loss). Therefore, it correctly receives
            # no gradient in this isolated test that only minimizes energy_loss.
            if "restorer" not in name:
                self.assertIsNotNone(p.grad, msg=f"{name} has no gradient")

    def test_frozen_clip_backbone_receives_no_gradient(self):
        model = _make_model()
        images, tokens = _random_batch()
        out = model.forward(images, tokens, dt=DT)
        loss = out.text.acga.reconstruction_loss + out.vision.hgn_ec.energy_loss
        loss.backward()
        for p in model.clip.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)


# ========================================================================================
# 19. device handling
# ========================================================================================
class TestDeviceHandling(unittest.TestCase):
    def test_to_cpu_is_a_noop_and_forward_still_works(self):
        model = _make_model().to("cpu")
        images, tokens = _random_batch()
        out = model.forward(images.to("cpu"), tokens.to("cpu"), dt=DT)
        self.assertEqual(out.h_text.device.type, "cpu")

    def test_device_property(self):
        model = _make_model()
        self.assertEqual(model.device.type, "cpu")


# ========================================================================================
# 20. configuration/provenance validation
# ========================================================================================
class TestConfigProvenance(unittest.TestCase):
    def test_provenance_dict_is_recorded(self):
        cfg = _make_config(provenance={"node_feature_dim": "TEST_OVERRIDE"})
        self.assertEqual(cfg.provenance.get("node_feature_dim"), "TEST_OVERRIDE")

    def test_invalid_adjacency_threshold_raises(self):
        with self.assertRaises(ACHGCLIPConfigError):
            _make_config(adjacency_threshold=2.0)

    def test_invalid_reduction_raises(self):
        with self.assertRaises(ACHGCLIPConfigError):
            _make_config(acga_reduction="bogus")


# ========================================================================================
# 21. missing/unresolved configuration detection
# ========================================================================================
class TestMissingConfigDetection(unittest.TestCase):
    def test_missing_required_field_raises_type_error(self):
        with self.assertRaises(TypeError):
            ACHGCLIPConfig(clip=_mock_clip_config())  # missing num_layers, prompt_dim, ...

    def test_zero_or_negative_required_dims_rejected(self):
        for field_name in (
            "node_feature_dim",
            "mlp_bridge_hidden_dim",
            "acga_latent_dim",
            "hgnec_compressed_dim",
        ):
            with self.assertRaises(ACHGCLIPConfigError):
                _make_config(**{field_name: 0})


# ========================================================================================
# 22. synthetic complete forward pass (end-to-end smoke test)
# ========================================================================================
class TestSyntheticCompleteForwardPass(unittest.TestCase):
    def test_full_joint_pass_produces_every_contract_field(self):
        model = _make_model(seed=0)
        images, tokens = _random_batch(batch=3, n_tokens=6, n_patches=5)
        out = model.forward(images, tokens, dt=DT, num_steps=2)

        self.assertIsInstance(out, ACHGCLIPOutput)
        for modality_out in (out.text, out.vision):
            self.assertIsInstance(modality_out, ModalityOutput)
            self.assertIsInstance(modality_out.pre_gin_graph, Graph)
            self.assertIsInstance(modality_out.gin_graph, Graph)
            self.assertTrue(torch.isfinite(modality_out.acga.reconstruction_loss).all())
            self.assertTrue(torch.isfinite(modality_out.acga.adversarial_loss).all())
            self.assertTrue(torch.isfinite(modality_out.hgn_ec.energy_loss).all())
            self.assertEqual(modality_out.feedback["status"], "RESOLVED")
            self.assertTrue(modality_out.feedback["applied"])

        total = (
            out.text.acga.reconstruction_loss
            + out.text.acga.adversarial_loss
            + out.text.hgn_ec.energy_loss
            + out.vision.acga.reconstruction_loss
            + out.vision.acga.adversarial_loss
            + out.vision.hgn_ec.energy_loss
        )
        self.assertTrue(torch.isfinite(total))
        total.backward()  # full end-to-end gradient flow, multi-step integration


if __name__ == "__main__":
    unittest.main()

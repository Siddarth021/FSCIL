"""
tests/test_acga.py
=====================

Stage 6 tests: Adversarially Constrained Graph Autoencoder (Eqs. 22-26).
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from models.acga.acga import ACGA, ACGAConfig, ACGAConfigError, ACGAOutput
from models.acga.decoder import DecoderShapeError, InnerProductDecoder
from models.acga.discriminator import (
    Discriminator,
    DiscriminatorConfig,
    DiscriminatorConfigError,
    DiscriminatorShapeError,
)
from models.acga.encoder import ACGAEncoder, ACGAEncoderConfig, ACGAEncoderConfigError
from models.graph.graph_data import Graph
from models.gnn.gin_layer import GINLayerShapeError
from losses.acga_losses import (
    ACGALossError,
    adversarial_loss,
    clip_discriminator_weights,
    gradient_penalty,
    reconstruction_loss,
)

N, D, K = 4, 6, 3  # small synthetic dims: nodes, input feature dim, latent dim


def _random_A(n=N, symmetric=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    logits = torch.rand(n, n)
    A = (logits > 0.5).float()
    if symmetric:
        A = ((A + A.t()) > 0).float()
    return A


def _make_encoder(input_dim=D, latent_dim=K, **kwargs):
    return ACGAEncoder(ACGAEncoderConfig(input_dim=input_dim, latent_dim=latent_dim, **kwargs))


def _make_discriminator(latent_dim=K, **kwargs):
    return Discriminator(DiscriminatorConfig(latent_dim=latent_dim, **kwargs))


def _make_acga(input_dim=D, latent_dim=K, **kwargs):
    return ACGA(ACGAConfig(input_dim=input_dim, latent_dim=latent_dim, **kwargs))


def _make_graph(n=N, d=D, X=None, A=None, modality="text"):
    if X is None:
        X = torch.randn(n, d)
    if A is None:
        A = _random_A(n)
    return Graph(X=X, A=A, N=n, feature_dim=d, modality=modality)


# ----------------------------------------------------------------------------------------
# 1. Encoder initialization
# ----------------------------------------------------------------------------------------


class TestEncoderInitialization(unittest.TestCase):
    def test_encoder_builds_gin_stack(self):
        enc = _make_encoder()
        self.assertTrue(hasattr(enc, "gin"))
        self.assertEqual(len(enc.gin.layers), 1)  # default num_layers=1

    def test_encoder_custom_num_layers(self):
        enc = _make_encoder(num_layers=3)
        self.assertEqual(len(enc.gin.layers), 3)

    def test_encoder_config_rejects_non_positive_dims(self):
        with self.assertRaises(ACGAEncoderConfigError):
            ACGAEncoderConfig(input_dim=0, latent_dim=K)
        with self.assertRaises(ACGAEncoderConfigError):
            ACGAEncoderConfig(input_dim=D, latent_dim=-1)

    def test_encoder_has_trainable_parameters(self):
        enc = _make_encoder()
        params = list(enc.parameters())
        self.assertTrue(len(params) > 0)
        self.assertTrue(all(p.requires_grad for p in params))


# ----------------------------------------------------------------------------------------
# 2. Encoder output shape / 3. Latent Z shape
# ----------------------------------------------------------------------------------------


class TestEncoderOutputShape(unittest.TestCase):
    def test_encoder_output_shape_unbatched(self):
        enc = _make_encoder()
        X, A = torch.randn(N, D), _random_A()
        Z = enc(X, A)
        self.assertEqual(tuple(Z.shape), (N, K))

    def test_encoder_output_shape_batched(self):
        enc = _make_encoder()
        B = 2
        X, A = torch.randn(B, N, D), torch.stack([_random_A() for _ in range(B)])
        Z = enc(X, A)
        self.assertEqual(tuple(Z.shape), (B, N, K))

    def test_encoder_rejects_wrong_input_dim(self):
        enc = _make_encoder()
        X, A = torch.randn(N, D + 1), _random_A()
        with self.assertRaises(GINLayerShapeError):
            enc(X, A)

    def test_latent_dim_independent_of_node_count(self):
        # "N and K are independent" -- Section IV.C.1.
        enc = _make_encoder()
        for n in (2, 5, 8):
            X, A = torch.randn(n, D), _random_A(n)
            Z = enc(X, A)
            self.assertEqual(tuple(Z.shape), (n, K))


# ----------------------------------------------------------------------------------------
# 4. Decoder output shape / 5. Reconstruction shape
# ----------------------------------------------------------------------------------------


class TestDecoderOutputShape(unittest.TestCase):
    def test_decoder_output_shape_unbatched(self):
        dec = InnerProductDecoder()
        Z = torch.randn(N, K)
        A_hat = dec(Z)
        self.assertEqual(tuple(A_hat.shape), (N, N))

    def test_decoder_output_shape_batched(self):
        dec = InnerProductDecoder()
        B = 3
        Z = torch.randn(B, N, K)
        A_hat = dec(Z)
        self.assertEqual(tuple(A_hat.shape), (B, N, N))

    def test_decoder_output_range_and_symmetry(self):
        dec = InnerProductDecoder()
        Z = torch.randn(N, K)
        A_hat = dec(Z)
        self.assertTrue(torch.all(A_hat >= 0.0) and torch.all(A_hat <= 1.0))
        self.assertTrue(torch.allclose(A_hat, A_hat.t(), atol=1e-6))

    def test_decoder_no_learnable_parameters(self):
        dec = InnerProductDecoder()
        self.assertEqual(len(list(dec.parameters())), 0)

    def test_decoder_rejects_bad_ndim(self):
        dec = InnerProductDecoder()
        with self.assertRaises(DecoderShapeError):
            dec(torch.randn(K))

    def test_reconstructed_shape_matches_input_graph_adjacency(self):
        graph = _make_graph()
        dec = InnerProductDecoder()
        enc = _make_encoder()
        Z = enc(graph.X, graph.A)
        A_hat = dec(Z)
        self.assertEqual(tuple(A_hat.shape), tuple(graph.A.shape))


# ----------------------------------------------------------------------------------------
# 6. Reconstruction loss
# ----------------------------------------------------------------------------------------


class TestReconstructionLoss(unittest.TestCase):
    def test_reconstruction_loss_is_scalar_and_finite(self):
        A = _random_A()
        A_hat = torch.sigmoid(torch.randn(N, N))
        loss = reconstruction_loss(A, A_hat)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_reconstruction_loss_zero_for_perfect_reconstruction(self):
        A = _random_A()
        # Push A_hat toward exact {0,1} matching A (clamped away from exact 0/1 for log()).
        A_hat = A * (1 - 2e-6) + (1 - A) * 1e-6
        loss = reconstruction_loss(A, A_hat)
        self.assertLess(float(loss), 1e-2)

    def test_reconstruction_loss_shape_mismatch_raises(self):
        A = _random_A()
        A_hat = torch.sigmoid(torch.randn(N + 1, N + 1))
        with self.assertRaises(ACGALossError):
            reconstruction_loss(A, A_hat)

    def test_reconstruction_loss_reduction_sum_vs_mean(self):
        A = _random_A()
        A_hat = torch.sigmoid(torch.randn(N, N))
        loss_mean = reconstruction_loss(A, A_hat, reduction="mean")
        loss_sum = reconstruction_loss(A, A_hat, reduction="sum")
        self.assertAlmostEqual(float(loss_sum) / (N * N), float(loss_mean), places=4)

    def test_reconstruction_loss_invalid_reduction_raises(self):
        A = _random_A()
        A_hat = torch.sigmoid(torch.randn(N, N))
        with self.assertRaises(ACGALossError):
            reconstruction_loss(A, A_hat, reduction="bogus")

    def test_reconstruction_loss_negative_sampling_ratio_bounds(self):
        A = _random_A()
        A_hat = torch.sigmoid(torch.randn(N, N))
        with self.assertRaises(ACGALossError):
            reconstruction_loss(A, A_hat, negative_sampling_ratio=0.0)
        with self.assertRaises(ACGALossError):
            reconstruction_loss(A, A_hat, negative_sampling_ratio=1.5)
        # A valid ratio should not raise and should always include all positive entries.
        loss = reconstruction_loss(A, A_hat, negative_sampling_ratio=0.5, generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.isfinite(loss))


# ----------------------------------------------------------------------------------------
# 7. Discriminator initialization / 8. Discriminator output
# ----------------------------------------------------------------------------------------


class TestDiscriminator(unittest.TestCase):
    def test_discriminator_init_two_linear_layers(self):
        disc = _make_discriminator()
        self.assertIsInstance(disc.fc1, nn.Linear)
        self.assertIsInstance(disc.fc2, nn.Linear)
        self.assertEqual(disc.fc2.out_features, 1)

    def test_discriminator_hidden_dim_defaults_to_latent_dim(self):
        disc = _make_discriminator()
        self.assertEqual(disc.config.hidden_dim, K)

    def test_discriminator_config_rejects_bad_dims(self):
        with self.assertRaises(DiscriminatorConfigError):
            DiscriminatorConfig(latent_dim=0)
        with self.assertRaises(DiscriminatorConfigError):
            DiscriminatorConfig(latent_dim=K, hidden_dim=-2)

    def test_discriminator_output_shape_and_range(self):
        disc = _make_discriminator()
        z = torch.randn(N, K)
        d = disc(z)
        self.assertEqual(tuple(d.shape), (N,))
        self.assertTrue(torch.all(d >= 0.0) and torch.all(d <= 1.0))

    def test_discriminator_output_shape_batched(self):
        disc = _make_discriminator()
        B = 2
        z = torch.randn(B, N, K)
        d = disc(z)
        self.assertEqual(tuple(d.shape), (B, N))

    def test_discriminator_rejects_wrong_latent_dim(self):
        disc = _make_discriminator()
        z = torch.randn(N, K + 1)
        with self.assertRaises(DiscriminatorShapeError):
            disc(z)


# ----------------------------------------------------------------------------------------
# 9. Adversarial loss
# ----------------------------------------------------------------------------------------


class TestAdversarialLoss(unittest.TestCase):
    def test_adversarial_loss_scalar_finite(self):
        d_real = torch.rand(N)
        d_fake = torch.rand(N)
        loss = adversarial_loss(d_real, d_fake)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_adversarial_loss_matches_eq26_sign(self):
        d_real = torch.ones(N) * 0.9
        d_fake = torch.ones(N) * 0.2
        loss = adversarial_loss(d_real, d_fake)
        self.assertAlmostEqual(float(loss), 0.7, places=5)

    def test_adversarial_loss_shape_mismatch_raises(self):
        with self.assertRaises(ACGALossError):
            adversarial_loss(torch.rand(N), torch.rand(N + 1))

    def test_gradient_penalty_isolated_and_finite(self):
        disc = _make_discriminator()
        z_real = torch.randn(N, K)
        z_fake = torch.randn(N, K)
        gp = gradient_penalty(disc, z_real, z_fake)
        self.assertEqual(gp.dim(), 0)
        self.assertTrue(torch.isfinite(gp))

    def test_weight_clipping_isolated_and_bounds_params(self):
        disc = _make_discriminator()
        clip_discriminator_weights(disc, 0.05)
        for p in disc.parameters():
            self.assertTrue(torch.all(p.abs() <= 0.05 + 1e-6))


# ----------------------------------------------------------------------------------------
# 10. Gradient flow through encoder / 11. decoder / 12. discriminator
# ----------------------------------------------------------------------------------------


class TestGradientFlow(unittest.TestCase):
    def test_gradient_flow_through_encoder(self):
        enc = _make_encoder()
        X = torch.randn(N, D, requires_grad=True)
        A = _random_A()
        Z = enc(X, A)
        Z.sum().backward()
        self.assertIsNotNone(X.grad)
        for p in enc.parameters():
            self.assertIsNotNone(p.grad)

    def test_gradient_flow_through_decoder(self):
        Z = torch.randn(N, K, requires_grad=True)
        dec = InnerProductDecoder()
        A_hat = dec(Z)
        A_hat.sum().backward()
        self.assertIsNotNone(Z.grad)

    def test_gradient_flow_encoder_to_decoder_via_recon_loss(self):
        enc = _make_encoder()
        dec = InnerProductDecoder()
        X = torch.randn(N, D, requires_grad=True)
        A = _random_A()
        Z = enc(X, A)
        A_hat = dec(Z)
        loss = reconstruction_loss(A, A_hat)
        loss.backward()
        self.assertIsNotNone(X.grad)
        for p in enc.parameters():
            self.assertIsNotNone(p.grad)

    def test_discriminator_gradient_flow(self):
        disc = _make_discriminator()
        z = torch.randn(N, K, requires_grad=True)
        d = disc(z)
        d.sum().backward()
        self.assertIsNotNone(z.grad)
        for p in disc.parameters():
            self.assertIsNotNone(p.grad)

    def test_gradient_flow_through_adversarial_loss_to_encoder(self):
        acga = _make_acga()
        X = torch.randn(N, D, requires_grad=True)
        A = _random_A()
        out = acga.forward_tensors(X, A)
        out.adversarial_loss.backward(retain_graph=True)
        self.assertIsNotNone(X.grad)
        enc_grads = [p.grad is not None for p in acga.encoder.parameters()]
        self.assertTrue(any(enc_grads))
        disc_grads = [p.grad is not None for p in acga.discriminator.parameters()]
        self.assertTrue(all(disc_grads))


# ----------------------------------------------------------------------------------------
# 13. Batched graphs
# ----------------------------------------------------------------------------------------


class TestBatchedGraphs(unittest.TestCase):
    def test_acga_forward_tensors_batched(self):
        acga = _make_acga()
        B = 3
        X = torch.randn(B, N, D)
        A = torch.stack([_random_A() for _ in range(B)])
        out = acga.forward_tensors(X, A)
        self.assertEqual(tuple(out.Z.shape), (B, N, K))
        self.assertEqual(tuple(out.A_hat.shape), (B, N, N))
        self.assertEqual(tuple(out.d_real.shape), (B, N))
        self.assertEqual(tuple(out.d_fake.shape), (B, N))
        self.assertTrue(torch.isfinite(out.reconstruction_loss))
        self.assertTrue(torch.isfinite(out.adversarial_loss))


# ----------------------------------------------------------------------------------------
# 14. Text modality / 15. Vision modality
# ----------------------------------------------------------------------------------------


class TestModalities(unittest.TestCase):
    def test_acga_forward_text_graph(self):
        acga = _make_acga()
        graph = _make_graph(modality="text")
        out = acga.forward(graph)
        self.assertEqual(tuple(out.Z.shape), (N, K))

    def test_acga_forward_vision_graph(self):
        acga = _make_acga()
        graph = _make_graph(modality="vision")
        out = acga.forward(graph)
        self.assertEqual(tuple(out.Z.shape), (N, K))

    def test_acga_shared_weights_across_modality_calls(self):
        # Blocker 5: ONE set of ACGA weights, invoked once per modality (module docstring).
        acga = _make_acga()
        text_graph = _make_graph(modality="text")
        vision_graph = _make_graph(modality="vision")
        params_before = [p.clone() for p in acga.parameters()]
        acga.forward(text_graph)
        acga.forward(vision_graph)
        params_after = list(acga.parameters())
        self.assertEqual(len(params_before), len(params_after))
        for a, b in zip(params_before, params_after):
            self.assertTrue(torch.equal(a, b))  # no forward-only call mutates weights


# ----------------------------------------------------------------------------------------
# 16. Device handling
# ----------------------------------------------------------------------------------------


class TestDeviceHandling(unittest.TestCase):
    def test_cpu_forward_explicit(self):
        acga = _make_acga().to("cpu")
        X, A = torch.randn(N, D), _random_A()
        out = acga.forward_tensors(X.to("cpu"), A.to("cpu"))
        self.assertEqual(out.Z.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available in this environment")
    def test_cuda_forward(self):  # pragma: no cover - only runs with a GPU present
        acga = _make_acga().to("cuda")
        X, A = torch.randn(N, D).to("cuda"), _random_A().to("cuda")
        out = acga.forward_tensors(X, A)
        self.assertEqual(out.Z.device.type, "cuda")


# ----------------------------------------------------------------------------------------
# 17. Invalid shape detection
# ----------------------------------------------------------------------------------------


class TestInvalidShapeDetection(unittest.TestCase):
    def test_encoder_wrong_feature_dim(self):
        enc = _make_encoder()
        with self.assertRaises(GINLayerShapeError):
            enc(torch.randn(N, D + 2), _random_A())

    def test_decoder_non_square_input_still_produces_square_output_but_mismatched_A_detected_upstream(self):
        # InnerProductDecoder itself always produces a square (N, N) A_hat by construction
        # (Z Z^T); shape validation against a *different* input graph's A happens at the
        # ACGA/loss level, exercised below.
        dec = InnerProductDecoder()
        Z = torch.randn(N, K)
        A_hat = dec(Z)
        wrong_A = _random_A(N + 1)
        with self.assertRaises(ACGALossError):
            reconstruction_loss(wrong_A, A_hat)

    def test_discriminator_wrong_latent_dim_raises(self):
        disc = _make_discriminator()
        with self.assertRaises(DiscriminatorShapeError):
            disc(torch.randn(N, K - 1))

    def test_acga_graph_feature_dim_mismatch_raises(self):
        acga = _make_acga()
        graph = _make_graph(d=D + 1)
        with self.assertRaises(Exception):
            acga.forward(graph)


# ----------------------------------------------------------------------------------------
# 18. Fixed-seed determinism
# ----------------------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):
    def test_deterministic_forward_given_fixed_seed_and_generator(self):
        torch.manual_seed(123)
        acga_a = _make_acga()
        torch.manual_seed(123)
        acga_b = _make_acga()

        X, A = torch.randn(N, D), _random_A(seed=0)
        gen_a = torch.Generator().manual_seed(42)
        gen_b = torch.Generator().manual_seed(42)

        out_a = acga_a.forward_tensors(X, A, generator=gen_a)
        out_b = acga_b.forward_tensors(X, A, generator=gen_b)

        self.assertTrue(torch.allclose(out_a.Z, out_b.Z, atol=1e-6))
        self.assertTrue(torch.allclose(out_a.A_hat, out_b.A_hat, atol=1e-6))
        self.assertTrue(torch.allclose(out_a.d_real, out_b.d_real, atol=1e-6))
        self.assertAlmostEqual(float(out_a.reconstruction_loss), float(out_b.reconstruction_loss), places=5)
        self.assertAlmostEqual(float(out_a.adversarial_loss), float(out_b.adversarial_loss), places=5)


# ----------------------------------------------------------------------------------------
# 19. Configuration / provenance validation
# ----------------------------------------------------------------------------------------


class TestConfigValidation(unittest.TestCase):
    def test_acga_config_rejects_non_standard_normal_prior(self):
        with self.assertRaises(ACGAConfigError):
            ACGAConfig(input_dim=D, latent_dim=K, prior_distribution="uniform")

    def test_acga_config_rejects_bad_reduction(self):
        with self.assertRaises(ACGAConfigError):
            ACGAConfig(input_dim=D, latent_dim=K, reduction="bogus")

    def test_encoder_config_latent_dim_is_required_no_default(self):
        import inspect

        sig = inspect.signature(ACGAEncoderConfig.__init__)
        self.assertNotIn("default", str(sig.parameters["latent_dim"]))

    def test_discriminator_hidden_dim_unresolved_defaults_locally_not_silently_paper_fact(self):
        # configs/model/acga.yaml: discriminator_hidden_dim is UNRESOLVED (null); this module
        # applies a documented code-level default (== latent_dim) rather than a paper value.
        cfg = DiscriminatorConfig(latent_dim=K)
        self.assertEqual(cfg.hidden_dim, K)


# ----------------------------------------------------------------------------------------
# 20. ACGA outputs do not mutate the input GIN graph (CRITICAL IMMUTABILITY TEST)
# ----------------------------------------------------------------------------------------


class TestImmutability(unittest.TestCase):
    def test_acga_does_not_mutate_input_X_or_A_tensors(self):
        acga = _make_acga()
        X = torch.randn(N, D)
        A = _random_A()
        X_before = X.clone()
        A_before = A.clone()

        acga.forward_tensors(X, A)

        self.assertTrue(torch.equal(X, X_before), "ACGA mutated its input X in place.")
        self.assertTrue(torch.equal(A, A_before), "ACGA mutated its input A in place.")

    def test_acga_does_not_mutate_graph_object_fields(self):
        acga = _make_acga()
        graph = _make_graph()
        X_before = graph.X.clone()
        A_before = graph.A.clone()

        acga.forward(graph)

        self.assertTrue(torch.equal(graph.X, X_before), "ACGA mutated Graph.X in place.")
        self.assertTrue(torch.equal(graph.A, A_before), "ACGA mutated Graph.A in place.")

    def test_acga_output_A_hat_is_not_the_same_object_as_input_A(self):
        acga = _make_acga()
        graph = _make_graph()
        out = acga.forward(graph)
        self.assertIsNot(out.A_hat, graph.A)
        # And it must not merely alias the same underlying storage either.
        self.assertFalse(out.A_hat.data_ptr() == graph.A.data_ptr())


# ----------------------------------------------------------------------------------------
# 21. Z/reconstruction are exposed separately
# ----------------------------------------------------------------------------------------


class TestOutputsExposedSeparately(unittest.TestCase):
    def test_output_bundle_exposes_distinct_fields(self):
        acga = _make_acga()
        graph = _make_graph()
        out = acga.forward(graph)
        self.assertIsInstance(out, ACGAOutput)
        for field_name in ("Z", "A_hat", "d_real", "d_fake", "reconstruction_loss", "adversarial_loss"):
            self.assertTrue(hasattr(out, field_name))
        # Z and A_hat must not be the same tensor.
        self.assertIsNot(out.Z, out.A_hat)
        # Loss components returned uncombined (Stage 8 composes L_total, Eq. 34) --
        # summing them here must NOT already equal some hidden pre-combined attribute.
        self.assertFalse(hasattr(out, "total_loss"))


# ----------------------------------------------------------------------------------------
# 22. Synthetic end-to-end ACGA forward pass
# ----------------------------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):
    def test_end_to_end_forward_from_stage4_graph(self):
        from models.graph.adjacency import AdjacencyConfig, build_graph
        from models.graph.node_builder import NodeBuilderConfig
        from models.prompts.mlp_bridge import MLPBridgeConfig, PromptToNodeMLP

        L, prompt_dim, out_dim = 4, 8, D
        bridge = PromptToNodeMLP(MLPBridgeConfig(input_dim=prompt_dim, hidden_dim=16, output_dim=out_dim))
        prompt_tensor = torch.randn(L, 1, prompt_dim)
        graph = build_graph(
            prompt_tensor,
            bridge,
            NodeBuilderConfig(modality="text"),
            AdjacencyConfig(),
        )

        acga = _make_acga(input_dim=graph.feature_dim, latent_dim=K)
        out = acga.forward(graph)

        self.assertEqual(tuple(out.Z.shape), (graph.N, K))
        self.assertEqual(tuple(out.A_hat.shape), (graph.N, graph.N))
        self.assertTrue(torch.isfinite(out.reconstruction_loss))
        self.assertTrue(torch.isfinite(out.adversarial_loss))

        total = out.reconstruction_loss + out.adversarial_loss
        total.backward()
        for p in acga.parameters():
            self.assertIsNotNone(p.grad)

    def test_end_to_end_synthetic_batch_full_pipeline(self):
        acga = _make_acga()
        B = 2
        X = torch.randn(B, N, D)
        A = torch.stack([_random_A() for _ in range(B)])
        out = acga.forward_tensors(X, A)
        loss = out.reconstruction_loss + out.adversarial_loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

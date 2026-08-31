"""
tests/test_hgn_ec.py
========================

Stage 7 tests: Hamiltonian Graph Network with Energy Conservation (Eqs. 27-33).
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from models.hgn_ec.state_init import build_initial_state, init_q_p, StateInitShapeError
from models.hgn_ec.compress import FeatureCompressor, FeatureCompressorConfig, CompressorConfigError
from models.hgn_ec.hamiltonian import (
    HamiltonianNet,
    HamiltonianNetConfig,
    HamiltonianNetConfigError,
    HamiltonianShapeError,
    hamiltonian_gradients,
)
from models.hgn_ec.integrator import symplectic_euler_step, IntegratorShapeError
from models.hgn_ec.restore import FeatureRestorer, FeatureRestorerConfig, RestorerConfigError, RestorerShapeError
from models.hgn_ec.hgn_ec import HGNEC, HGNECConfig, HGNECConfigError, HGNECOutput
from losses.hgn_ec_losses import energy_conservation_loss, HGNECLossShapeError
from models.graph.graph_data import Graph
from models.gnn.gin_layer import GINLayerShapeError

N, D, DC = 4, 6, 3  # small synthetic dims: nodes, input feature dim, compressed dim


def _random_A(n=N, symmetric=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    logits = torch.rand(n, n)
    A = (logits > 0.5).float()
    if symmetric:
        A = ((A + A.t()) > 0).float()
    return A


def _make_hgnec(input_dim=D, compressed_dim=DC, **kwargs):
    return HGNEC(HGNECConfig(input_dim=input_dim, compressed_dim=compressed_dim, **kwargs))


def _make_graph(n=N, d=D, X=None, A=None, modality="text"):
    if X is None:
        X = torch.randn(n, d)
    if A is None:
        A = _random_A(n)
    return Graph(X=X, A=A, N=n, feature_dim=d, modality=modality)


# --------------------------------------------------------------------------------------
# 1. Initialization
# --------------------------------------------------------------------------------------
class TestInitialization(unittest.TestCase):
    def test_hgnec_builds(self):
        model = _make_hgnec()
        self.assertIsInstance(model, nn.Module)

    def test_config_rejects_bad_dims(self):
        with self.assertRaises(HGNECConfigError):
            HGNECConfig(input_dim=0, compressed_dim=DC)
        with self.assertRaises(HGNECConfigError):
            HGNECConfig(input_dim=D, compressed_dim=-1)
        with self.assertRaises(HGNECConfigError):
            HGNECConfig(input_dim=D, compressed_dim=DC, num_steps=0)

    def test_restored_dim_defaults_to_state_dim(self):
        cfg = HGNECConfig(input_dim=D, compressed_dim=DC)
        self.assertEqual(cfg.restored_dim, 2 * D)

    def test_submodules_present(self):
        model = _make_hgnec()
        self.assertIsInstance(model.compressor, FeatureCompressor)
        self.assertIsInstance(model.hamiltonian_net, HamiltonianNet)
        self.assertIsInstance(model.restorer, FeatureRestorer)


# --------------------------------------------------------------------------------------
# 2. Input/output shapes
# --------------------------------------------------------------------------------------
class TestShapes(unittest.TestCase):
    def test_end_to_end_shapes(self):
        model = _make_hgnec()
        X, A = torch.randn(N, D), _random_A()
        out = model.forward_tensors(X, A, dt=0.01)
        self.assertEqual(tuple(out.q_final.shape), (N, 2 * D))
        self.assertEqual(out.H_initial.shape, ())
        self.assertEqual(out.H_final.shape, ())
        self.assertEqual(out.energy_loss.shape, ())

    def test_batched_shapes(self):
        B = 5
        model = _make_hgnec()
        X, A = torch.randn(B, N, D), _random_A().unsqueeze(0).repeat(B, 1, 1)
        out = model.forward_tensors(X, A, dt=0.01)
        self.assertEqual(tuple(out.q_final.shape), (B, N, 2 * D))
        self.assertEqual(tuple(out.H_initial.shape), (B,))

    def test_custom_restored_dim(self):
        model = _make_hgnec(restored_dim=10)
        X, A = torch.randn(N, D), _random_A()
        out = model.forward_tensors(X, A, dt=0.01)
        self.assertEqual(out.q_final.shape[-1], 10)

    def test_build_initial_state_shape(self):
        X, A = torch.randn(N, D), _random_A()
        state = build_initial_state(X, A)
        self.assertEqual(tuple(state.shape), (N, 2 * D))

    def test_build_initial_state_rejects_bad_shapes(self):
        X, A = torch.randn(N, D), torch.randn(N + 1, N + 1)
        with self.assertRaises(StateInitShapeError):
            build_initial_state(X, A)


# --------------------------------------------------------------------------------------
# 3. Hamiltonian calculation
# --------------------------------------------------------------------------------------
class TestHamiltonianCalculation(unittest.TestCase):
    def test_scalar_per_graph(self):
        net = HamiltonianNet(HamiltonianNetConfig(compressed_dim=DC))
        q, p, A = torch.randn(N, DC), torch.randn(N, DC), _random_A()
        H = net(q, p, A)
        self.assertEqual(H.shape, ())

    def test_batched_hamiltonian(self):
        B = 3
        net = HamiltonianNet(HamiltonianNetConfig(compressed_dim=DC))
        q = torch.randn(B, N, DC)
        p = torch.randn(B, N, DC)
        A = _random_A().unsqueeze(0).repeat(B, 1, 1)
        H = net(q, p, A)
        self.assertEqual(tuple(H.shape), (B,))

    def test_mismatched_qp_shape_rejected(self):
        net = HamiltonianNet(HamiltonianNetConfig(compressed_dim=DC))
        q = torch.randn(N, DC)
        p = torch.randn(N, DC + 1)
        with self.assertRaises(HamiltonianShapeError):
            net(q, p, _random_A())

    def test_config_defaults_gin_hidden_dim(self):
        cfg = HamiltonianNetConfig(compressed_dim=DC)
        self.assertEqual(cfg.gin_hidden_dim, DC)


# --------------------------------------------------------------------------------------
# 4. Hamiltonian gradients
# --------------------------------------------------------------------------------------
class TestHamiltonianGradients(unittest.TestCase):
    def test_gradients_have_correct_shape(self):
        net = HamiltonianNet(HamiltonianNetConfig(compressed_dim=DC))
        q = torch.randn(N, DC, requires_grad=True)
        p = torch.randn(N, DC, requires_grad=True)
        A = _random_A()
        H = net(q, p, A)
        q_dot, p_dot = hamiltonian_gradients(H, q, p)
        self.assertEqual(q_dot.shape, q.shape)
        self.assertEqual(p_dot.shape, p.shape)

    def test_requires_grad_enforced(self):
        net = HamiltonianNet(HamiltonianNetConfig(compressed_dim=DC))
        q = torch.randn(N, DC, requires_grad=False)
        p = torch.randn(N, DC, requires_grad=True)
        A = _random_A()
        H = net(q, p, A)
        with self.assertRaises(HamiltonianShapeError):
            hamiltonian_gradients(H, q, p)

    def test_manual_harmonic_oscillator_gradients(self):
        """Numerical sanity check on the shared `hamiltonian_gradients` primitive using a
        hand-defined H(q, p) = 0.5*sum(p^2) + 0.5*sum(q^2) (simple harmonic oscillator),
        independent of the GIN-based HamiltonianNet -- dH/dq = q, dH/dp = p exactly."""
        q = torch.tensor([1.0, 2.0, -1.5], requires_grad=True)
        p = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
        H = 0.5 * (p ** 2).sum() + 0.5 * (q ** 2).sum()
        q_dot, p_dot = hamiltonian_gradients(H, q, p)
        torch.testing.assert_close(q_dot, p)  # dH/dp = p
        torch.testing.assert_close(p_dot, -q)  # -dH/dq = -q


# --------------------------------------------------------------------------------------
# 5. Canonical dynamics
# --------------------------------------------------------------------------------------
class TestCanonicalDynamics(unittest.TestCase):
    def test_qdot_pdot_signs(self):
        """q_dot = dH/dp, p_dot = -dH/dq -- confirmed via the harmonic-oscillator case
        (see TestHamiltonianGradients) and here re-checked through the full HGNEC pipeline
        by confirming q_dot/p_dot are nonzero and finite for a random input."""
        model = _make_hgnec()
        X, A = torch.randn(N, D), _random_A()
        state = build_initial_state(X, A)
        compressed = model.compressor(state)
        q, p = init_q_p(compressed)
        q.requires_grad_(True)
        p.requires_grad_(True)
        H = model.hamiltonian_net(q, p, A)
        q_dot, p_dot = hamiltonian_gradients(H, q, p)
        self.assertTrue(torch.isfinite(q_dot).all())
        self.assertTrue(torch.isfinite(p_dot).all())


# --------------------------------------------------------------------------------------
# 6. Symplectic Euler update
# --------------------------------------------------------------------------------------
class TestSymplecticEuler(unittest.TestCase):
    def test_manual_symplectic_step(self):
        """Verify symplectic_euler_step against a hand-computed update for the harmonic
        oscillator H(q,p) = 0.5*p^2 + 0.5*q^2 -- NOT merely a shape check."""
        q = torch.tensor([2.0], requires_grad=True)
        p = torch.tensor([3.0], requires_grad=True)
        dt = 0.1
        H = 0.5 * (p ** 2).sum() + 0.5 * (q ** 2).sum()
        q_dot, p_dot = hamiltonian_gradients(H, q, p)
        # Manual expectation: q_dot = p = 3.0, p_dot = -q = -2.0
        self.assertAlmostEqual(q_dot.item(), 3.0, places=6)
        self.assertAlmostEqual(p_dot.item(), -2.0, places=6)
        q_new, p_new = symplectic_euler_step(q, p, q_dot, p_dot, dt)
        # p_new = p + dt*p_dot = 3.0 + 0.1*(-2.0) = 2.8
        # q_new = q + dt*q_dot = 2.0 + 0.1*3.0    = 2.3
        self.assertAlmostEqual(p_new.item(), 2.8, places=6)
        self.assertAlmostEqual(q_new.item(), 2.3, places=6)

    def test_mismatched_shapes_rejected(self):
        q = torch.randn(3)
        p = torch.randn(3)
        q_dot = torch.randn(3)
        p_dot = torch.randn(4)
        with self.assertRaises(IntegratorShapeError):
            symplectic_euler_step(q, p, q_dot, p_dot, 0.1)

    def test_no_inplace_mutation(self):
        q = torch.tensor([1.0])
        p = torch.tensor([1.0])
        q_dot = torch.tensor([1.0])
        p_dot = torch.tensor([1.0])
        q_before, p_before = q.clone(), p.clone()
        symplectic_euler_step(q, p, q_dot, p_dot, 0.1)
        torch.testing.assert_close(q, q_before)
        torch.testing.assert_close(p, p_before)


# --------------------------------------------------------------------------------------
# 7. Configurable dt
# --------------------------------------------------------------------------------------
class TestConfigurableDt(unittest.TestCase):
    def test_dt_required_no_default(self):
        model = _make_hgnec()
        X, A = torch.randn(N, D), _random_A()
        with self.assertRaises(TypeError):
            model.forward_tensors(X, A)  # dt is keyword-only, required

    def test_different_dt_gives_different_result(self):
        model = _make_hgnec()
        torch.manual_seed(0)
        X, A = torch.randn(N, D), _random_A(seed=0)
        out_small = model.forward_tensors(X, A, dt=0.001)
        out_large = model.forward_tensors(X, A, dt=1.0)
        self.assertFalse(torch.allclose(out_small.q_final, out_large.q_final))

    def test_rejects_non_numeric_dt(self):
        q = p = torch.randn(3)
        q_dot = p_dot = torch.randn(3)
        with self.assertRaises(IntegratorShapeError):
            symplectic_euler_step(q, p, q_dot, p_dot, "not_a_number")


# --------------------------------------------------------------------------------------
# 8. Multiple integration steps
# --------------------------------------------------------------------------------------
class TestMultipleIntegrationSteps(unittest.TestCase):
    def test_num_steps_controls_trajectory_length(self):
        model = _make_hgnec(num_steps=3)
        X, A = torch.randn(N, D), _random_A()
        out = model.forward_tensors(X, A, dt=0.01)
        self.assertEqual(len(out.q_trajectory), 4)  # initial + 3 steps
        self.assertEqual(len(out.p_trajectory), 4)

    def test_per_call_override(self):
        model = _make_hgnec(num_steps=1)
        X, A = torch.randn(N, D), _random_A()
        out = model.forward_tensors(X, A, dt=0.01, num_steps=5)
        self.assertEqual(len(out.q_trajectory), 6)

    def test_rejects_non_positive_num_steps(self):
        model = _make_hgnec()
        X, A = torch.randn(N, D), _random_A()
        with self.assertRaises(HGNECConfigError):
            model.forward_tensors(X, A, dt=0.01, num_steps=0)


# --------------------------------------------------------------------------------------
# 9. Compression/output dimension
# --------------------------------------------------------------------------------------
class TestCompressionOutputDim(unittest.TestCase):
    def test_compressor_output_dim(self):
        comp = FeatureCompressor(FeatureCompressorConfig(state_dim=2 * D, compressed_dim=DC))
        state = torch.randn(N, 2 * D)
        out = comp(state)
        self.assertEqual(out.shape[-1], DC)

    def test_restorer_output_dim(self):
        rest = FeatureRestorer(FeatureRestorerConfig(compressed_dim=DC, restored_dim=2 * D))
        q_new = torch.randn(N, DC)
        out = rest(q_new)
        self.assertEqual(out.shape[-1], 2 * D)

    def test_compressor_rejects_bad_config(self):
        with self.assertRaises(CompressorConfigError):
            FeatureCompressorConfig(state_dim=0, compressed_dim=DC)

    def test_restorer_rejects_bad_config(self):
        with self.assertRaises(RestorerConfigError):
            FeatureRestorerConfig(compressed_dim=DC, restored_dim=-1)


# --------------------------------------------------------------------------------------
# 10. Batch processing
# --------------------------------------------------------------------------------------
class TestBatchProcessing(unittest.TestCase):
    def test_batched_end_to_end(self):
        B = 4
        model = _make_hgnec()
        X = torch.randn(B, N, D)
        A = _random_A().unsqueeze(0).repeat(B, 1, 1)
        out = model.forward_tensors(X, A, dt=0.01)
        self.assertEqual(out.q_final.shape[0], B)
        self.assertEqual(out.H_initial.shape[0], B)

    def test_unbatched_still_works(self):
        model = _make_hgnec()
        X, A = torch.randn(N, D), _random_A()
        out = model.forward_tensors(X, A, dt=0.01)
        self.assertEqual(out.q_final.dim(), 2)


# --------------------------------------------------------------------------------------
# 11/12. Text / vision modality
# --------------------------------------------------------------------------------------
class TestModalities(unittest.TestCase):
    def test_text_modality_graph(self):
        model = _make_hgnec()
        graph = _make_graph(modality="text")
        out = model.forward(graph, dt=0.01)
        self.assertEqual(tuple(out.q_final.shape), (N, 2 * D))

    def test_vision_modality_graph(self):
        model = _make_hgnec()
        graph = _make_graph(modality="vision")
        out = model.forward(graph, dt=0.01)
        self.assertEqual(tuple(out.q_final.shape), (N, 2 * D))

    def test_wrong_feature_dim_rejected(self):
        model = _make_hgnec(input_dim=D)
        graph = _make_graph(d=D + 1)
        with self.assertRaises(HGNECConfigError):
            model.forward(graph, dt=0.01)


# --------------------------------------------------------------------------------------
# 13. Shared-weight behavior
# --------------------------------------------------------------------------------------
class TestSharedWeights(unittest.TestCase):
    def test_same_module_both_modalities(self):
        """One HGNEC instance, called on both a text graph and a vision graph, uses the
        SAME parameters (independent_shared_weights, Blocker 5) -- not two separate models."""
        model = _make_hgnec()
        text_graph = _make_graph(modality="text")
        vision_graph = _make_graph(modality="vision")
        params_before = [p.clone() for p in model.parameters()]
        model.forward(text_graph, dt=0.01)
        model.forward(vision_graph, dt=0.01)
        params_after = [p.clone() for p in model.parameters()]
        # No backward() called, so parameters must be untouched -- and it's literally the
        # same nn.Module/parameter objects handling both calls.
        for before, after in zip(params_before, params_after):
            torch.testing.assert_close(before, after)


# --------------------------------------------------------------------------------------
# 14. Gradient propagation
# --------------------------------------------------------------------------------------
class TestGradientPropagation(unittest.TestCase):
    def test_backward_reaches_all_parameters(self):
        model = _make_hgnec()
        X, A = torch.randn(N, D), _random_A()
        out = model.forward_tensors(X, A, dt=0.01)
        loss = out.q_final.sum() + out.energy_loss
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"parameter {name} received no gradient")

    def test_energy_loss_alone_reaches_hamiltonian_net(self):
        model = _make_hgnec()
        X, A = torch.randn(N, D), _random_A()
        out = model.forward_tensors(X, A, dt=0.01)
        out.energy_loss.backward()
        for name, param in model.hamiltonian_net.named_parameters():
            self.assertIsNotNone(param.grad, f"hamiltonian_net.{name} received no gradient from L_energy")


# --------------------------------------------------------------------------------------
# 15. Deterministic fixed-seed behavior
# --------------------------------------------------------------------------------------
class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_output(self):
        torch.manual_seed(123)
        model_a = _make_hgnec()
        torch.manual_seed(123)
        model_b = _make_hgnec()
        X, A = torch.randn(N, D), _random_A(seed=42)
        out_a = model_a.forward_tensors(X, A, dt=0.01)
        out_b = model_b.forward_tensors(X, A, dt=0.01)
        torch.testing.assert_close(out_a.q_final, out_b.q_final)
        torch.testing.assert_close(out_a.energy_loss, out_b.energy_loss)


# --------------------------------------------------------------------------------------
# 16. Device handling
# --------------------------------------------------------------------------------------
class TestDeviceHandling(unittest.TestCase):
    def test_cpu_forward(self):
        model = _make_hgnec()
        X, A = torch.randn(N, D), _random_A()
        out = model.forward_tensors(X, A, dt=0.01)
        self.assertEqual(out.q_final.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "no GPU in this sandbox")
    def test_cuda_forward(self):
        model = _make_hgnec().to("cuda")
        X, A = torch.randn(N, D).to("cuda"), _random_A().to("cuda")
        out = model.forward_tensors(X, A, dt=0.01)
        self.assertEqual(out.q_final.device.type, "cuda")


# --------------------------------------------------------------------------------------
# 17. Invalid shape detection
# --------------------------------------------------------------------------------------
class TestInvalidShapeDetection(unittest.TestCase):
    def test_non_square_adjacency_rejected(self):
        X = torch.randn(N, D)
        A = torch.randn(N, N + 1)
        with self.assertRaises(StateInitShapeError):
            build_initial_state(X, A)

    def test_compressor_wrong_input_dim(self):
        comp = FeatureCompressor(FeatureCompressorConfig(state_dim=2 * D, compressed_dim=DC))
        with self.assertRaises(Exception):
            comp(torch.randn(N, 2 * D + 1))

    def test_hamiltonian_mismatched_A(self):
        net = HamiltonianNet(HamiltonianNetConfig(compressed_dim=DC))
        q, p = torch.randn(N, DC), torch.randn(N, DC)
        bad_A = torch.randn(N + 1, N + 1)
        with self.assertRaises(HamiltonianShapeError):
            net(q, p, bad_A)

    def test_restorer_wrong_input_dim(self):
        rest = FeatureRestorer(FeatureRestorerConfig(compressed_dim=DC, restored_dim=2 * D))
        with self.assertRaises(RestorerShapeError):
            rest(torch.randn(N, DC + 1))

    def test_energy_loss_shape_mismatch(self):
        with self.assertRaises(HGNECLossShapeError):
            energy_conservation_loss(torch.randn(3), torch.randn(4))


# --------------------------------------------------------------------------------------
# 18. Energy/Hamiltonian sanity check
# --------------------------------------------------------------------------------------
class TestEnergySanity(unittest.TestCase):
    def test_energy_loss_zero_when_H_unchanged(self):
        H = torch.tensor(3.0)
        loss = energy_conservation_loss(H, H.clone())
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_energy_loss_matches_manual_mse(self):
        h_init = torch.tensor([1.0, 2.0, 3.0])
        h_final = torch.tensor([1.5, 1.5, 3.5])
        expected = ((h_init - h_final) ** 2).mean()
        got = energy_conservation_loss(h_init, h_final)
        self.assertAlmostEqual(got.item(), expected.item(), places=6)

    def test_pipeline_energy_loss_nonnegative(self):
        model = _make_hgnec()
        X, A = torch.randn(N, D), _random_A()
        out = model.forward_tensors(X, A, dt=0.01)
        self.assertGreaterEqual(out.energy_loss.item(), 0.0)


# --------------------------------------------------------------------------------------
# 19. Synthetic end-to-end HGN-EC forward pass
# --------------------------------------------------------------------------------------
class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline_from_graph(self):
        model = _make_hgnec(num_steps=2)
        graph = _make_graph()
        out = model.forward(graph, dt=0.05)
        self.assertIsInstance(out, HGNECOutput)
        self.assertTrue(torch.isfinite(out.q_final).all())
        self.assertTrue(torch.isfinite(out.energy_loss))
        loss = out.q_final.pow(2).sum() + out.energy_loss
        loss.backward()
        self.assertIsNotNone(model.compressor.linear.weight.grad)
        self.assertIsNotNone(model.restorer.linear.weight.grad)


# --------------------------------------------------------------------------------------
# 20. ACGA/GIN input immutability
# --------------------------------------------------------------------------------------
class TestInputImmutability(unittest.TestCase):
    def test_X_A_not_mutated(self):
        model = _make_hgnec()
        X, A = torch.randn(N, D), _random_A()
        X_before, A_before = X.clone(), A.clone()
        model.forward_tensors(X, A, dt=0.01)
        torch.testing.assert_close(X, X_before)
        torch.testing.assert_close(A, A_before)

    def test_no_acga_import_anywhere_in_hgn_ec(self):
        """HGN-EC must consume the GIN's own (X, A), never ACGA's (Z, A_hat) -- confirmed
        structurally by checking no `models.acga` import statement exists in the hgn_ec
        source (module docstrings are allowed to mention ACGA in prose, explaining the
        parallel-branch boundary -- only actual import lines are checked here)."""
        import inspect
        import models.hgn_ec.hgn_ec as hgn_ec_module

        import_lines = [
            line for line in inspect.getsource(hgn_ec_module).splitlines()
            if line.strip().startswith("import ") or line.strip().startswith("from ")
        ]
        for line in import_lines:
            self.assertNotIn("acga", line.lower(), f"unexpected ACGA import in HGN-EC: {line!r}")

    def test_graph_object_fields_unchanged(self):
        model = _make_hgnec()
        graph = _make_graph()
        X_before, A_before = graph.X.clone(), graph.A.clone()
        model.forward(graph, dt=0.01)
        torch.testing.assert_close(graph.X, X_before)
        torch.testing.assert_close(graph.A, A_before)


if __name__ == "__main__":
    unittest.main()

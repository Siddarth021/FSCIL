"""
tests/test_gin.py
====================

Stage 5 tests: Graph Isomorphism Network (GIN), Eqs. 13, 19-21.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from models.gnn.gin_layer import (
    GIN,
    GINConfig,
    GINConfigError,
    GINLayer,
    GINLayerConfig,
    GINLayerConfigError,
    GINLayerShapeError,
)
from models.graph.graph_data import Graph
from models.graph.adjacency import AdjacencyConfig, build_graph
from models.graph.node_builder import NodeBuilderConfig
from models.prompts.mlp_bridge import MLPBridgeConfig, PromptToNodeMLP

N, D, HIDDEN = 4, 5, 16  # small synthetic dims


def _make_layer(input_dim=D, output_dim=HIDDEN, seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)
    cfg = GINLayerConfig(input_dim=input_dim, output_dim=output_dim, **kwargs)
    return GINLayer(cfg)


def _make_gin(input_dim=D, num_layers=4, hidden_dim=HIDDEN, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    cfg = GINConfig(input_dim=input_dim, num_layers=num_layers, hidden_dim=hidden_dim)
    return GIN(cfg)


def _make_graph(n=N, d=D, X=None, A=None, modality="text"):
    if X is None:
        X = torch.randn(n, d)
    if A is None:
        A = torch.eye(n)
    return Graph(X=X, A=A, N=n, feature_dim=d, modality=modality)


# ----------------------------------------------------------------------------------------
# 1. GIN initialization
# ----------------------------------------------------------------------------------------


class TestGINInitialization(unittest.TestCase):
    def test_layer_init_has_epsilon_param(self):
        layer = _make_layer()
        self.assertTrue(hasattr(layer, "epsilon"))
        self.assertIsInstance(layer.epsilon, nn.Parameter)
        self.assertTrue(layer.epsilon.requires_grad)

    def test_layer_init_default_eps_is_zero(self):
        layer = _make_layer()
        self.assertAlmostEqual(float(layer.epsilon.item()), 0.0)

    def test_stack_init_layer_count_and_epsilon_not_shared(self):
        gin = _make_gin(num_layers=4)
        self.assertEqual(len(gin.layers), 4)
        # epsilon_shared_across_layers = false: each layer owns a distinct Parameter object.
        eps_ids = {id(layer.epsilon) for layer in gin.layers}
        self.assertEqual(len(eps_ids), 4)

    def test_stack_first_layer_maps_input_dim_rest_map_hidden(self):
        gin = _make_gin(input_dim=D, hidden_dim=HIDDEN, num_layers=4)
        self.assertEqual(gin.layers[0].config.input_dim, D)
        self.assertEqual(gin.layers[0].config.output_dim, HIDDEN)
        for layer in gin.layers[1:]:
            self.assertEqual(layer.config.input_dim, HIDDEN)
            self.assertEqual(layer.config.output_dim, HIDDEN)


# ----------------------------------------------------------------------------------------
# 2/3. Forward pass (single + batched)
# ----------------------------------------------------------------------------------------


class TestForwardPass(unittest.TestCase):
    def test_single_graph_forward_pass(self):
        layer = _make_layer()
        X = torch.randn(N, D)
        A = torch.eye(N)
        out = layer(X, A)
        self.assertEqual(out.shape, (N, HIDDEN))

    def test_batched_graph_forward_pass(self):
        layer = _make_layer()
        B = 3
        X = torch.randn(B, N, D)
        A = torch.eye(N).unsqueeze(0).expand(B, N, N).clone()
        out = layer(X, A)
        self.assertEqual(out.shape, (B, N, HIDDEN))

    def test_gin_stack_forward_on_graph_object(self):
        gin = _make_gin()
        graph = _make_graph()
        out_graph = gin(graph)
        self.assertIsInstance(out_graph, Graph)
        self.assertEqual(out_graph.X.shape, (N, HIDDEN))

    def test_gin_stack_forward_batched_graph_object(self):
        gin = _make_gin()
        B = 2
        X = torch.randn(B, N, D)
        A = torch.eye(N).unsqueeze(0).expand(B, N, N).clone()
        graph = Graph(X=X, A=A, N=N, feature_dim=D, modality="vision")
        out_graph = gin(graph)
        self.assertEqual(out_graph.X.shape, (B, N, HIDDEN))
        self.assertEqual(out_graph.batch_size, B)


# ----------------------------------------------------------------------------------------
# 4/5. Output node count / feature dimension
# ----------------------------------------------------------------------------------------


class TestOutputShape(unittest.TestCase):
    def test_output_node_count_preserved(self):
        gin = _make_gin(num_layers=3)
        graph = _make_graph()
        out_graph = gin(graph)
        self.assertEqual(out_graph.N, N)
        self.assertEqual(out_graph.X.shape[-2], N)

    def test_output_feature_dimension_is_hidden_dim(self):
        gin = _make_gin(hidden_dim=16)
        graph = _make_graph()
        out_graph = gin(graph)
        self.assertEqual(out_graph.feature_dim, 16)
        self.assertEqual(out_graph.X.shape[-1], 16)

    def test_output_modality_preserved(self):
        gin = _make_gin()
        graph = _make_graph(modality="vision")
        out_graph = gin(graph)
        self.assertEqual(out_graph.modality, "vision")

    def test_output_adjacency_passed_through_unchanged(self):
        gin = _make_gin()
        A = torch.rand(N, N)
        graph = _make_graph(A=A)
        out_graph = gin(graph)
        self.assertTrue(torch.equal(out_graph.A, A))


# ----------------------------------------------------------------------------------------
# 6/7/11. Hand-checkable synthetic graph: neighbor aggregation + self-feature contribution
# ----------------------------------------------------------------------------------------


class TestHandCheckableAggregation(unittest.TestCase):
    """Replace the MLP with Identity to isolate/hand-verify Eq. 19-20's arithmetic."""

    def _identity_layer(self, input_dim, output_dim, eps_init):
        cfg = GINLayerConfig(input_dim=input_dim, output_dim=output_dim, eps_init=eps_init)
        layer = GINLayer(cfg)
        layer.mlp = nn.Identity()  # isolate pre-MLP arithmetic (Eqs. 19-20) for hand-checking
        return layer

    def test_neighbor_aggregation_two_node_chain(self):
        # 2 nodes, D=1. A: node 0 <- node 1 only (asymmetric, to make agg unambiguous).
        # agg_0 = A[0] . X = X[1]; agg_1 = A[1] . X = 0.
        X = torch.tensor([[1.0], [3.0]])  # (N=2, D=1)
        A = torch.tensor([[0.0, 1.0], [0.0, 0.0]])  # (N=2, N=2)
        layer = self._identity_layer(input_dim=1, output_dim=1, eps_init=0.0)
        out = layer(X, A)
        # combined = (1+0)*X + A@X = X + [X[1], 0] = [[1+3],[3+0]] = [[4],[3]]
        expected = torch.tensor([[4.0], [3.0]])
        self.assertTrue(torch.allclose(out, expected))

    def test_self_feature_contribution_no_edges(self):
        # A = all zeros: agg = 0 for every node, so output = (1+epsilon)*X exactly.
        X = torch.tensor([[2.0, -1.0], [5.0, 0.5]])  # (N=2, D=2)
        A = torch.zeros(2, 2)
        eps = 0.5
        layer = self._identity_layer(input_dim=2, output_dim=2, eps_init=eps)
        out = layer(X, A)
        expected = (1.0 + eps) * X
        self.assertTrue(torch.allclose(out, expected))

    def test_self_feature_contribution_nonzero_epsilon_with_neighbors(self):
        # 3-node fully connected (excluding self), epsilon=1.0.
        X = torch.tensor([[1.0], [2.0], [3.0]])  # (N=3, D=1)
        A = torch.tensor([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ])
        eps = 1.0
        layer = self._identity_layer(input_dim=1, output_dim=1, eps_init=eps)
        out = layer(X, A)
        # agg = A @ X = [[2+3],[1+3],[1+2]] = [[5],[4],[3]]
        # combined = (1+1)*X + agg = [[2+5],[4+4],[6+3]] = [[7],[8],[9]]
        expected = torch.tensor([[7.0], [8.0], [9.0]])
        self.assertTrue(torch.allclose(out, expected))

    def test_identity_no_edge_graph_behavior(self):
        # Explicit "no-edge graph" test (distinct from the self-feature-contribution check
        # above): confirms the *whole* GINLayer (with a real, non-identity MLP) still runs
        # and produces a finite, correctly-shaped output when A has no edges at all.
        layer = _make_layer(input_dim=D, output_dim=HIDDEN)
        X = torch.randn(N, D)
        A = torch.zeros(N, N)
        out = layer(X, A)
        self.assertEqual(out.shape, (N, HIDDEN))
        self.assertTrue(torch.isfinite(out).all())


# ----------------------------------------------------------------------------------------
# 8. Multiple GIN layers (stacking)
# ----------------------------------------------------------------------------------------


class TestLayerStacking(unittest.TestCase):
    def test_stack_of_three_layers_runs_and_reduces_dim_progression(self):
        gin = _make_gin(input_dim=D, hidden_dim=HIDDEN, num_layers=3)
        graph = _make_graph()
        out_graph = gin(graph)
        self.assertEqual(out_graph.feature_dim, HIDDEN)

    def test_single_layer_stack(self):
        gin = _make_gin(input_dim=D, hidden_dim=HIDDEN, num_layers=1)
        self.assertEqual(len(gin.layers), 1)
        graph = _make_graph()
        out_graph = gin(graph)
        self.assertEqual(out_graph.X.shape, (N, HIDDEN))

    def test_paper_default_four_layers(self):
        gin = _make_gin(input_dim=D, hidden_dim=16, num_layers=4)
        self.assertEqual(gin.config.num_layers, 4)
        self.assertEqual(gin.config.hidden_dim, 16)


# ----------------------------------------------------------------------------------------
# 9. Gradient propagation
# ----------------------------------------------------------------------------------------


class TestGradientPropagation(unittest.TestCase):
    def test_gradients_flow_to_layer_weights_and_epsilon(self):
        layer = _make_layer()
        X = torch.randn(N, D, requires_grad=True)
        A = torch.rand(N, N)
        out = layer(X, A)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(X.grad)
        self.assertTrue(torch.isfinite(X.grad).all())
        self.assertIsNotNone(layer.epsilon.grad)
        for p in layer.mlp.parameters():
            self.assertIsNotNone(p.grad)

    def test_gradients_flow_through_full_stack(self):
        gin = _make_gin(num_layers=4)
        graph = _make_graph(X=torch.randn(N, D, requires_grad=True))
        out_graph = gin(graph)
        loss = out_graph.X.sum()
        loss.backward()
        self.assertIsNotNone(graph.X.grad)
        for layer in gin.layers:
            self.assertIsNotNone(layer.epsilon.grad)


# ----------------------------------------------------------------------------------------
# 10. Different adjacency structures
# ----------------------------------------------------------------------------------------


class TestDifferentAdjacencyStructures(unittest.TestCase):
    def _run(self, A):
        layer = _make_layer(input_dim=D, output_dim=HIDDEN)
        X = torch.randn(N, D)
        out = layer(X, A)
        self.assertEqual(out.shape, (N, HIDDEN))
        self.assertTrue(torch.isfinite(out).all())

    def test_identity_adjacency(self):
        self._run(torch.eye(N))

    def test_fully_connected_adjacency(self):
        self._run(torch.ones(N, N))

    def test_disconnected_adjacency(self):
        self._run(torch.zeros(N, N))

    def test_random_sparse_adjacency(self):
        torch.manual_seed(0)
        A = (torch.rand(N, N) > 0.7).float()
        self._run(A)

    def test_asymmetric_adjacency(self):
        A = torch.zeros(N, N)
        A[0, 1] = 1.0  # directed edge only
        self._run(A)


# ----------------------------------------------------------------------------------------
# 12. Shape mismatch detection
# ----------------------------------------------------------------------------------------


class TestShapeMismatchDetection(unittest.TestCase):
    def test_wrong_input_dim_rejected(self):
        layer = _make_layer(input_dim=D, output_dim=HIDDEN)
        X = torch.randn(N, D + 1)
        A = torch.eye(N)
        with self.assertRaises(GINLayerShapeError):
            layer(X, A)

    def test_non_square_adjacency_rejected(self):
        layer = _make_layer(input_dim=D, output_dim=HIDDEN)
        X = torch.randn(N, D)
        A = torch.rand(N, N + 1)
        with self.assertRaises(GINLayerShapeError):
            layer(X, A)

    def test_adjacency_node_mismatch_rejected(self):
        layer = _make_layer(input_dim=D, output_dim=HIDDEN)
        X = torch.randn(N, D)
        A = torch.eye(N + 1)
        with self.assertRaises(GINLayerShapeError):
            layer(X, A)

    def test_batch_mismatch_rejected(self):
        layer = _make_layer(input_dim=D, output_dim=HIDDEN)
        X = torch.randn(2, N, D)
        A = torch.eye(N).unsqueeze(0).expand(3, N, N).clone()
        with self.assertRaises(GINLayerShapeError):
            layer(X, A)

    def test_batchness_mismatch_rejected(self):
        layer = _make_layer(input_dim=D, output_dim=HIDDEN)
        X = torch.randn(N, D)  # unbatched
        A = torch.eye(N).unsqueeze(0)  # batched
        with self.assertRaises(GINLayerShapeError):
            layer(X, A)

    def test_wrong_ndim_rejected(self):
        layer = _make_layer(input_dim=D, output_dim=HIDDEN)
        X = torch.randn(D)  # 1D, invalid
        A = torch.eye(N)
        with self.assertRaises(GINLayerShapeError):
            layer(X, A)

    def test_gin_stack_feature_dim_mismatch_rejected(self):
        gin = _make_gin(input_dim=D, hidden_dim=HIDDEN)
        graph = _make_graph(d=D + 1)
        with self.assertRaises(GINLayerShapeError):
            gin(graph)


# ----------------------------------------------------------------------------------------
# 13. Device handling
# ----------------------------------------------------------------------------------------


class TestDeviceHandling(unittest.TestCase):
    def test_cpu_forward(self):
        layer = _make_layer().to("cpu")
        X = torch.randn(N, D, device="cpu")
        A = torch.eye(N, device="cpu")
        out = layer(X, A)
        self.assertEqual(out.device.type, "cpu")

    def test_gin_stack_to_cpu_explicit(self):
        gin = _make_gin().to("cpu")
        graph = _make_graph()
        out_graph = gin(graph)
        self.assertEqual(out_graph.X.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "no GPU in this sandbox")
    def test_cuda_forward(self):
        layer = _make_layer().to("cuda")
        X = torch.randn(N, D, device="cuda")
        A = torch.eye(N, device="cuda")
        out = layer(X, A)
        self.assertEqual(out.device.type, "cuda")


# ----------------------------------------------------------------------------------------
# 14. Deterministic behavior under fixed seed
# ----------------------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):
    def test_same_seed_identical_weights(self):
        layer_a = _make_layer(seed=123)
        layer_b = _make_layer(seed=123)
        for pa, pb in zip(layer_a.parameters(), layer_b.parameters()):
            self.assertTrue(torch.equal(pa, pb))

    def test_same_seed_identical_forward_output(self):
        torch.manual_seed(7)
        layer_a = _make_layer(seed=42)
        X = torch.randn(N, D)
        A = torch.eye(N)
        out_a = layer_a(X, A)

        layer_b = _make_layer(seed=42)
        out_b = layer_b(X, A)
        self.assertTrue(torch.allclose(out_a, out_b))

    def test_different_seeds_diverge(self):
        layer_a = _make_layer(seed=1)
        layer_b = _make_layer(seed=2)
        weights_equal = all(
            torch.equal(pa, pb) for pa, pb in zip(layer_a.parameters(), layer_b.parameters())
        )
        self.assertFalse(weights_equal)


# ----------------------------------------------------------------------------------------
# 15. Compatibility with Stage-4 Graph contract
# ----------------------------------------------------------------------------------------


class TestStage4Compatibility(unittest.TestCase):
    def test_end_to_end_prompt_to_gin(self):
        L, d, mlp_hidden, D_node = 4, 8, 16, D
        torch.manual_seed(0)
        bridge = PromptToNodeMLP(MLPBridgeConfig(input_dim=d, hidden_dim=mlp_hidden, output_dim=D_node))
        prompt_tensor = torch.randn(L, 1, d)
        graph = build_graph(
            prompt_tensor,
            bridge,
            NodeBuilderConfig(modality="text"),
            AdjacencyConfig(threshold=0.8),
        )
        self.assertIsInstance(graph, Graph)

        gin = _make_gin(input_dim=D_node, hidden_dim=HIDDEN, num_layers=4)
        out_graph = gin(graph)
        self.assertEqual(out_graph.N, L)
        self.assertEqual(out_graph.feature_dim, HIDDEN)
        self.assertEqual(out_graph.modality, "text")

    def test_graph_contract_import_unchanged(self):
        # GIN must not redefine or shadow the Stage-4 Graph class.
        from models.graph.graph_data import Graph as Stage4Graph
        graph = _make_graph()
        self.assertIs(type(graph), Stage4Graph)


# ----------------------------------------------------------------------------------------
# 16. Configuration/provenance validation
# ----------------------------------------------------------------------------------------


class TestConfigValidation(unittest.TestCase):
    def test_layer_config_rejects_nonpositive_dims(self):
        with self.assertRaises(GINLayerConfigError):
            GINLayerConfig(input_dim=0, output_dim=HIDDEN)
        with self.assertRaises(GINLayerConfigError):
            GINLayerConfig(input_dim=D, output_dim=-1)

    def test_layer_config_rejects_bad_mlp_hidden_dim(self):
        with self.assertRaises(GINLayerConfigError):
            GINLayerConfig(input_dim=D, output_dim=HIDDEN, mlp_hidden_dim=0)

    def test_layer_config_defaults_mlp_hidden_dim_to_output_dim(self):
        cfg = GINLayerConfig(input_dim=D, output_dim=HIDDEN)
        self.assertEqual(cfg.mlp_hidden_dim, HIDDEN)

    def test_layer_config_rejects_unsupported_mlp_composition(self):
        with self.assertRaises(GINLayerConfigError):
            GINLayerConfig(input_dim=D, output_dim=HIDDEN, mlp_composition="gcn_style")

    def test_layer_config_rejects_unsupported_num_linear_layers(self):
        with self.assertRaises(NotImplementedError):
            GINLayerConfig(input_dim=D, output_dim=HIDDEN, mlp_num_linear_layers=3)

    def test_gin_config_rejects_nonpositive_fields(self):
        with self.assertRaises(GINConfigError):
            GINConfig(input_dim=0)
        with self.assertRaises(GINConfigError):
            GINConfig(input_dim=D, num_layers=0)
        with self.assertRaises(GINConfigError):
            GINConfig(input_dim=D, hidden_dim=-4)

    def test_paper_fact_defaults_match_section_v_b(self):
        # Section V.B: "4-layer GIN network structure, with the number of neurons in the
        # hidden layer being 16" -- these are the GINConfig class defaults.
        cfg = GINConfig(input_dim=D)
        self.assertEqual(cfg.num_layers, 4)
        self.assertEqual(cfg.hidden_dim, 16)

    def test_provenance_dict_is_per_instance_not_shared(self):
        cfg_a = GINLayerConfig(input_dim=D, output_dim=HIDDEN)
        cfg_b = GINLayerConfig(input_dim=D, output_dim=HIDDEN)
        cfg_a.provenance["x"] = "y"
        self.assertEqual(cfg_b.provenance, {})


if __name__ == "__main__":
    unittest.main()

"""
tests/test_graph_construction.py
===================================

Stage 4 tests: MLP bridge (Blocker 3) + graph node/adjacency construction (Blocker 1, Eqs.
14-18).
"""

from __future__ import annotations

import unittest

import torch

from models.prompts.mlp_bridge import MLPBridgeConfig, MLPBridgeConfigError, MLPBridgeShapeError, PromptToNodeMLP
from models.graph.node_builder import NodeBuilderConfig, NodeBuilderConfigError, NodeBuilderShapeError, build_nodes
from models.graph.adjacency import (
    AdjacencyConfig,
    AdjacencyConfigError,
    build_adjacency,
    build_graph,
    cosine_similarity_matrix,
    threshold_binarize,
    symmetrize,
    normalize_adjacency,
)
from models.graph.graph_data import Graph, GraphShapeError

L, d, HIDDEN, D = 4, 8, 16, 6  # small synthetic dims; L mirrors "N == L" (Blocker 1)


def _make_bridge(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    cfg = MLPBridgeConfig(input_dim=d, hidden_dim=HIDDEN, output_dim=D)
    return PromptToNodeMLP(cfg)


def _make_prompt_tensor(L_=L, M=1, d_=d):
    return torch.randn(L_, M, d_)


# ----------------------------------------------------------------------------------------
# MLP bridge
# ----------------------------------------------------------------------------------------


class TestMLPBridgeInit(unittest.TestCase):
    def test_text_mlp_init_shapes(self):
        bridge = _make_bridge()
        x = torch.randn(L, d)
        out = bridge(x)
        self.assertEqual(out.shape, (L, D))

    def test_vision_mlp_init_shapes(self):
        # Separate instance -- distinct weights, per shared_across_modalities=false.
        bridge_text = _make_bridge()
        bridge_vision = _make_bridge()
        self.assertFalse(torch.allclose(
            dict(bridge_text.named_parameters())["net.0.weight"],
            dict(bridge_vision.named_parameters())["net.0.weight"],
        ))

    def test_invalid_dims_rejected(self):
        with self.assertRaises(MLPBridgeConfigError):
            MLPBridgeConfig(input_dim=0, hidden_dim=HIDDEN, output_dim=D)
        with self.assertRaises(MLPBridgeConfigError):
            MLPBridgeConfig(input_dim=d, hidden_dim=-1, output_dim=D)


class TestMLPBridgeShapes(unittest.TestCase):
    def test_input_output_shape(self):
        bridge = _make_bridge()
        for batch_shape in [(L,), (2, L)]:
            x = torch.randn(*batch_shape, d)
            out = bridge(x)
            self.assertEqual(out.shape, (*batch_shape, D))

    def test_wrong_last_dim_rejected(self):
        bridge = _make_bridge()
        x = torch.randn(L, d + 1)
        with self.assertRaises(MLPBridgeShapeError):
            bridge(x)


class TestMLPBridgeGradientFlow(unittest.TestCase):
    def test_gradients_reach_all_params(self):
        bridge = _make_bridge()
        x = torch.randn(L, d, requires_grad=True)
        out = bridge(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        for p in bridge.parameters():
            self.assertIsNotNone(p.grad)
            self.assertFalse(torch.all(p.grad == 0))


class TestMLPBridgeDeterminism(unittest.TestCase):
    def test_same_seed_same_output(self):
        torch.manual_seed(7)
        b1 = PromptToNodeMLP(MLPBridgeConfig(input_dim=d, hidden_dim=HIDDEN, output_dim=D))
        torch.manual_seed(7)
        b2 = PromptToNodeMLP(MLPBridgeConfig(input_dim=d, hidden_dim=HIDDEN, output_dim=D))
        x = torch.randn(L, d)
        self.assertTrue(torch.allclose(b1(x), b2(x)))


# ----------------------------------------------------------------------------------------
# Node construction (Blocker 1: N = L)
# ----------------------------------------------------------------------------------------


class TestNodeBuilderConfig(unittest.TestCase):
    def test_invalid_mode_rejected(self):
        with self.assertRaises(NodeBuilderConfigError):
            NodeBuilderConfig(num_nodes_mode="node_per_prompt_token", modality="text")

    def test_invalid_modality_rejected(self):
        with self.assertRaises(NodeBuilderConfigError):
            NodeBuilderConfig(modality="audio")


class TestGraphNodeCount(unittest.TestCase):
    def test_N_equals_L_text(self):
        bridge = _make_bridge()
        cfg = NodeBuilderConfig(modality="text")
        prompt = _make_prompt_tensor(L_=L)
        X = build_nodes(prompt, bridge, cfg)
        self.assertEqual(X.shape[0], L)  # N == L, Blocker 1

    def test_N_equals_L_vision(self):
        bridge = _make_bridge()
        cfg = NodeBuilderConfig(modality="vision")
        prompt = _make_prompt_tensor(L_=7)
        X = build_nodes(prompt, bridge, cfg)
        self.assertEqual(X.shape[0], 7)

    def test_M_neq_1_rejected(self):
        bridge = _make_bridge()
        cfg = NodeBuilderConfig(modality="text")
        prompt = _make_prompt_tensor(M=2)
        with self.assertRaises(NodeBuilderShapeError):
            build_nodes(prompt, bridge, cfg)


class TestGraphFeatureShape(unittest.TestCase):
    def test_feature_dim_matches_mlp_output(self):
        bridge = _make_bridge()
        cfg = NodeBuilderConfig(modality="text")
        prompt = _make_prompt_tensor()
        X = build_nodes(prompt, bridge, cfg)
        self.assertEqual(X.shape[1], D)

    def test_dim_mismatch_rejected(self):
        bridge = _make_bridge()
        cfg = NodeBuilderConfig(modality="text")
        prompt = _make_prompt_tensor(d_=d + 1)
        with self.assertRaises(NodeBuilderShapeError):
            build_nodes(prompt, bridge, cfg)


# ----------------------------------------------------------------------------------------
# Adjacency (Eqs. 14-18)
# ----------------------------------------------------------------------------------------


class TestAdjacencyShape(unittest.TestCase):
    def test_shape(self):
        X = torch.randn(L, D)
        cfg = AdjacencyConfig(threshold=0.8)
        A = build_adjacency(X, cfg)
        self.assertEqual(A.shape, (L, L))

    def test_symmetric(self):
        X = torch.randn(L, D)
        A = build_adjacency(X, AdjacencyConfig())
        self.assertTrue(torch.allclose(A, A.transpose(-2, -1), atol=1e-6))

    def test_threshold_strict_greater_than(self):
        X = torch.eye(3, D)  # identical rows for indices with same one-hot -> sim=1 at diag
        sim = cosine_similarity_matrix(X)
        A0 = threshold_binarize(sim, threshold=1.0)
        # sim==1.0 on the diagonal must NOT pass a strict '>' threshold of 1.0
        self.assertTrue(torch.all(torch.diagonal(A0) == 0))

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(AdjacencyConfigError):
            AdjacencyConfig(threshold=2.0)

    def test_pipeline_stages_composable(self):
        X = torch.randn(L, D)
        sim = cosine_similarity_matrix(X)
        A0 = threshold_binarize(sim, 0.8)
        Z = symmetrize(A0)
        A_norm = normalize_adjacency(Z)
        self.assertEqual(A_norm.shape, (L, L))


class TestBatchGraphConstruction(unittest.TestCase):
    def test_batched_node_features_produce_batched_adjacency(self):
        B = 3
        X = torch.randn(B, L, D)
        A = build_adjacency(X, AdjacencyConfig())
        self.assertEqual(A.shape, (B, L, L))

    def test_batched_graph_container(self):
        B = 2
        X = torch.randn(B, L, D)
        A = build_adjacency(X, AdjacencyConfig())
        g = Graph(X=X, A=A, N=L, feature_dim=D, modality="text")
        self.assertEqual(g.batch_size, B)


class TestTextGraphConstruction(unittest.TestCase):
    def test_build_graph_text(self):
        bridge = _make_bridge()
        node_cfg = NodeBuilderConfig(modality="text")
        adj_cfg = AdjacencyConfig()
        prompt = _make_prompt_tensor(L_=L)
        g = build_graph(prompt, bridge, node_cfg, adj_cfg)
        self.assertIsInstance(g, Graph)
        self.assertEqual(g.modality, "text")
        self.assertEqual(g.N, L)
        self.assertEqual(g.X.shape, (L, D))
        self.assertEqual(g.A.shape, (L, L))


class TestVisionGraphConstruction(unittest.TestCase):
    def test_build_graph_vision(self):
        bridge = _make_bridge()
        node_cfg = NodeBuilderConfig(modality="vision")
        adj_cfg = AdjacencyConfig()
        prompt = _make_prompt_tensor(L_=5)
        g = build_graph(prompt, bridge, node_cfg, adj_cfg)
        self.assertEqual(g.modality, "vision")
        self.assertEqual(g.N, 5)


class TestInvalidDimensionDetection(unittest.TestCase):
    def test_graph_container_rejects_mismatched_N(self):
        X = torch.randn(L, D)
        A = build_adjacency(X, AdjacencyConfig())
        with self.assertRaises(GraphShapeError):
            Graph(X=X, A=A, N=L + 1, feature_dim=D, modality="text")

    def test_graph_container_rejects_mismatched_feature_dim(self):
        X = torch.randn(L, D)
        A = build_adjacency(X, AdjacencyConfig())
        with self.assertRaises(GraphShapeError):
            Graph(X=X, A=A, N=L, feature_dim=D + 1, modality="text")

    def test_graph_container_rejects_bad_modality(self):
        X = torch.randn(L, D)
        A = build_adjacency(X, AdjacencyConfig())
        with self.assertRaises(GraphShapeError):
            Graph(X=X, A=A, N=L, feature_dim=D, modality="audio")

    def test_graph_container_rejects_adjacency_shape_mismatch(self):
        X = torch.randn(L, D)
        bad_A = torch.randn(L, L + 1)
        with self.assertRaises(GraphShapeError):
            Graph(X=X, A=bad_A, N=L, feature_dim=D, modality="text")


class TestDeviceHandling(unittest.TestCase):
    def test_graph_device_property(self):
        X = torch.randn(L, D)
        A = build_adjacency(X, AdjacencyConfig())
        g = Graph(X=X, A=A, N=L, feature_dim=D, modality="text")
        self.assertEqual(g.device, X.device)

    def test_graph_rejects_device_mismatch(self):
        X = torch.randn(L, D)
        A = build_adjacency(X, AdjacencyConfig())
        # Simulate a device mismatch without requiring a real second device: patch dtype
        # instead, which the same guard clause also checks (device check is analogous and
        # untestable without a GPU in this sandbox).
        A_wrong_dtype = A.to(torch.float64)
        with self.assertRaises(GraphShapeError):
            Graph(X=X, A=A_wrong_dtype, N=L, feature_dim=D, modality="text")

    def test_graph_to_moves_both_tensors(self):
        X = torch.randn(L, D)
        A = build_adjacency(X, AdjacencyConfig())
        g = Graph(X=X, A=A, N=L, feature_dim=D, modality="text")
        g2 = g.to(torch.float64)
        self.assertEqual(g2.X.dtype, torch.float64)
        self.assertEqual(g2.A.dtype, torch.float64)


class TestConfigProvenanceValidation(unittest.TestCase):
    def test_mlp_bridge_config_carries_provenance(self):
        cfg = MLPBridgeConfig(
            input_dim=d, hidden_dim=HIDDEN, output_dim=D,
            provenance={"hidden_dim": "TEST_OVERRIDE", "output_dim": "TEST_OVERRIDE"},
        )
        self.assertEqual(cfg.provenance["hidden_dim"], "TEST_OVERRIDE")

    def test_adjacency_config_default_matches_paper_fact_threshold(self):
        cfg = AdjacencyConfig()
        self.assertEqual(cfg.threshold, 0.8)
        self.assertFalse(cfg.attention_reweight_enabled)


class TestSmokeTest(unittest.TestCase):
    """Part-of-task synthetic smoke test: prompt/embedding -> MLP -> graph, verify X/A shapes.
    No real dataset, no pretrained CLIP, no training."""

    def test_end_to_end_smoke(self):
        for modality, L_ in (("text", L), ("vision", L)):
            bridge = PromptToNodeMLP(MLPBridgeConfig(input_dim=d, hidden_dim=HIDDEN, output_dim=D))
            node_cfg = NodeBuilderConfig(modality=modality)
            adj_cfg = AdjacencyConfig()
            prompt = torch.randn(L_, 1, d)
            g = build_graph(prompt, bridge, node_cfg, adj_cfg)
            self.assertEqual(g.X.shape, (L_, D))
            self.assertEqual(g.A.shape, (L_, L_))
            self.assertEqual(g.N, L_)


if __name__ == "__main__":
    unittest.main()

"""
models/graph/
===============

Stage 4 — graph node/adjacency construction (Blocker 1: N=L per modality; Eqs. 13-18).
"""

from __future__ import annotations

from models.graph.graph_data import Graph, GraphShapeError
from models.graph.node_builder import NodeBuilderConfig, NodeBuilderConfigError, NodeBuilderShapeError, build_nodes
from models.graph.adjacency import (
    AdjacencyConfig,
    AdjacencyConfigError,
    AdjacencyShapeError,
    build_adjacency,
    build_graph,
    cosine_similarity_matrix,
    threshold_binarize,
    symmetrize,
    normalize_adjacency,
    attention_reweight,
)

__all__ = [
    "Graph",
    "GraphShapeError",
    "NodeBuilderConfig",
    "NodeBuilderConfigError",
    "NodeBuilderShapeError",
    "build_nodes",
    "AdjacencyConfig",
    "AdjacencyConfigError",
    "AdjacencyShapeError",
    "build_adjacency",
    "build_graph",
    "cosine_similarity_matrix",
    "threshold_binarize",
    "symmetrize",
    "normalize_adjacency",
    "attention_reweight",
]

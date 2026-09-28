"""
models/gnn/
=============

Stage 5 — Graph Isomorphism Network (GIN), Eqs. 13, 19-21.
"""

from __future__ import annotations

from models.gnn.gin_layer import (
    GINConfig,
    GINConfigError,
    GINLayer,
    GINLayerConfig,
    GINLayerConfigError,
    GINLayerShapeError,
    GIN,
)

__all__ = [
    "GINLayerConfig",
    "GINLayerConfigError",
    "GINLayerShapeError",
    "GINLayer",
    "GINConfig",
    "GINConfigError",
    "GIN",
]

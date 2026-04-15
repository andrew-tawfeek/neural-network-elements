"""Polyhedral decomposition + dual-graph construction for ReLU networks.

Mirrors the grid-sampling approach used by `js/decision-boundary.js` and `js/dual-graph.js`:
each hidden-layer activation pattern (one bit per hidden neuron) labels a region in input space.
Adjacent regions in the dual graph are those whose binary patterns differ by exactly one bit
(Hamming distance 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import MultiLayerNetwork


@dataclass
class Decomposition:
    region_grid: np.ndarray          # (resolution, resolution) int array of region IDs
    region_patterns: np.ndarray      # (n_regions, n_hidden) uint8, one row per unique region
    region_ids: np.ndarray           # (n_regions,) the IDs in the same order as region_patterns
    extent: tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax) for matplotlib imshow


def compute_decomposition(
    net: MultiLayerNetwork,
    x_min: float = -3.0,
    x_max: float = 3.0,
    y_min: float = -3.0,
    y_max: float = 3.0,
    resolution: int = 100,
) -> Decomposition:
    """Sample a regular grid in [x_min, x_max] x [y_min, y_max] and label each cell by its
    hidden-layer binary activation pattern. Requires `net.architecture[0] == 2`."""
    if net.architecture[0] != 2:
        raise ValueError("compute_decomposition requires a 2-input network")

    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(xs, ys)
    inputs = np.stack([xx.ravel(), yy.ravel()], axis=0)  # (2, N)

    _, masks = net.forward_batch(inputs)
    if not masks:
        # No hidden layers: every point is in the same region.
        region_grid = np.zeros((resolution, resolution), dtype=np.int64)
        return Decomposition(
            region_grid=region_grid,
            region_patterns=np.zeros((1, 0), dtype=np.uint8),
            region_ids=np.array([0], dtype=np.int64),
            extent=(x_min, x_max, y_min, y_max),
        )

    pattern_matrix = np.concatenate(masks, axis=0).T  # (N, n_hidden)
    unique_patterns, inverse = np.unique(pattern_matrix, axis=0, return_inverse=True)
    region_grid = inverse.reshape(resolution, resolution)
    region_ids = np.arange(unique_patterns.shape[0], dtype=np.int64)

    return Decomposition(
        region_grid=region_grid,
        region_patterns=unique_patterns.astype(np.uint8),
        region_ids=region_ids,
        extent=(x_min, x_max, y_min, y_max),
    )


def build_dual_graph(decomp: Decomposition) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Construct the dual graph where nodes are unique regions and edges connect regions whose
    binary patterns differ by exactly one bit. Returns (centroids, edges) with centroids of shape
    (n_regions, 2) in real coordinates and edges as (i, j) index pairs into that array.
    """
    patterns = decomp.region_patterns
    n_regions = patterns.shape[0]

    # Centroids in real coordinates.
    res_y, res_x = decomp.region_grid.shape
    x_min, x_max, y_min, y_max = decomp.extent
    xs = np.linspace(x_min, x_max, res_x)
    ys = np.linspace(y_min, y_max, res_y)
    xx, yy = np.meshgrid(xs, ys)

    centroids = np.zeros((n_regions, 2), dtype=float)
    for rid in range(n_regions):
        mask = decomp.region_grid == rid
        if not mask.any():
            continue
        centroids[rid, 0] = xx[mask].mean()
        centroids[rid, 1] = yy[mask].mean()

    # Hamming-distance-1 edges (vectorized).
    edges: list[tuple[int, int]] = []
    if patterns.shape[1] > 0 and n_regions > 1:
        # XOR all pairs and count differing bits.
        diff = patterns[:, None, :] ^ patterns[None, :, :]
        hamming = diff.sum(axis=2)
        i_idx, j_idx = np.where(np.triu(hamming == 1, k=1))
        edges = list(zip(i_idx.tolist(), j_idx.tolist()))

    return centroids, edges

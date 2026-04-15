"""Exact polyhedral decomposition of a 2-D input plane under a ReLU network.

Each hidden neuron's activation boundary is the zero-set of its pre-activation, which — restricted
to any region where the sign pattern of all earlier hidden neurons is fixed — is an affine
function of the input and therefore a straight line. We walk the hidden layers neuron-by-neuron,
splitting every current region by that neuron's boundary line and tracking the running affine map
``x ↦ a_l(x) = A_l x + c_l`` from input to the post-ReLU activation of the deepest processed layer.

The result is the set of maximal convex polyhedra over which the whole network is affine, tagged
by their hidden-layer binary activation patterns — i.e. exactly the decomposition that the JS
version approximates by grid sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .network import MultiLayerNetwork
from .polytope import polygon_centroid, split_convex_polygon


@dataclass
class Region:
    polygon: np.ndarray        # (k, 2) CCW vertices in input-plane coordinates
    pattern: tuple[int, ...]   # binary activation pattern over all hidden neurons (layer-flat)

    @property
    def centroid(self) -> np.ndarray:
        return polygon_centroid(self.polygon)


@dataclass
class Decomposition:
    regions: list[Region]
    extent: tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    # Boundary segments grouped by the hidden-neuron index (flat, layer-by-layer) that created
    # them. Each entry is (p, q) — the two endpoints of a line segment on that neuron's
    # activation boundary, clipped to the containing region.
    neuron_segments: list[list[tuple[np.ndarray, np.ndarray]]]

    @property
    def patterns(self) -> np.ndarray:
        if not self.regions:
            return np.zeros((0, 0), dtype=np.uint8)
        return np.array([r.pattern for r in self.regions], dtype=np.uint8)


def compute_decomposition(
    net: MultiLayerNetwork,
    x_min: float = -3.0,
    x_max: float = 3.0,
    y_min: float = -3.0,
    y_max: float = 3.0,
) -> Decomposition:
    """Compute the exact polyhedral decomposition over the rectangle [x_min,x_max]×[y_min,y_max].

    Requires ``net.architecture[0] == 2``. Output layer is ignored (it is affine everywhere given
    the hidden layers' sign pattern and so does not contribute a new bend).
    """
    if net.architecture[0] != 2:
        raise ValueError("compute_decomposition requires a 2-input network")

    initial = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=float
    )

    n_in = net.architecture[0]
    # Running affine map a_l(x) = A x + c; at depth 0, a_0(x) = x.
    working: list[tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray]] = [
        (initial, (), np.eye(n_in), np.zeros(n_in))
    ]

    neuron_segments: list[list[tuple[np.ndarray, np.ndarray]]] = []

    num_hidden_layers = net.layers - 2  # skip input and output layers
    for l in range(num_hidden_layers):
        W = net.weights[l]            # (layer_size, prev_size)
        b = net.biases[l].ravel()     # (layer_size,)
        layer_size = W.shape[0]

        # --- Step 1: split each current region by each neuron's boundary line.
        for i in range(layer_size):
            w_i = W[i]
            b_i = float(b[i])
            segments_for_neuron: list[tuple[np.ndarray, np.ndarray]] = []
            next_working = []
            for (poly, pat, A_cur, c_cur) in working:
                # Pre-activation as an affine function of x in this region:
                #   z_i(x) = w_i · (A_cur x + c_cur) + b_i = (w_i @ A_cur) · x + (w_i · c_cur + b_i)
                n_vec = w_i @ A_cur
                d_val = float(w_i @ c_cur) + b_i

                pos_poly, neg_poly = split_convex_polygon(poly, n_vec, d_val)

                if pos_poly is not None and neg_poly is not None:
                    # The splitting line actually crossed this region — collect the segment
                    # shared by both halves for visualization of the "bent hyperplane".
                    seg = _shared_edge(pos_poly, neg_poly)
                    if seg is not None:
                        segments_for_neuron.append(seg)

                if pos_poly is not None:
                    next_working.append((pos_poly, pat + (1,), A_cur, c_cur))
                if neg_poly is not None:
                    next_working.append((neg_poly, pat + (0,), A_cur, c_cur))
            working = next_working
            neuron_segments.append(segments_for_neuron)

        # --- Step 2: fold the ReLU mask into the affine map for each region.
        # a_{l+1}(x) = ReLU(W a_l(x) + b) = diag(s) (W A_cur x + W c_cur + b),
        # where s is the 0/1 bit pattern for this layer in this region.
        updated = []
        for (poly, pat, A_cur, c_cur) in working:
            s = np.array(pat[-layer_size:], dtype=float).reshape(-1, 1)
            A_new = s * (W @ A_cur)
            c_new = s.ravel() * (W @ c_cur + b)
            updated.append((poly, pat, A_new, c_new))
        working = updated

    regions = [Region(polygon=p, pattern=pat) for (p, pat, _, _) in working]
    return Decomposition(
        regions=regions,
        extent=(x_min, x_max, y_min, y_max),
        neuron_segments=neuron_segments,
    )


def _shared_edge(
    pos_poly: np.ndarray, neg_poly: np.ndarray, tol: float = 1e-7
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the two endpoints of the edge shared between the two halves of a split polygon.

    Both halves were produced by `split_convex_polygon`, so their boundaries include exactly two
    vertices that are coincident between the two polygons (the intersection points with the line).
    """
    matches: list[np.ndarray] = []
    for v in pos_poly:
        for u in neg_poly:
            if np.linalg.norm(v - u) < tol:
                matches.append(v)
                break
        if len(matches) == 2:
            break
    if len(matches) != 2:
        return None
    return matches[0], matches[1]


def build_dual_graph(decomp: Decomposition) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Dual graph of the decomposition.

    Nodes are the polyhedral regions (at their centroids); edges connect regions whose
    activation patterns differ by exactly one bit — equivalently, regions that share a
    full-dimensional edge in the decomposition.
    """
    n = len(decomp.regions)
    if n == 0:
        return np.zeros((0, 2)), []
    centroids = np.array([r.centroid for r in decomp.regions])
    if n < 2 or not decomp.regions[0].pattern:
        return centroids, []
    patterns = np.array([r.pattern for r in decomp.regions], dtype=np.int8)
    diff = patterns[:, None, :] ^ patterns[None, :, :]
    hamming = diff.sum(axis=2)
    iu, ju = np.where(np.triu(hamming == 1, k=1))
    return centroids, list(zip(iu.tolist(), ju.tolist()))


def region_containing(decomp: Decomposition, point: Sequence[float]) -> Region | None:
    """Return the region whose polygon contains `point`. Useful for lookups from a UI click."""
    px, py = float(point[0]), float(point[1])
    for region in decomp.regions:
        if _point_in_convex_polygon(region.polygon, px, py):
            return region
    return None


def _point_in_convex_polygon(poly: np.ndarray, px: float, py: float, eps: float = 1e-9) -> bool:
    k = len(poly)
    sign = 0
    for i in range(k):
        j = (i + 1) % k
        ex, ey = poly[j] - poly[i]
        rx, ry = px - poly[i, 0], py - poly[i, 1]
        cross = ex * ry - ey * rx
        if abs(cross) < eps:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

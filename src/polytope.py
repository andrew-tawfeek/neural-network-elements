"""Convex-polygon utilities used by the exact polyhedral decomposition.

The polygons here are represented as (k, 2) NumPy arrays of vertices in CCW order.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def polygon_area(poly: np.ndarray) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """Area-weighted centroid. Falls back to the mean of vertices if the polygon is degenerate."""
    if poly is None or len(poly) < 3:
        return poly.mean(axis=0) if poly is not None and len(poly) else np.zeros(2)
    x = poly[:, 0]
    y = poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    A2 = float(np.sum(cross))
    if abs(A2) < 1e-14:
        return poly.mean(axis=0)
    A6 = 3.0 * A2  # 6 * (A2/2)
    cx = float(np.sum((x + np.roll(x, -1)) * cross)) / A6
    cy = float(np.sum((y + np.roll(y, -1)) * cross)) / A6
    return np.array([cx, cy])


def split_convex_polygon(
    poly: np.ndarray,
    normal: np.ndarray,
    d: float,
    eps: float = _EPS,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Split a convex CCW polygon by the oriented line ``normal · x + d = 0``.

    Returns ``(positive, negative)`` — the polygon clipped to the two closed halfspaces
    ``{x : n·x + d ≥ 0}`` and ``{x : n·x + d ≤ 0}`` respectively. Either may be ``None`` if
    the corresponding side has no area. The shared line becomes an edge of both outputs.
    """
    if poly is None or len(poly) < 3:
        return None, None

    norm_n = float(np.linalg.norm(normal))
    if norm_n < 1e-14:
        # Degenerate: constant sign of `d` over the whole plane.
        return (poly, None) if d >= 0 else (None, poly)

    # Normalize so that `vals` are signed Euclidean distances — makes `eps` scale-free.
    n_hat = normal / norm_n
    d_hat = d / norm_n
    vals = poly @ n_hat + d_hat

    k = len(poly)
    pos_pts: list[np.ndarray] = []
    neg_pts: list[np.ndarray] = []
    for i in range(k):
        j = (i + 1) % k
        vi, vj = poly[i], poly[j]
        di, dj = float(vals[i]), float(vals[j])

        if di > eps:
            pos_pts.append(vi)
        elif di < -eps:
            neg_pts.append(vi)
        else:
            pos_pts.append(vi)
            neg_pts.append(vi)

        # Strict crossing only — if either endpoint is on the line it's already on both sides.
        if (di > eps and dj < -eps) or (di < -eps and dj > eps):
            t = di / (di - dj)
            inter = vi + t * (vj - vi)
            pos_pts.append(inter)
            neg_pts.append(inter)

    pos = np.asarray(pos_pts) if len(pos_pts) >= 3 else None
    neg = np.asarray(neg_pts) if len(neg_pts) >= 3 else None
    if pos is not None:
        pos = _dedupe_vertices(pos)
        if polygon_area(pos) < eps:
            pos = None
    if neg is not None:
        neg = _dedupe_vertices(neg)
        if polygon_area(neg) < eps:
            neg = None
    return pos, neg


def _dedupe_vertices(poly: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Drop consecutive duplicate vertices (which appear when the split line grazes a vertex)."""
    if len(poly) < 2:
        return poly
    keep = [poly[0]]
    for v in poly[1:]:
        if np.linalg.norm(v - keep[-1]) > tol:
            keep.append(v)
    # Handle wrap-around
    if len(keep) > 1 and np.linalg.norm(keep[0] - keep[-1]) <= tol:
        keep.pop()
    return np.asarray(keep) if len(keep) >= 3 else poly

from __future__ import annotations
import numpy as np
from scipy.spatial import Voronoi


def polygon_area(vertices: np.ndarray) -> float:
    """Shoelace area. vertices: (M,2) CCW or CW; unbounded check is caller's job."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_perimeter(vertices: np.ndarray) -> float:
    diffs = np.diff(np.vstack([vertices, vertices[0]]), axis=0)
    return np.sqrt((diffs * diffs).sum(axis=1)).sum()


def voronoi_edges(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, Voronoi]:
    """Return undirected edge list (src,dst) from Voronoi.ridge_points and the Voronoi itself."""
    vor = Voronoi(points)
    rp = vor.ridge_points
    # Unique undirected pairs
    src = rp[:, 0].astype(np.int32, copy=False)
    dst = rp[:, 1].astype(np.int32, copy=False)
    # Deduplicate: sort within rows then unique rows
    u = np.sort(np.stack([src, dst], axis=1), axis=1)
    u, idx = np.unique(u, axis=0, return_index=True)
    return u[:, 0], u[:, 1], vor


def cell_areas(points: np.ndarray, types: np.ndarray, vor: Voronoi) -> np.ndarray:
    """Area per cell; zero for boundary or unbounded regions (-1)."""
    N = points.shape[0]
    areas = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if types[i] == 1:
            areas[i] = 0.0
            continue
        reg_idx = vor.point_region[i]
        region = vor.regions[reg_idx]
        if len(region) == 0 or -1 in region:
            # Unbounded; caller should fix boundary; we choose 0.0 to match prior behavior.
            areas[i] = 0.0
        else:
            poly = vor.vertices[region]
            areas[i] = polygon_area(poly)
    return areas


def edges_from_voronoi(vor: Voronoi) -> tuple[np.ndarray, np.ndarray]:
    """Extract unique undirected edges (src, dst) from an existing Voronoi."""
    rp = vor.ridge_points.astype(np.int32, copy=False)
    u = np.sort(rp, axis=1)
    u, _ = np.unique(u, axis=0, return_index=True)
    return u[:, 0], u[:, 1]

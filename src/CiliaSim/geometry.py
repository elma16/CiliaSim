from __future__ import annotations
import numpy as np
from scipy.spatial import Voronoi


def polygon_area(vertices: np.ndarray) -> float:
    """
    Compute polygon area using the shoelace formula.

    Args:
        vertices: Array of shape (M, 2) with polygon vertices ordered CW or CCW.

    Returns:
        Absolute area of the polygon. The caller is responsible for excluding
        unbounded regions or invalid vertex lists.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_perimeter(vertices: np.ndarray) -> float:
    """
    Compute the perimeter of a closed polygon.

    Args:
        vertices: Array of shape (M, 2) with polygon vertices ordered CW or CCW.

    Returns:
        Total edge length, including the closing edge from last to first vertex.
    """
    diffs = np.diff(np.vstack([vertices, vertices[0]]), axis=0)
    return np.sqrt((diffs * diffs).sum(axis=1)).sum()


def voronoi_edges(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, Voronoi]:
    """
    Build an undirected edge list from a Voronoi tessellation.

    Args:
        points: Array of shape (N, 2) of site coordinates.

    Returns:
        Tuple of (src, dst, vor) where src and dst are int arrays of length E
        describing unique undirected edges between Voronoi neighbors, and vor
        is the computed SciPy Voronoi object.
    """
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
    """
    Compute Voronoi cell areas with boundary and unbounded handling.

    Args:
        points: Array of shape (N, 2) of site coordinates.
        types: Array of shape (N,) of cell types; boundary cells are type 1.
        vor: Precomputed Voronoi tessellation for points.

    Returns:
        Array of shape (N,) with area per cell. Boundary cells and cells with
        unbounded regions (-1 in region) are assigned area 0.0.
    """
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
    """
    Extract unique undirected edges from an existing Voronoi object.

    Args:
        vor: Precomputed Voronoi tessellation.

    Returns:
        Tuple of (src, dst) int arrays describing unique undirected edges
        between Voronoi neighbors.
    """
    rp = vor.ridge_points.astype(np.int32, copy=False)
    u = np.sort(rp, axis=1)
    u, _ = np.unique(u, axis=0, return_index=True)
    return u[:, 0], u[:, 1]

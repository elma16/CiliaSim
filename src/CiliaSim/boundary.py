from __future__ import annotations
import numpy as np
from scipy.spatial import Voronoi
from typing import Tuple


def _reflect_point_across_segment(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> Tuple[bool, np.ndarray]:
    """Mirror c across line segment ab if angle at c with a and b is obtuse (cos < 0)."""
    ac = c - a
    bc = c - b
    ac_norm = np.linalg.norm(ac)
    bc_norm = np.linalg.norm(bc)
    if ac_norm == 0.0 or bc_norm == 0.0:
        return False, c
    angle_cos = np.dot(ac, bc) / (ac_norm * bc_norm)
    if angle_cos < 0.0:
        edge = b - a
        en = np.linalg.norm(edge)
        if en == 0.0:
            return False, c
        u = edge / en
        proj_len = np.dot(ac, u)
        proj = proj_len * u
        reflected = 2.0 * (a + proj) - c
        return True, reflected
    return False, c


def build_voronoi_neighbors(vor: Voronoi, N: int) -> list[set[int]]:
    nb = [set() for _ in range(N)]
    for i, j in vor.ridge_points:
        nb[i].add(int(j))
        nb[j].add(int(i))
    return nb


def _has_unbounded_nonboundary(vor: Voronoi, types: np.ndarray) -> bool:
    """Return True if any non-boundary cell has an unbounded Voronoi region (-1 in region)."""
    N = types.shape[0]
    for i in range(N):
        if types[i] == 1:  # boundary
            continue
        reg = vor.regions[vor.point_region[i]]
        if len(reg) == 0 or (-1 in reg):
            return True
    return False


def _add_enclosing_boundary_ring(
    points: np.ndarray,
    types: np.ndarray,
    boundary_cycle: np.ndarray,
    n: int = 8,
    factor: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Append an outer ring of `n` boundary points around the current point cloud.
    The ring radius is `factor` times the max side of the bounding box (with a small floor).
    Returns updated (points, types, boundary_cycle) where the cycle is set to the new ring.
    """
    # center and scale from current points
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    center = 0.5 * (pmin + pmax)
    extent = pmax - pmin
    base = float(max(extent.max(), 1.0))
    R = factor * base

    angles = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)
    ring = np.stack(
        [center[0] + R * np.cos(angles), center[1] + R * np.sin(angles)], axis=1
    )

    N0 = points.shape[0]
    points = np.vstack([points, ring])
    types = np.append(types, np.ones(n, dtype=types.dtype))  # mark as boundary
    new_cycle = np.arange(N0, N0 + n, dtype=np.int64)
    return points, types, new_cycle


def evaluate_boundary(
    points: np.ndarray, types: np.ndarray, boundary_cycle: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Voronoi]:
    """
    Recompute Voronoi first, then ensure boundary cycle connectivity (prev/next neighbors),
    add mirrored boundary points where needed, prune sharp boundary nodes,
    rebuild adjacency inputs from fresh Voronoi.
    Returns (points, types, boundary_cycle, vor).
    """
    # First Voronoi pass to check boundedness
    vor = Voronoi(points)

    # If the current configuration cannot bound interior cells, force an enclosing ring.
    if (boundary_cycle.size < 3) or _has_unbounded_nonboundary(vor, types):
        points, types, boundary_cycle = _add_enclosing_boundary_ring(
            points, types, boundary_cycle
        )
        vor = Voronoi(points)  # recompute with the new ring

    N = points.shape[0]
    neighbors = build_voronoi_neighbors(vor, N)

    boundary_set = set(map(int, boundary_cycle.tolist()))
    bcyc = boundary_cycle.astype(np.int64, copy=True)

    # Enforce cycle neighbor constraint and mark deletions
    delete_list: list[int] = []
    for k in range(len(bcyc)):
        i = int(bcyc[k])
        prev_i = int(bcyc[(k - 1) % len(bcyc)])
        next_i = int(bcyc[(k + 1) % len(bcyc)])
        # keep only prev/next from boundary inside neighbor set
        neighbors[i] = (neighbors[i] - boundary_set) | {prev_i, next_i}
        # pruning: small interior angle
        v_prev = points[prev_i] - points[i]
        v_next = points[next_i] - points[i]
        n_prev = np.linalg.norm(v_prev)
        n_next = np.linalg.norm(v_next)
        if n_prev == 0.0 or n_next == 0.0:
            continue
        cosang = np.dot(v_prev, v_next) / (n_prev * n_next)
        angle = np.arccos(np.clip(cosang, -1.0, 1.0))
        if angle < np.pi / 2.0:
            delete_list.append(i)

    # Add mirrored boundary points between consecutive boundary nodes
    edges = np.stack([bcyc, np.roll(bcyc, -1)], axis=1)
    new_cells: list[tuple[int, int]] = []  # (insert_after_index, new_point_idx)
    for k in range(edges.shape[0]):
        a = int(edges[k, 0])
        b = int(edges[k, 1])
        shared = neighbors[a] & neighbors[b]
        shared_non_boundary = list(shared - boundary_set)
        if not shared_non_boundary:
            continue
        c = int(shared_non_boundary[0])
        ok, reflected = _reflect_point_across_segment(points[a], points[b], points[c])
        if not ok:
            continue
        # Append new boundary point
        points = np.vstack([points, reflected])
        types = np.append(types, 1)  # boundary
        new_idx = points.shape[0] - 1
        new_cells.append((k + 1, new_idx))
        boundary_set.add(new_idx)

    # Insert new boundary indices into the cycle in reverse order to keep offsets valid
    for ins_pos, idx in reversed(new_cells):
        bcyc = np.insert(bcyc, ins_pos, idx)

    # Handle deletions (sorted descending)
    for del_idx in sorted(set(delete_list), reverse=True):
        # Delete from arrays
        points = np.delete(points, del_idx, axis=0)
        types = np.delete(types, del_idx, axis=0)
        # Fix boundary cycle indices
        bcyc = bcyc[bcyc != del_idx]
        bcyc[bcyc > del_idx] -= 1

    # Rebuild Voronoi after edits
    vor = Voronoi(points)

    return points, types, bcyc, vor

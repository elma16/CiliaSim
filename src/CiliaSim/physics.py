from __future__ import annotations
import numpy as np
from numba import njit
from .params import Params, BOUNDARY, MULTICILIATED


@njit(cache=True, fastmath=True)
def _compute_edge_contribs(
    points: np.ndarray,
    types: np.ndarray,
    areas: np.ndarray,
    target_areas: np.ndarray,
    degree: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    target_spring_length: float,
    critical_length_delta: float,
    boundary_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-edge force contributions for scatter-add accumulation.

    Args:
        points: Array of shape (N, 2) of site coordinates.
        types: Array of shape (N,) of cell types.
        areas: Array of shape (N,) of current cell areas.
        target_areas: Array of shape (N,) of target cell areas.
        degree: Array of shape (N,) of node degrees.
        src: Array of shape (E,) of edge source indices.
        dst: Array of shape (E,) of edge destination indices.
        target_spring_length: Rest length for edge springs.
        critical_length_delta: Clamp for negative spring extension.
        boundary_factor: Spring scaling factor for boundary nodes.

    Returns:
        Tuple (contrib_to_dst, contrib_to_src) each of shape (E, 2) containing
        the per-edge force vectors to scatter-add to dst and src nodes.
    """
    E = src.shape[0]
    contrib_dst = np.zeros((E, 2), dtype=np.float64)
    contrib_src = np.zeros((E, 2), dtype=np.float64)

    for k in range(E):
        i = int(src[k])
        j = int(dst[k])
        dx0 = points[j, 0] - points[i, 0]
        dx1 = points[j, 1] - points[i, 1]
        dist = np.hypot(dx0, dx1)
        if dist == 0.0:
            continue  # skip degenerate
        ux = dx0 / dist
        uy = dx1 / dist

        # Springs from source i and source j (row-wise scaling)
        s_i = target_spring_length - dist
        if types[i] == BOUNDARY:
            s_i *= boundary_factor
        if s_i < -critical_length_delta:
            s_i = -critical_length_delta

        s_j = target_spring_length - dist
        if types[j] == BOUNDARY:
            s_j *= boundary_factor
        if s_j < -critical_length_delta:
            s_j = -critical_length_delta

        # Pressure split by degree
        p_i = 0.0
        if types[i] != BOUNDARY:
            deg_i = max(1, int(degree[i]))
            p_i = (target_areas[i] - areas[i]) / float(deg_i)
        p_j = 0.0
        if types[j] != BOUNDARY:
            deg_j = max(1, int(degree[j]))
            p_j = (target_areas[j] - areas[j]) / float(deg_j)

        # Source i → j
        f_ij = s_i + p_i
        contrib_dst[k, 0] = f_ij * ux
        contrib_dst[k, 1] = f_ij * uy

        # Source j → i (note sign)
        f_ji = s_j + p_j
        contrib_src[k, 0] = -f_ji * ux
        contrib_src[k, 1] = -f_ji * uy

    return contrib_dst, contrib_src


def accumulate_forces_numba(
    points: np.ndarray,
    types: np.ndarray,
    areas: np.ndarray,
    target_areas: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    cilia_forces: np.ndarray,
    flow_force: np.ndarray,
    params: Params,
) -> np.ndarray:
    """
    Assemble net node forces from edge and exogenous contributions.

    Args:
        points: Array of shape (N, 2) of site coordinates.
        types: Array of shape (N,) of cell types.
        areas: Array of shape (N,) of current cell areas.
        target_areas: Array of shape (N,) of target cell areas.
        src: Array of shape (E,) of edge source indices.
        dst: Array of shape (E,) of edge destination indices.
        cilia_forces: Array of shape (N, 2) with zeros for non-multiciliated.
        flow_force: Array of shape (2,) applied only to multiciliated cells.
        params: Simulation parameter bundle.

    Returns:
        Array of shape (N, 2) of net forces for each node.
    """
    N = points.shape[0]
    degree = np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)

    contrib_dst, contrib_src = _compute_edge_contribs(
        points,
        types,
        areas,
        target_areas,
        degree,
        src,
        dst,
        params.target_spring_length,
        params.critical_length_delta,
        params.boundary_spring_factor,
    )

    F = np.zeros((N, 2), dtype=np.float64)
    # Scatter-add
    np.add.at(F, dst, contrib_dst)
    np.add.at(F, src, contrib_src)

    # External forces on multiciliated cells
    mc_mask = types == MULTICILIATED
    F[mc_mask] += cilia_forces[mc_mask] + flow_force[None, :]

    return F

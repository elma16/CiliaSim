import numpy as np
from ciliasim.params import Params, BOUNDARY, MULTICILIATED
from ciliasim.geometry import voronoi_edges, cell_areas
from ciliasim.physics import accumulate_forces_numba
from ciliasim.utils import build_target_areas


def ref_dense(points, types, areas, target_areas, src, dst, params):
    N = points.shape[0]
    deg = np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)
    F = np.zeros((N, 2), float)
    # build symmetric adjacency from edges
    adj = np.zeros((N, N), int)
    adj[src, dst] = 1
    adj[dst, src] = 1
    for i in range(N):
        nbrs = np.where(adj[i] == 1)[0]
        for j in nbrs:
            diff = points[j] - points[i]
            dist = np.linalg.norm(diff)
            if dist == 0:
                continue
            u = diff / dist
            s_i = params.target_spring_length - dist
            if types[i] == BOUNDARY:
                s_i *= params.boundary_spring_factor
            s_i = max(s_i, -params.critical_length_delta)
            p_i = (
                0.0
                if types[i] == BOUNDARY
                else (target_areas[i] - areas[i]) / max(1, deg[i])
            )
            # contribution on j from i
            F[j] += (s_i + p_i) * u
    return F


def test_edge_forces_match_reference(triangle_points):
    P, types, _ = triangle_points
    params = Params()
    src, dst, vor = voronoi_edges(P)
    areas = cell_areas(P, types, vor)
    target_areas = build_target_areas(types, params.target_cell_area)
    cilia = np.zeros((P.shape[0], 2))
    flow = np.zeros(2)
    F_edge = accumulate_forces_numba(
        P, types, areas, target_areas, src, dst, cilia, flow, params
    )
    F_ref = ref_dense(P, types, areas, target_areas, src, dst, params)
    assert np.allclose(F_edge, F_ref, atol=1e-10)


def test_exogenous_forces_only_apply_to_multiciliated():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    types = np.array([MULTICILIATED, 0, BOUNDARY], dtype=int)
    areas = np.zeros(3, dtype=float)
    target_areas = np.zeros(3, dtype=float)
    src = np.array([], dtype=np.int32)
    dst = np.array([], dtype=np.int32)
    cilia = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    flow = np.array([0.5, -0.5], dtype=float)
    params = Params()

    F = accumulate_forces_numba(
        points, types, areas, target_areas, src, dst, cilia, flow, params
    )

    expected = np.zeros_like(cilia)
    expected[0] = cilia[0] + flow
    assert np.allclose(F, expected)

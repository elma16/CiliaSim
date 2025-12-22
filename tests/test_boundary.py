import numpy as np
from ciliasim.boundary import (
    _add_enclosing_boundary_ring,
    _has_unbounded_nonboundary,
    _reflect_point_across_segment,
    build_voronoi_neighbors,
    evaluate_boundary,
)
from scipy.spatial import Voronoi


def test_boundary_cycle_is_consistent(triangle_points):
    P, types, cyc = triangle_points
    P2, types2, cyc2, vor = evaluate_boundary(P, types, cyc)
    assert P2.shape[1] == 2
    assert types2.shape[0] == P2.shape[0]
    assert cyc2.ndim == 1
    # cycle indices valid
    assert cyc2.min() >= 0 and cyc2.max() < P2.shape[0]
    # boundary nodes are labeled boundary
    assert np.all(types2[cyc2] == 1)
    # non-boundary regions should prefer bounded regions
    v2 = Voronoi(P2)
    for i in np.where(types2 != 1)[0]:
        region = v2.regions[v2.point_region[i]]
        assert -1 not in region


def test_simulator_keeps_cells_bounded(triangle_points):
    from ciliasim.simulator import Simulator, State

    P, types, cyc = triangle_points
    sim = Simulator(State(P, types, cyc))
    for i in range(5):
        sim.step(i=i)  # default topology_every=1
        # assert bounded for non-boundary
        v = Voronoi(sim.points)
        for j in np.where(sim.types != 1)[0]:
            assert -1 not in v.regions[v.point_region[j]]


def test_reflect_point_across_segment_when_obtuse():
    a = np.array([0.0, 0.0])
    b = np.array([2.0, 0.0])
    c = np.array([1.0, 0.2])
    ok, reflected = _reflect_point_across_segment(a, b, c)
    assert ok is True
    assert np.allclose(reflected, [1.0, -0.2])

    ok2, reflected2 = _reflect_point_across_segment(a, b, np.array([1.0, 2.0]))
    assert ok2 is False
    assert np.allclose(reflected2, [1.0, 2.0])


def test_add_enclosing_boundary_ring_appends_cycle():
    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    types = np.array([0, 0], dtype=int)
    boundary_cycle = np.array([], dtype=int)
    points2, types2, cycle2 = _add_enclosing_boundary_ring(
        points, types, boundary_cycle, n=6, factor=3.0
    )
    assert points2.shape[0] == points.shape[0] + 6
    assert np.all(types2[-6:] == 1)
    assert np.array_equal(
        cycle2, np.arange(points.shape[0], points.shape[0] + 6, dtype=np.int64)
    )


def test_build_voronoi_neighbors_is_symmetric(triangle_points):
    P, _, _ = triangle_points
    vor = Voronoi(P)
    neighbors = build_voronoi_neighbors(vor, P.shape[0])
    for i, j in vor.ridge_points:
        assert int(j) in neighbors[int(i)]
        assert int(i) in neighbors[int(j)]


def test_has_unbounded_nonboundary_detects_unbounded():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    types = np.array([0, 0, 0], dtype=int)
    vor = Voronoi(points)
    assert _has_unbounded_nonboundary(vor, types)
    assert not _has_unbounded_nonboundary(vor, np.ones_like(types))

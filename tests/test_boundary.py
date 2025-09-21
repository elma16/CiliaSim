import numpy as np
from ciliasim.boundary import evaluate_boundary
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

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)

from ciliasim.boundary import _add_enclosing_boundary_ring
from ciliasim.plotting import PlotManager


def test_plot_manager_draws_polys_and_forces():
    points = np.array([[0.0, 0.0], [0.3, 0.1], [-0.2, 0.2]], dtype=float)
    types = np.array([0, 2, 0], dtype=int)
    boundary_cycle = np.array([], dtype=int)

    points2, types2, boundary_cycle2 = _add_enclosing_boundary_ring(
        points, types, boundary_cycle, n=8, factor=5.0
    )

    pm = PlotManager()
    pm.draw_tissue(points2, types2, boundary_cycle2, title="test")

    assert pm.boundary_scatter is not None
    assert len(pm.poly_basic.get_paths()) == np.sum(types2 == 0)
    assert len(pm.poly_multi.get_paths()) == np.sum(types2 == 2)

    F = np.zeros_like(points2)
    pm.draw_forces(points2, F)
    assert pm.quiv is not None

    pm.draw_forces(points2, np.ones_like(points2) * 0.1)

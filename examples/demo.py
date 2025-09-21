from __future__ import annotations
import time
import numpy as np
import matplotlib.pyplot as plt

from ciliasim.simulator import Simulator, State
from ciliasim.params import Params
from ciliasim.boundary import evaluate_boundary
from ciliasim.utils import build_target_areas, cilia_array_from_dict
from ciliasim.plotting import PlotManager
from ciliasim.geometry import edges_from_voronoi, cell_areas
from ciliasim.physics import accumulate_forces_numba
from scipy.spatial import Voronoi


def hex_spiral_layout(num_target: int, x: float, y: float) -> np.ndarray:
    """Replicate the legacy hex spiral (1 + 3R(R+1) points)."""
    R = int(np.floor(0.5 + np.sqrt(12 * num_target - 3) / 6.0))
    cx, cy = x / 2.0, y / 2.0
    pts = [(cx, cy)]
    for i in range(1, R + 1):
        for j in range(6 * i):
            angle = j * np.pi / (3.0 * i)
            if i % 2 == 0:
                angle += np.pi / (3.0 * i)
            pts.append((cx + i * np.cos(angle), cy + i * np.sin(angle)))
    return np.array(pts, dtype=np.float64)


def make_initial_state(x: int = 15, y: int = 15, center_only: bool = True) -> State:
    # Legacy did: num_cells = (x-1)*(y-1)
    num_cells = (x - 1) * (y - 1)
    P = hex_spiral_layout(num_cells, x, y)  # ~169 points for 15×15
    types = np.zeros(P.shape[0], dtype=np.int32)
    if center_only and P.size > 0:
        center = np.array([x / 2.0, y / 2.0])
        idx = int(np.argmin(np.sum((P - center) ** 2, axis=1)))
        types[idx] = 2  # multiciliated
    boundary_cycle = np.array([], dtype=np.int64)  # let boundary evaluation add a ring
    return State(points=P, types=types, boundary_cycle=boundary_cycle)


def main():
    x = y = 15
    params = Params()
    sim = Simulator(make_initial_state(x, y, center_only=True), params=params)

    # Ensure a valid boundary and fresh target areas before timing:
    sim.points, sim.types, sim.boundary_cycle, vor = evaluate_boundary(
        sim.points, sim.types, sim.boundary_cycle
    )
    sim.target_areas = build_target_areas(sim.types, sim.params.target_cell_area)

    t0 = time.perf_counter()
    sim.run(steps=1000, topology_every=1, progress=True)  # Phase 1: no cilia
    sim.set_uniform_cilia(direction=np.array([0.0, 1.0]), magnitude=0.4)
    sim.run(steps=5000, topology_every=1, progress=True)  # Phase 2
    t1 = time.perf_counter()
    print(
        f"Total time: {t1 - t0:.3f}s for 1000 + 5000 steps on {sim.points.shape[0]} cells."
    )

    # Final plot (block until the window is closed)
    pm = PlotManager()
    pm.draw_tissue(sim.points, sim.types, sim.boundary_cycle, title="Final tissue")
    vor = Voronoi(sim.points)  # one more build for the final force quiver
    src, dst = edges_from_voronoi(vor)
    areas = cell_areas(sim.points, sim.types, vor)
    cilia = cilia_array_from_dict(sim.points.shape[0], sim.cilia_dict)
    F = accumulate_forces_numba(
        sim.points,
        sim.types,
        areas,
        sim.target_areas,
        src,
        dst,
        cilia,
        sim.flow_force,
        sim.params,
    )
    pm.draw_forces(sim.points, F)
    plt.show(block=True)


if __name__ == "__main__":
    main()

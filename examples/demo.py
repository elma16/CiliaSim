from __future__ import annotations
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from ciliasim.simulator import Simulator, State
from ciliasim.params import Params
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


def mark_boundary_cells_kdtree(
    points: np.ndarray, x: int, y: int, num_edge_points: int = 50
) -> np.ndarray:
    """Mark boundary cells using KDTree edge detection (matches original method)."""
    types = np.zeros(len(points), dtype=np.int32)

    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    kd_tree = KDTree(points)

    # Create edge points
    top_edge = np.linspace([x_min, y_max], [x_max, y_max], num_edge_points)
    right_edge = np.linspace([x_max, y_max], [x_max, y_min], num_edge_points)
    bottom_edge = np.linspace([x_max, y_min], [x_min, y_min], num_edge_points)
    left_edge = np.linspace([x_min, y_min], [x_min, y_max], num_edge_points)

    boundary_indices = set()
    for edge in [top_edge, right_edge, bottom_edge, left_edge]:
        for comparison_point in edge:
            _, point_index = kd_tree.query(comparison_point)
            boundary_indices.add(point_index)
            types[point_index] = 1  # BOUNDARY

    return types, np.array(sorted(boundary_indices), dtype=np.int64)


def make_initial_state(x: int = 15, y: int = 15, center_only: bool = True) -> State:
    """Create initial state matching original setup."""
    num_cells = (x - 1) * (y - 1)
    points = hex_spiral_layout(num_cells, x, y)

    # Use KDTree-based boundary detection like original
    types, boundary_cycle = mark_boundary_cells_kdtree(points, x, y)

    if center_only and len(points) > 0:
        center = np.array([x / 2.0, y / 2.0])
        idx = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
        types[idx] = 2  # multiciliated

    return State(points=points, types=types, boundary_cycle=boundary_cycle)


def main():
    x = y = 15
    params = Params()

    # Create initial state with KDTree boundaries (matches original)
    sim = Simulator(make_initial_state(x, y, center_only=True), params=params)

    print(f"Starting simulation with {sim.points.shape[0]} cells")

    t0 = time.perf_counter()
    # Aggressive caching for maximum speed:
    # - topology_every=20: Update edges/boundary every 20 steps
    # - area_every=5: Recompute areas every 5 steps
    # - In between: use cached values (trades some accuracy for 3-4x speedup)
    sim.run(steps=1000, topology_every=20, area_every=5, progress=True)
    sim.set_uniform_cilia(direction=np.array([0.0, 1.0]), magnitude=0.4)
    sim.run(steps=5000, topology_every=20, area_every=5, progress=True)
    t1 = time.perf_counter()

    print(f"Total time: {t1 - t0:.3f}s for 6000 steps on {sim.points.shape[0]} cells")
    print(f"Average: {(t1-t0)/6000*1000:.2f}ms per step")

    # Final plot
    pm = PlotManager()
    pm.draw_tissue(sim.points, sim.types, sim.boundary_cycle, title="Final tissue")

    # Compute final forces for visualization
    vor = Voronoi(sim.points)
    src, dst = edges_from_voronoi(vor)
    areas = cell_areas(sim.points, sim.types, vor)
    F = accumulate_forces_numba(
        sim.points,
        sim.types,
        areas,
        sim.target_areas,
        src,
        dst,
        sim._cilia_forces,
        sim.flow_force,
        sim.params,
    )
    pm.draw_forces(sim.points, F)
    plt.show(block=True)


if __name__ == "__main__":
    main()

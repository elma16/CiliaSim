from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .params import Params
from .geometry import cell_areas, edges_from_voronoi
from .boundary import evaluate_boundary
from .physics import accumulate_forces_numba
from .utils import build_target_areas, cilia_array_from_dict

# Still need Voronoi for edges/areas even if topology unchanged
from tqdm import trange
from scipy.spatial import Voronoi


@dataclass
class State:
    points: np.ndarray  # (N,2)
    types: np.ndarray  # (N,)
    boundary_cycle: np.ndarray  # (B,)


class Simulator:
    def __init__(self, state: State, params: Params | None = None):
        self.params = params or Params()
        self.points = state.points.astype(np.float64, copy=True)
        self.types = state.types.astype(np.int32, copy=True)
        self.boundary_cycle = state.boundary_cycle.astype(np.int64, copy=True)
        self.flow_force = np.zeros(2, dtype=np.float64)
        self.cilia_dict: dict[int, np.ndarray] = {}
        self.target_areas = build_target_areas(self.types, self.params.target_cell_area)

    def set_uniform_cilia(self, direction: np.ndarray, magnitude: float):
        force = np.asarray(direction, dtype=np.float64) * float(magnitude)
        idx = np.where(self.types == 2)[0]
        for i in idx:
            self.cilia_dict[int(i)] = force.copy()

    def set_random_cilia(self, magnitude: float, seed: int = 42):
        rng = np.random.default_rng(seed)
        from .utils import random_cilia_forces

        vecs = random_cilia_forces(self.types, magnitude, rng)
        idx = np.where(self.types == 2)[0]
        for i in idx:
            self.cilia_dict[int(i)] = vecs[i]

    def set_flow(self, direction: np.ndarray, magnitude: float):
        self.flow_force = np.asarray(direction, dtype=np.float64) * float(magnitude)

    def step(self, topology_every: int = 1, i: int = 0):
        # Boundary maintenance every step (you can pass topology_every>1 to decimate)
        if i % topology_every == 0:
            self.points, self.types, self.boundary_cycle, vor = evaluate_boundary(
                self.points, self.types, self.boundary_cycle
            )
            # types may have changed → refresh targets
            self.target_areas = build_target_areas(
                self.types, self.params.target_cell_area
            )
        else:
            vor = Voronoi(self.points)

        # ⬇️ Reuse 'vor' instead of recomputing it
        src, dst = edges_from_voronoi(vor)
        areas = cell_areas(self.points, self.types, vor)

        cilia_forces = cilia_array_from_dict(self.points.shape[0], self.cilia_dict)
        F = accumulate_forces_numba(
            self.points,
            self.types,
            areas,
            self.target_areas,
            src,
            dst,
            cilia_forces,
            self.flow_force,
            self.params,
        )
        self.points = self.points + self.params.dt * F

    def run(self, steps: int, topology_every: int = 1, progress: bool = True):
        it = range(steps)
        if progress:
            it = trange(steps, desc="Sim")
        for i in it:
            self.step(topology_every=topology_every, i=i)
        return self.points.copy()

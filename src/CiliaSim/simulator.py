from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .params import Params
from .geometry import cell_areas, edges_from_voronoi
from .boundary import evaluate_boundary
from .physics import accumulate_forces_numba
from .utils import build_target_areas
from tqdm import trange
from scipy.spatial import Voronoi


@dataclass
class State:
    """
    Container for simulator state arrays.

    Attributes:
        points: Array of shape (N, 2) of site coordinates.
        types: Array of shape (N,) of cell types.
        boundary_cycle: Array of boundary indices in cyclic order.
    """

    points: np.ndarray  # (N,2)
    types: np.ndarray  # (N,)
    boundary_cycle: np.ndarray  # (B,)


class Simulator:
    """
    Evolve a Voronoi-based tissue model with cached topology updates.
    """

    def __init__(self, state: State, params: Params | None = None) -> None:
        """
        Initialize the simulator with a state and optional parameters.

        Args:
            state: State object with points, types, and boundary cycle arrays.
            params: Optional Params; defaults to Params() if not provided.
        """
        self.params = params or Params()
        self.points = state.points.astype(np.float64, copy=True)
        self.types = state.types.astype(np.int32, copy=True)
        self.boundary_cycle = state.boundary_cycle.astype(np.int64, copy=True)
        self.flow_force = np.zeros(2, dtype=np.float64)
        self.cilia_dict: dict[int, np.ndarray] = {}
        self.target_areas = build_target_areas(self.types, self.params.target_cell_area)

        # Pre-allocate cilia forces array
        self._cilia_forces = np.zeros((self.points.shape[0], 2), dtype=np.float64)

        # Cached topology data (edges don't change between topology updates)
        self._cached_src = None
        self._cached_dst = None
        self._cached_vor = None
        self._cached_areas = None

    def set_uniform_cilia(self, direction: np.ndarray, magnitude: float) -> None:
        """
        Apply the same cilia force vector to all multiciliated cells.

        Args:
            direction: 2D direction vector (not normalized internally).
            magnitude: Scalar magnitude to scale the direction.

        Notes:
            Multiciliated cells are identified by type value 2.
        """
        force = np.asarray(direction, dtype=np.float64) * float(magnitude)
        idx = np.where(self.types == 2)[0]
        for i in idx:
            self.cilia_dict[int(i)] = force.copy()
        self._update_cilia_forces()

    def set_random_cilia(self, magnitude: float, seed: int = 42) -> None:
        """
        Assign random cilia directions to multiciliated cells.

        Args:
            magnitude: Magnitude of each cilia force vector.
            seed: Seed for the RNG for reproducible directions.

        Notes:
            Multiciliated cells are identified by type value 2.
        """
        rng = np.random.default_rng(seed)
        from .utils import random_cilia_forces

        vecs = random_cilia_forces(self.types, magnitude, rng)
        idx = np.where(self.types == 2)[0]
        for i in idx:
            self.cilia_dict[int(i)] = vecs[i]
        self._update_cilia_forces()

    def set_flow(self, direction: np.ndarray, magnitude: float) -> None:
        """
        Set the global flow force applied to multiciliated cells.

        Args:
            direction: 2D direction vector (not normalized internally).
            magnitude: Scalar magnitude to scale the direction.
        """
        self.flow_force = np.asarray(direction, dtype=np.float64) * float(magnitude)

    def _update_cilia_forces(self) -> None:
        """
        Rebuild the cached per-cell cilia forces array.

        The array is resized to match the current number of points and filled
        with zeros before applying entries from cilia_dict. Indices outside
        the current range are ignored.
        """
        n = self.points.shape[0]
        if self._cilia_forces.shape[0] != n:
            self._cilia_forces = np.zeros((n, 2), dtype=np.float64)
        else:
            self._cilia_forces.fill(0.0)
        for k, v in self.cilia_dict.items():
            if k < n:
                self._cilia_forces[k] = v

    def step(self, topology_every: int = 1, area_every: int = 1, i: int = 0) -> None:
        """
        Step the simulation.

        Args:
            topology_every: Update topology (boundary + edges) every N steps
            area_every: Recompute areas every N steps (must divide topology_every)
            i: Current iteration number

        Notes:
            On topology updates, boundary handling, Voronoi edges, and areas are
            recomputed and cached. On area-only updates, topology is reused and
            areas are recomputed from a fresh Voronoi. Otherwise cached areas
            are reused for speed.
        """
        # Full topology update: boundary, Voronoi, edges, areas
        if i % topology_every == 0:
            self.points, self.types, self.boundary_cycle, vor = evaluate_boundary(
                self.points, self.types, self.boundary_cycle
            )
            self.target_areas = build_target_areas(
                self.types, self.params.target_cell_area
            )
            # Cache edges (topology-dependent)
            self._cached_src, self._cached_dst = edges_from_voronoi(vor)
            self._cached_vor = vor
            # Compute areas
            self._cached_areas = cell_areas(self.points, self.types, vor)

            # Update cilia forces if cell count changed
            if self._cilia_forces.shape[0] != self.points.shape[0]:
                self._update_cilia_forces()

        # Partial update: recompute areas only (points moved, topology unchanged)
        elif i % area_every == 0:
            vor = Voronoi(self.points)
            self._cached_areas = cell_areas(self.points, self.types, vor)

        # No update: use cached areas (approximation - trades accuracy for speed)
        # This is valid for small timesteps where points don't move much

        # Compute forces using cached data
        F = accumulate_forces_numba(
            self.points,
            self.types,
            self._cached_areas,
            self.target_areas,
            self._cached_src,
            self._cached_dst,
            self._cilia_forces,
            self.flow_force,
            self.params,
        )

        # Update positions in-place
        self.points += self.params.dt * F

    def run(
        self,
        steps: int,
        topology_every: int = 1,
        area_every: int = 1,
        progress: bool = True,
    ) -> np.ndarray:
        """
        Run simulation.

        Args:
            steps: Number of steps
            topology_every: Update topology every N steps (expensive)
            area_every: Update areas every N steps (moderate cost)
            progress: Show progress bar

        Returns:
            Copy of the final points array after all steps.
        """
        it = range(steps)
        if progress:
            it = trange(steps, desc="Sim")
        for i in it:
            self.step(topology_every=topology_every, area_every=area_every, i=i)
        return self.points.copy()

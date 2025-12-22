import numpy as np

from ciliasim.params import BOUNDARY, MULTICILIATED, Params
from ciliasim.simulator import Simulator, State


def _simple_state():
    points = np.array([[0.0, 0.0], [1.0, 0.5], [0.5, -0.5]], dtype=float)
    types = np.array([MULTICILIATED, 0, BOUNDARY], dtype=int)
    boundary_cycle = np.array([2], dtype=int)
    return State(points=points, types=types, boundary_cycle=boundary_cycle)


def test_set_uniform_cilia_updates_cache():
    sim = Simulator(_simple_state())
    sim.set_uniform_cilia(direction=np.array([1.0, 0.0]), magnitude=2.0)
    assert np.allclose(sim._cilia_forces[0], [2.0, 0.0])
    assert np.all(sim._cilia_forces[1:] == 0.0)


def test_set_random_cilia_sets_only_multiciliated():
    sim = Simulator(_simple_state())
    sim.set_random_cilia(magnitude=1.5, seed=0)
    assert np.all(sim._cilia_forces[1:] == 0.0)
    assert np.isclose(np.linalg.norm(sim._cilia_forces[0]), 1.5)


def test_set_flow_sets_force():
    sim = Simulator(_simple_state())
    sim.set_flow(direction=np.array([0.0, -1.0]), magnitude=0.25)
    assert np.allclose(sim.flow_force, [0.0, -0.25])


def test_run_returns_copy(triangle_points):
    P, types, cyc = triangle_points
    sim = Simulator(State(P, types, cyc), params=Params(dt=0.0))
    out = sim.run(steps=1, progress=False)
    assert out.shape == sim.points.shape
    assert not np.shares_memory(out, sim.points)

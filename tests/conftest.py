import numpy as np
import pytest


@pytest.fixture
def triangle_points():
    # Simple acute triangle and a nearby point to make Voronoi bounded
    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8], [0.5, -0.8]], dtype=float)
    types = np.array([0, 0, 0, 1], dtype=int)  # last is boundary
    boundary_cycle = np.array([3], dtype=int)
    return P, types, boundary_cycle

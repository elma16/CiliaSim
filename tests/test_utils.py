import numpy as np

from ciliasim.params import MULTICILIATED
from ciliasim.utils import (
    build_target_areas,
    cilia_array_from_dict,
    normalize_rows,
    random_cilia_forces,
)


def test_normalize_rows_handles_zero_row():
    vecs = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=float)
    out = normalize_rows(vecs)
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.all(out[1] == 0.0)
    assert np.isfinite(out).all()


def test_build_target_areas_marks_boundary_zero():
    types = np.array([0, 1, 2, 1], dtype=int)
    target = 2.5
    out = build_target_areas(types, target)
    assert np.all(out[types == 1] == 0.0)
    assert np.allclose(out[types != 1], target)


def test_cilia_array_from_dict_accepts_string_keys():
    cilia_dict = {"1": [1.0, -1.0], 3: np.array([0.5, 0.25])}
    out = cilia_array_from_dict(5, cilia_dict)
    assert out.shape == (5, 2)
    assert np.allclose(out[1], [1.0, -1.0])
    assert np.allclose(out[3], [0.5, 0.25])
    assert np.all(out[[0, 2, 4]] == 0.0)


def test_random_cilia_forces_only_multiciliated():
    types = np.array([0, MULTICILIATED, 1, MULTICILIATED], dtype=int)
    rng = np.random.default_rng(0)
    mag = 2.0
    vecs = random_cilia_forces(types, mag, rng)
    assert np.all(vecs[types != MULTICILIATED] == 0.0)
    norms = np.linalg.norm(vecs[types == MULTICILIATED], axis=1)
    assert np.allclose(norms, mag)

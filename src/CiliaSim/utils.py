from __future__ import annotations
import numpy as np
from .params import MULTICILIATED


def normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def build_target_areas(types: np.ndarray, target_cell_area: float) -> np.ndarray:
    ta = np.zeros_like(types, dtype=np.float64)
    ta[types != 1] = target_cell_area
    return ta


def cilia_array_from_dict(
    n_cells: int, cilia_dict: dict[int, np.ndarray]
) -> np.ndarray:
    out = np.zeros((n_cells, 2), dtype=np.float64)
    for k, v in cilia_dict.items():
        out[int(k)] = np.asarray(v, dtype=np.float64)
    return out


def random_cilia_forces(
    types: np.ndarray, magnitude: float, rng: np.random.Generator
) -> np.ndarray:
    idx = np.where(types == MULTICILIATED)[0]
    dirs = rng.uniform(-1.0, 1.0, size=(idx.size, 2))
    dirs = normalize_rows(dirs)
    vecs = np.zeros((types.shape[0], 2), dtype=np.float64)
    vecs[idx] = dirs * magnitude
    return vecs

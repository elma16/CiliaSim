from __future__ import annotations
from typing import Mapping, Sequence

import numpy as np
from .params import MULTICILIATED


def normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize each row vector to unit length with numerical stability.

    Args:
        v: Array of shape (N, 2) or (N, D) of row vectors.
        eps: Minimum norm used to avoid division by zero.

    Returns:
        Array of the same shape as v with rows scaled to unit length where
        possible. Zero rows remain zero.
    """
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def build_target_areas(types: np.ndarray, target_cell_area: float) -> np.ndarray:
    """
    Build the per-cell target area array.

    Args:
        types: Array of shape (N,) of cell types; boundary cells are type 1.
        target_cell_area: Target area assigned to non-boundary cells.

    Returns:
        Array of shape (N,) where non-boundary cells get target_cell_area and
        boundary cells get 0.0.
    """
    ta = np.zeros_like(types, dtype=np.float64)
    ta[types != 1] = target_cell_area
    return ta


def cilia_array_from_dict(
    n_cells: int, cilia_dict: Mapping[int | str, np.ndarray | Sequence[float]]
) -> np.ndarray:
    """
    Convert a sparse cilia dictionary to a dense per-cell array.

    Args:
        n_cells: Total number of cells in the system.
        cilia_dict: Mapping from cell index (int or int-like string) to a
            2D force vector.

    Returns:
        Array of shape (n_cells, 2) with zeros for cells without cilia forces.
    """
    out = np.zeros((n_cells, 2), dtype=np.float64)
    for k, v in cilia_dict.items():
        out[int(k)] = np.asarray(v, dtype=np.float64)
    return out


def random_cilia_forces(
    types: np.ndarray, magnitude: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate random cilia force vectors for multiciliated cells only.

    Args:
        types: Array of shape (N,) of cell types.
        magnitude: Magnitude of each force vector.
        rng: NumPy random number generator used for direction sampling.

    Returns:
        Array of shape (N, 2) where multiciliated cells have random unit
        directions scaled by magnitude and all other rows are zero.
    """
    idx = np.where(types == MULTICILIATED)[0]
    dirs = rng.uniform(-1.0, 1.0, size=(idx.size, 2))
    dirs = normalize_rows(dirs)
    vecs = np.zeros((types.shape[0], 2), dtype=np.float64)
    vecs[idx] = dirs * magnitude
    return vecs

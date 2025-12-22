from __future__ import annotations
import json
import numpy as np
from typing import Any


def save_simulation(
    path: str,
    x: int,
    y: int,
    cilia_density: float,
    types: np.ndarray,
    target_areas: np.ndarray,
    cell_states: dict[int, list[list[float]]],
    force_states: dict[int, dict[int, list[float]]],
    net_energy: np.ndarray | list[float] = (),
) -> None:
    """
    Serialize simulation outputs to a JSON file.

    Args:
        path: Output file path to write.
        x: Grid width parameter saved for metadata.
        y: Grid height parameter saved for metadata.
        cilia_density: Fraction of cells that are multiciliated.
        types: Array of shape (N,) of cell types.
        target_areas: Array of shape (N,) of target cell areas.
        cell_states: Mapping from time index to a list of [x, y] positions.
        force_states: Mapping from time index to per-cell force vectors.
        net_energy: Sequence of energy values per time step.
    """
    data: dict[str, Any] = {
        "parameters": {"x": x, "y": y, "cilia_density": cilia_density},
        "cell_types": types.tolist(),
        "target_areas": target_areas.tolist(),
        "cell_states": {str(k): v for k, v in cell_states.items()},
        "force_states": {str(k): v for k, v in force_states.items()},
        "net_energy": list(net_energy),
    }
    with open(path, "w") as f:
        json.dump(data, f)

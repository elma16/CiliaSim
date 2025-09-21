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
):
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

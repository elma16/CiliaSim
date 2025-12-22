import json

import numpy as np

from ciliasim.io import save_simulation


def test_save_simulation_writes_expected_json(tmp_path):
    path = tmp_path / "sim.json"
    types = np.array([0, 1, 2], dtype=int)
    target_areas = np.array([1.0, 0.0, 1.2], dtype=float)
    cell_states = {0: [[0.1, 0.2], [0.2, 0.3]], 2: [[1.0, 1.1]]}
    force_states = {0: {0: [1.0, 0.0]}, 2: {1: [0.0, -1.0]}}
    net_energy = [0.5, 0.6]

    save_simulation(
        str(path),
        x=2,
        y=3,
        cilia_density=0.25,
        types=types,
        target_areas=target_areas,
        cell_states=cell_states,
        force_states=force_states,
        net_energy=net_energy,
    )

    data = json.loads(path.read_text())
    assert data["parameters"] == {"x": 2, "y": 3, "cilia_density": 0.25}
    assert data["cell_types"] == types.tolist()
    assert data["target_areas"] == target_areas.tolist()
    assert data["cell_states"]["0"][0] == [0.1, 0.2]
    assert "2" in data["cell_states"]
    assert "0" in data["force_states"]
    assert data["net_energy"] == net_energy

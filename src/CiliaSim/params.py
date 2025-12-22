from __future__ import annotations
from dataclasses import dataclass
import numpy as np

BOUNDARY = 1
MULTICILIATED = 2


@dataclass(frozen=True)
class Params:
    """
    Immutable simulation parameters for force and integration settings.

    Attributes:
        target_spring_length: Rest length for edge springs.
        critical_length_delta: Clamp on negative spring extension.
        target_cell_area: Desired Voronoi area for non-boundary cells.
        boundary_spring_factor: Spring scaling for boundary nodes.
        dt: Time step for explicit Euler integration.
    """

    target_spring_length: float = 1.0
    critical_length_delta: float = 0.2
    target_cell_area: float = float(np.sqrt(3.0) / 2.0)
    boundary_spring_factor: float = 0.1
    dt: float = 9.5e-3

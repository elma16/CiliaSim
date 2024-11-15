import numpy as np
from numba import njit

__all__ = [
    "calculate_force_matrix",
    "calculate_boundary_reflection",
    "hexagonal_grid_layout",
]


@njit(cache=True, fastmath=True)
def polygon_area(vertices: np.ndarray) -> float:
    """Optimized polygon area calculation using Shoelace formula"""
    x = vertices[:, 0]
    y = vertices[:, 1]
    # Preallocate arrays and use direct indexing
    n = len(x)
    shifted_x = np.empty(n)
    shifted_y = np.empty(n)
    shifted_x[:-1] = x[1:]
    shifted_x[-1] = x[0]
    shifted_y[:-1] = y[1:]
    shifted_y[-1] = y[0]
    return 0.5 * np.abs(np.sum(x * shifted_y - y * shifted_x))


@njit(cache=True, fastmath=True)
def polygon_perimeter(vertices: np.ndarray) -> float:
    """Optimized polygon perimeter calculation"""
    n = len(vertices)
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = vertices[j, 0] - vertices[i, 0]
        dy = vertices[j, 1] - vertices[i, 1]
        total += np.sqrt(dx * dx + dy * dy)
    return total


@njit(cache=True)
def calculate_force_matrix(
    num_cells: int,
    target_spring_length: float,
    critical_length_delta: float,
    cell_points: np.ndarray,
    cell_types: np.ndarray,
    target_areas: np.ndarray,
    voronoi_vertices: list,
    adjacency_matrix: np.ndarray,
):
    spring_matrix = np.zeros((num_cells, num_cells), dtype=np.float64)
    pressure_matrix = np.zeros((num_cells, num_cells), dtype=np.float64)
    distance_matrix = np.zeros((num_cells, num_cells), dtype=np.float64)
    unit_vector_matrix = np.zeros((num_cells, num_cells, 2), dtype=np.float64)

    for i in range(num_cells):
        neighbours = np.nonzero(adjacency_matrix[i])[0]
        if neighbours.size == 0:
            continue

        differences = cell_points[neighbours] - cell_points[i]
        distances = np.sqrt((differences**2).sum(axis=1))
        unit_vectors = differences / distances[:, None]

        distance_matrix[i, neighbours] = distances
        unit_vector_matrix[i, neighbours] = unit_vectors

        spring_matrix[i, neighbours] = target_spring_length - distances

        if cell_types[i] != 1:
            area = polygon_area(voronoi_vertices[i])
            split_area_difference = (target_areas[i] - area) / len(neighbours)
            pressure_matrix[i, neighbours] = split_area_difference
            pressure_matrix[neighbours, i] = split_area_difference

    spring_matrix[cell_types == 1] *= 0.1
    np.clip(spring_matrix, -critical_length_delta, None, out=spring_matrix)
    force_matrix = (spring_matrix + pressure_matrix)[..., None] * unit_vector_matrix

    return force_matrix, distance_matrix


@njit(cache=True)
def calculate_boundary_reflection(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    ac = c - a
    bc = c - b
    ac_norm = np.linalg.norm(ac)
    bc_norm = np.linalg.norm(bc)
    ac_dot_bc = np.dot(ac, bc)

    angle_cos = ac_dot_bc / (ac_norm * bc_norm)
    if angle_cos < 0:  # Equivalent to angle > π/2, avoids costly arccos
        edge_vector = b - a
        edge_norm = np.linalg.norm(edge_vector)
        edge_unit_vector = edge_vector / edge_norm

        projection_length = np.dot(ac, edge_unit_vector)
        projection_vector = projection_length * edge_unit_vector

        reflected_point = 2 * (a + projection_vector) - c
        return True, reflected_point

    return False, np.empty_like(a)


@njit(cache=True)
def hexagonal_grid_layout(num_cells: int, x: int, y: int):
    num_rings = int(np.floor(1 / 2 + np.sqrt(12 * num_cells - 3) / 6))

    points = []
    cx = x / 2
    cy = y / 2

    points.append((cx, cy))

    for i in range(1, num_rings + 1):
        for j in range(6 * i):
            angle = j * np.pi / (3 * i)
            if i % 2 == 0:
                angle += np.pi / (3 * i)
            x = cx + i * np.cos(angle)
            y = cy + i * np.sin(angle)
            points.append((x, y))

    return points

import numpy as np
from ciliasim.geometry import (
    cell_areas,
    edges_from_voronoi,
    polygon_area,
    polygon_perimeter,
    voronoi_edges,
)


def test_polygon_area_perimeter_square():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    assert np.isclose(polygon_area(sq), 1.0)
    assert np.isclose(polygon_perimeter(sq), 4.0)


def test_voronoi_edges_nonempty(triangle_points):
    P, types, _ = triangle_points
    src, dst, vor = voronoi_edges(P)
    assert src.shape == dst.shape
    assert src.size > 0
    areas = cell_areas(P, types, vor)
    assert areas.shape[0] == P.shape[0]
    # boundary area zero by convention
    assert areas[types == 1].sum() == 0.0


def test_edges_from_voronoi_unique(triangle_points):
    P, _, _ = triangle_points
    src, dst, vor = voronoi_edges(P)
    src2, dst2 = edges_from_voronoi(vor)
    pairs = np.sort(np.stack([src, dst], axis=1), axis=1)
    pairs2 = np.sort(np.stack([src2, dst2], axis=1), axis=1)
    pairs = np.unique(pairs, axis=0)
    pairs2 = np.unique(pairs2, axis=0)
    assert np.array_equal(pairs, pairs2)

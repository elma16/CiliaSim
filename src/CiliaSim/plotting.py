from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi


class PlotManager:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.poly_basic = PolyCollection(
            [], facecolors="lightgrey", edgecolors="k", alpha=0.6
        )
        self.poly_multi = PolyCollection(
            [], facecolors="orange", edgecolors="k", alpha=0.6
        )
        self.boundary_scatter = None
        self.quiv = None
        self.ax.add_collection(self.poly_basic)
        self.ax.add_collection(self.poly_multi)

    def _polys_by_type(self, points: np.ndarray, types: np.ndarray):
        vor = Voronoi(points)
        basic, multi = [], []
        for i in range(points.shape[0]):
            region = vor.regions[vor.point_region[i]]
            if len(region) == 0 or -1 in region:
                continue
            poly = vor.vertices[region]
            if types[i] == 2:
                multi.append(poly)
            elif types[i] == 0:
                basic.append(poly)
        return basic, multi

    def draw_tissue(
        self,
        points: np.ndarray,
        types: np.ndarray,
        boundary_idx: np.ndarray,
        title: str = "",
    ):
        basic, multi = self._polys_by_type(points, types)
        self.poly_basic.set_verts(basic)
        self.poly_multi.set_verts(multi)
        bpts = points[boundary_idx]
        if self.boundary_scatter is None:
            self.boundary_scatter = self.ax.scatter(
                bpts[:, 0], bpts[:, 1], s=20, color="green"
            )
        else:
            self.boundary_scatter.set_offsets(bpts)
        self.ax.set_title(title)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()

    def draw_forces(self, points: np.ndarray, F: np.ndarray):
        if self.quiv is None:
            self.quiv = self.ax.quiver(
                points[:, 0],
                points[:, 1],
                F[:, 0],
                F[:, 1],
                angles="xy",
                scale_units="xy",
                scale=0.1,
            )
        else:
            self.quiv.set_offsets(points)
            self.quiv.set_UVC(F[:, 0], F[:, 1])
        self.fig.canvas.draw_idle()

"""Patches: the unit a trained Sky2Trail model works on.

A patch has two squares around the same centre:

- the **core** (default 10 m), the area the model predicts. Stems and generated points
  belong to exactly one core, so cores tile a scene without gaps or double counting;
- the **input window** (core plus a margin on every side, default 6 m -> 22 m), the
  aerial lidar the model sees. The margin holds the crowns of trees near the core edge,
  which are the main evidence for their stems.

Cores are half-open squares [xmin, xmax) x [ymin, ymax), so a stem on a shared edge
goes to one patch only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import PointCloud


@dataclass(frozen=True)
class Patch:
    id: str
    cx: float
    cy: float
    core_size: float
    margin: float

    @property
    def core_bounds(self) -> tuple[float, float, float, float]:
        h = self.core_size / 2
        return self.cx - h, self.cy - h, self.cx + h, self.cy + h

    @property
    def input_bounds(self) -> tuple[float, float, float, float]:
        h = self.core_size / 2 + self.margin
        return self.cx - h, self.cy - h, self.cx + h, self.cy + h

    @property
    def input_size(self) -> float:
        return self.core_size + 2 * self.margin

    def in_core(self, x, y) -> np.ndarray:
        x0, y0, x1, y1 = self.core_bounds
        x, y = np.asarray(x), np.asarray(y)
        return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)

    def in_input(self, x, y) -> np.ndarray:
        x0, y0, x1, y1 = self.input_bounds
        x, y = np.asarray(x), np.asarray(y)
        return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)

    def corners(self) -> np.ndarray:
        x0, y0, x1, y1 = self.core_bounds
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _make(cxs, cys, core_size, margin, prefix) -> list[Patch]:
    return [Patch(f"{prefix}r{i:03d}c{j:03d}", float(cx), float(cy), core_size, margin)
            for i, cy in enumerate(cys) for j, cx in enumerate(cxs)]


def tile_patches(bounds, core_size: float = 10.0, margin: float = 6.0,
                 prefix: str = "") -> list[Patch]:
    """Cores edge to edge over a scene, starting at its lower-left corner.

    Cores along the upper and right edges may stick out of the scene; their input
    windows then have less data, which `coverage` reports.
    """
    xmin, ymin, xmax, ymax = bounds
    nx = max(1, int(np.ceil((xmax - xmin) / core_size - 1e-9)))
    ny = max(1, int(np.ceil((ymax - ymin) / core_size - 1e-9)))
    cxs = xmin + core_size * (np.arange(nx) + 0.5)
    cys = ymin + core_size * (np.arange(ny) + 0.5)
    return _make(cxs, cys, core_size, margin, prefix)


def plot_patches(cx: float, cy: float, shape: str, size: float, core_size: float = 10.0,
                 margin: float = 6.0, prefix: str = "") -> list[Patch]:
    """Cores lying completely inside a field plot, so every core cell has ground truth.

    shape: "circle" (size = diameter, e.g. the USGS Sierra plots, 20 m) or
    "square" (size = side, e.g. NEON plots, 20 or 40 m). Tries a core centred on the
    plot centre and a core corner on the plot centre, and keeps whichever fits more.
    """
    if shape not in ("circle", "square"):
        raise ValueError(f"shape must be 'circle' or 'square', not {shape!r}")
    n = int(np.ceil(size / core_size)) + 2
    best: list[Patch] = []
    for offset in (0.0, core_size / 2):
        k = np.arange(-n, n + 1)
        centres = k * core_size + offset
        cand = _make(cx + centres, cy + centres, core_size, margin, prefix)
        inside = []
        for p in cand:
            d = p.corners() - [cx, cy]
            if shape == "circle":
                ok = np.all(np.hypot(d[:, 0], d[:, 1]) <= size / 2 + 1e-9)
            else:
                ok = np.all(np.abs(d) <= size / 2 + 1e-9)
            if ok:
                inside.append(p)
        if len(inside) > len(best):
            best = inside
    return best


def cut_points(pc: PointCloud, patch: Patch) -> PointCloud:
    """Aerial points inside the patch's input window (world coordinates kept)."""
    return pc.subset(patch.in_input(pc.x, pc.y))


def assign_to_cores(x, y, patches: list[Patch]) -> np.ndarray:
    """Index of the patch whose core holds each (x, y); -1 if none."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    owner = np.full(len(x), -1, dtype=int)
    for i, p in enumerate(patches):
        hit = p.in_core(x, y) & (owner < 0)
        owner[hit] = i
    return owner


def coverage(pc: PointCloud, patch: Patch, cell: float = 1.0) -> float:
    """Share of input-window cells (cell x cell m) holding at least one aerial point."""
    x0, y0, _, _ = patch.input_bounds
    n = int(round(patch.input_size / cell))
    m = patch.in_input(pc.x, pc.y)
    i = np.clip(((pc.y[m] - y0) / cell).astype(int), 0, n - 1)
    j = np.clip(((pc.x[m] - x0) / cell).astype(int), 0, n - 1)
    filled = np.zeros((n, n), bool)
    filled[i, j] = True
    return float(filled.mean())

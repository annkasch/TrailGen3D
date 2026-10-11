"""North-up raster grids aligned to a common origin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Grid:
    """A north-up grid: row 0 is the northern edge, column 0 the western edge."""

    xmin: float
    ymax: float
    resolution: float
    nrows: int
    ncols: int

    @classmethod
    def from_bounds(cls, xmin, ymin, xmax, ymax, resolution) -> "Grid":
        # Snap the origin to a multiple of the resolution so grids line up.
        x0 = np.floor(xmin / resolution) * resolution
        y1 = np.ceil(ymax / resolution) * resolution
        ncols = int(np.ceil((xmax - x0) / resolution)) or 1
        nrows = int(np.ceil((y1 - ymin) / resolution)) or 1
        return cls(float(x0), float(y1), float(resolution), nrows, ncols)

    @property
    def shape(self) -> tuple[int, int]:
        return self.nrows, self.ncols

    @property
    def ymin(self) -> float:
        return self.ymax - self.nrows * self.resolution

    @property
    def xmax(self) -> float:
        return self.xmin + self.ncols * self.resolution

    @property
    def transform(self) -> tuple[float, ...]:
        """GDAL-style affine (a, b, c, d, e, f) for x = a*col + b*row + c, y = d*col + e*row + f."""
        return (self.resolution, 0.0, self.xmin, 0.0, -self.resolution, self.ymax)

    def resample_like(self, other_res: float) -> "Grid":
        return Grid.from_bounds(self.xmin, self.ymin, self.xmax, self.ymax, other_res)

    def index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row, column and an in-bounds mask for each point."""
        col = np.floor((x - self.xmin) / self.resolution).astype(np.int64)
        row = np.floor((self.ymax - y) / self.resolution).astype(np.int64)
        ok = (row >= 0) & (row < self.nrows) & (col >= 0) & (col < self.ncols)
        return row, col, ok

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        cols = self.xmin + (np.arange(self.ncols) + 0.5) * self.resolution
        rows = self.ymax - (np.arange(self.nrows) + 0.5) * self.resolution
        return np.meshgrid(cols, rows)

    def fractional_index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Continuous row/col coordinates of points, cell centres at integer values."""
        col = (x - self.xmin) / self.resolution - 0.5
        row = (self.ymax - y) / self.resolution - 0.5
        return row, col

    def sample(self, raster: np.ndarray, x: np.ndarray, y: np.ndarray, order: int = 1) -> np.ndarray:
        """Bilinear (order=1) sample of a raster on this grid at point positions."""
        row, col = self.fractional_index(x, y)
        return ndimage.map_coordinates(raster, [row, col], order=order, mode="nearest")


def bin_stat(grid: Grid, x, y, values, stat: str = "mean", fill=np.nan) -> np.ndarray:
    """Per-cell statistic ('mean', 'max', 'min', 'count') of point values."""
    row, col, ok = grid.index(x, y)
    flat = row[ok] * grid.ncols + col[ok]
    n = grid.nrows * grid.ncols
    if stat == "count":
        return np.bincount(flat, minlength=n).reshape(grid.shape).astype(float)
    v = np.asarray(values, dtype=float)[ok]
    if stat == "mean":
        s = np.bincount(flat, weights=v, minlength=n)
        c = np.bincount(flat, minlength=n)
        out = np.full(n, fill, dtype=float)
        has = c > 0
        out[has] = s[has] / c[has]
    elif stat in ("max", "min"):
        out = np.full(n, -np.inf if stat == "max" else np.inf)
        (np.maximum if stat == "max" else np.minimum).at(out, flat, v)
        out[~np.isfinite(out)] = fill
    else:
        raise ValueError(f"Unknown stat {stat!r}")
    return out.reshape(grid.shape)


def resample(src_grid: Grid, raster: np.ndarray, dst_grid: Grid, order: int = 1) -> np.ndarray:
    """Resample a raster onto another grid covering the same area."""
    xs, ys = dst_grid.cell_centers()
    return src_grid.sample(raster, xs.ravel(), ys.ravel(), order=order).reshape(dst_grid.shape)

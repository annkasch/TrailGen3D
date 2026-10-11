"""Ground model (DEM), height above ground, slope and roughness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import interpolate, ndimage

from .config import GroundCfg
from .grid import Grid, bin_stat
from .io import PointCloud


@dataclass
class Terrain:
    grid: Grid
    dem: np.ndarray            # ground elevation per cell, gaps filled
    ground_count: np.ndarray   # ground returns per cell (0 = interpolated)
    used_fallback: bool        # True if the delivered ground class was not usable

    def height_above_ground(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return z - self.grid.sample(self.dem, x, y)

    def slope_deg(self) -> np.ndarray:
        gy, gx = np.gradient(self.dem, self.grid.resolution)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    def roughness(self, window_m: float = 3.0) -> np.ndarray:
        """Std. dev. of the DEM around its local mean surface, over a window."""
        size = max(3, int(round(window_m / self.grid.resolution)) | 1)
        smooth = ndimage.uniform_filter(self.dem, size=size, mode="nearest")
        resid = self.dem - smooth
        return np.sqrt(ndimage.uniform_filter(resid**2, size=size, mode="nearest"))

    def ground_support(self, radius_m: float = 2.0) -> np.ndarray:
        """Share of nearby cells backed by real ground returns (1 = fully measured)."""
        size = max(1, int(round(2 * radius_m / self.grid.resolution)) | 1)
        measured = (self.ground_count > 0).astype(float)
        return ndimage.uniform_filter(measured, size=size, mode="nearest")


def _fill_gaps(grid: Grid, raster: np.ndarray) -> np.ndarray:
    """Linear interpolation inside the data hull, nearest neighbour outside it."""
    known = np.isfinite(raster)
    if known.all():
        return raster
    if not known.any():
        raise ValueError("No ground cells to interpolate from")
    xs, ys = grid.cell_centers()
    pts = np.column_stack([xs[known], ys[known]])
    vals = raster[known]
    target = np.column_stack([xs[~known], ys[~known]])
    filled = raster.copy()
    lin = interpolate.griddata(pts, vals, target, method="linear") if known.sum() >= 3 else None
    if lin is None:
        lin = np.full(len(target), np.nan)
    missing = ~np.isfinite(lin)
    if missing.any():
        lin[missing] = interpolate.griddata(pts, vals, target[missing], method="nearest")
    filled[~known] = lin
    return filled


def _lowest_point_ground(pc: PointCloud, grid: Grid, max_rise: float = 1.0) -> np.ndarray:
    """Fallback ground filter: lowest point per cell, dropping cells that sit well above
    their neighbourhood (canopy cells with no ground return underneath)."""
    low = bin_stat(grid, pc.x, pc.y, pc.z, "min")
    filled = _fill_gaps(grid, low)
    floor = ndimage.minimum_filter(filled, size=5, mode="nearest")
    ref = ndimage.uniform_filter(floor, size=5, mode="nearest")
    low[(low - ref) > max_rise] = np.nan
    return low


def _drop_ground_outliers(g: PointCloud, grid: Grid, max_residual: float, passes: int = 2) -> PointCloud:
    """Remove 'ground' points that sit well above the surrounding ground surface.

    Delivered classifications occasionally label low branches or shrubs as ground; left in,
    they create bumps under crowns that look like steep, blocked terrain.
    """
    for _ in range(passes):
        low = bin_stat(grid, g.x, g.y, g.z, "min")
        ref = ndimage.median_filter(_fill_gaps(grid, low), size=5, mode="nearest")
        resid = g.z - grid.sample(ref, g.x, g.y)
        keep = resid <= max_residual
        if keep.all():
            break
        g = g.subset(keep)
    return g


def build_terrain(pc: PointCloud, resolution: float, cfg: GroundCfg) -> Terrain:
    xmin, ymin, xmax, ymax = pc.bounds
    grid = Grid.from_bounds(xmin, ymin, xmax, ymax, resolution)
    is_ground = pc.classification == cfg.classification
    fallback = is_ground.mean() < cfg.min_ground_fraction
    if fallback:
        mean_ground = _lowest_point_ground(pc, grid)
        count = np.isfinite(mean_ground).astype(float)
    else:
        g = pc.subset(is_ground)
        g = _drop_ground_outliers(g, grid, cfg.max_ground_residual)
        mean_ground = bin_stat(grid, g.x, g.y, g.z, "mean")
        count = bin_stat(grid, g.x, g.y, None, "count")
    dem = _fill_gaps(grid, mean_ground)
    return Terrain(grid=grid, dem=dem, ground_count=count, used_fallback=bool(fallback))

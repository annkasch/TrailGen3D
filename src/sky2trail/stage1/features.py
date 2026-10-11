"""Input image of one window: what the aerial lidar shows above each 0.25 m pixel."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..config import Stage1Cfg
from ..grid import Grid, bin_stat


def channel_names(cfg: Stage1Cfg) -> list[str]:
    e = list(cfg.band_edges)
    bands = [f"returns {lo:g}-{hi:g} m" for lo, hi in zip(e[:-1], e[1:])] + [f"returns >{e[-1]:g} m"]
    return bands + ["max height", "mean height", "lowest return >0.3 m", "share not first return",
                    "ground hit", "pulse density", "empty"]


def window_grid(bounds, resolution: float) -> Grid:
    """Square grid exactly covering a window (no snapping, so pixels line up with the core)."""
    xmin, ymin, xmax, ymax = bounds
    n = int(round((xmax - xmin) / resolution))
    return Grid(float(xmin), float(ymax), float(resolution), n, n)


def rasterize(x, y, hag, is_ground, return_number, grid: Grid, cfg: Stage1Cfg) -> np.ndarray:
    """(channels, rows, cols) float32 image; see channel_names for the order.

    Profile channels count non-ground returns per height band, divided by the pulses
    (first returns) in the pixel, so they compare across QL2 and QL1 tiles.
    """
    row, col, ok = grid.index(np.asarray(x), np.asarray(y))
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    hag = np.clip(np.asarray(hag, float)[ok], 0.0, None)
    ground = np.asarray(is_ground, bool)[ok]
    first = np.asarray(return_number)[ok] == 1
    shape = grid.shape
    flat = row[ok] * grid.ncols + col[ok]
    n = grid.nrows * grid.ncols

    def count(sel):
        return np.bincount(flat[sel], minlength=n).reshape(shape).astype(np.float32)

    total = count(np.ones(len(flat), bool))
    pulses = count(first)
    denom = np.maximum(pulses, 1.0)
    s = cfg.height_scale
    veg = ~ground
    edges = list(cfg.band_edges) + [np.inf]
    chans = [count(veg & (hag >= lo) & (hag < hi)) / denom for lo, hi in zip(edges[:-1], edges[1:])]
    chans.append(bin_stat(grid, x[veg], y[veg], hag[veg], "max", fill=0.0) / s)
    chans.append(bin_stat(grid, x[veg], y[veg], hag[veg], "mean", fill=0.0) / s)
    low = veg & (hag >= 0.3)
    chans.append(bin_stat(grid, x[low], y[low], hag[low], "min", fill=0.0) / s)
    chans.append(count(~first) / np.maximum(total, 1.0))
    chans.append((count(ground) > 0).astype(np.float32))
    k = max(1, int(round(2 * cfg.density_radius / grid.resolution)))
    density = ndimage.uniform_filter(pulses, size=k, mode="constant") / grid.resolution**2
    chans.append(density / 10.0)
    chans.append((total == 0).astype(np.float32))
    return np.stack(chans).astype(np.float32)

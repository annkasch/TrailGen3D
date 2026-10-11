"""Low vegetation (shrubs, low branches) from returns in a height band above ground."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import VegetationCfg
from .grid import Grid, bin_stat


def low_vegetation_fraction(grid: Grid, x, y, hag, cfg: VegetationCfg) -> np.ndarray:
    """Share of returns falling in the low band, per cell, smoothed over a neighbourhood.

    Not a true density: canopy above hides what is below, so values under dense crowns
    are underestimated. NaN where there are no returns at all.
    """
    band = (hag >= cfg.low_min) & (hag <= cfg.low_max)
    # Only returns that reached the band or below count towards the denominator.
    reachable = hag <= cfg.low_max
    n_band = bin_stat(grid, x[band], y[band], None, "count")
    n_all = bin_stat(grid, x[reachable], y[reachable], None, "count")
    size = max(1, int(round(2 * cfg.smoothing / grid.resolution)) | 1)
    n_band = ndimage.uniform_filter(n_band, size=size, mode="nearest")
    n_all = ndimage.uniform_filter(n_all, size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_all > 0, n_band / n_all, np.nan)
    return frac

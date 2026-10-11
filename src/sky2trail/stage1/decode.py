"""From output maps to a stem list: peaks of the heatmap, refined by the offset."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..config import Stage1Cfg
from ..grid import Grid
from ..output import Stem
from .targets import pixel_to_world


def decode(heat: np.ndarray, offset: np.ndarray, size: np.ndarray, grid: Grid, cfg: Stage1Cfg,
           core_bounds=None, threshold: float | None = None, std=None) -> dict:
    """Stems as arrays x, y, dbh, height, confidence, small (bool).

    heat must already be probabilities (after the sigmoid). core_bounds (xmin, ymin, xmax,
    ymax), half-open, keeps only stems the window is responsible for.
    std: optional (offset_std, size_std) maps from the Laplace approximation; adds
    xy_std (m, per axis), dbh_std (m) and height_std (m). NaN without them.
    """
    thr = cfg.peak_threshold if threshold is None else threshold
    h = heat.max(axis=0)
    k = 2 * int(round(cfg.peak_radius / cfg.resolution)) + 1
    peak = (h == ndimage.maximum_filter(h, size=k, mode="constant", cval=0.0)) & (h >= thr)
    r, c = np.nonzero(peak)
    v = r + 0.5 + offset[0, r, c]
    u = c + 0.5 + offset[1, r, c]
    x, y = pixel_to_world(v, u, grid)
    dbh = np.exp(size[0, r, c])
    out = {
        "x": x, "y": y,
        "dbh": dbh,
        "height": size[1, r, c] * cfg.height_scale,
        "confidence": h[r, c],
        "small": heat[1, r, c] > heat[0, r, c],
    }
    nan = np.full(len(r), np.nan)
    if std is None:
        out.update(xy_std=nan, dbh_std=nan, height_std=nan)
    else:
        off_std, size_std = std
        # Position: mean of the two offset stds, in meters. DBH: log-normal std (delta method).
        out["xy_std"] = 0.5 * (off_std[0, r, c] + off_std[1, r, c]) * grid.resolution
        out["dbh_std"] = dbh * size_std[0, r, c]
        out["height_std"] = size_std[1, r, c] * cfg.height_scale
    if core_bounds is not None:
        x0, y0, x1, y1 = core_bounds
        keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        out = {key: val[keep] for key, val in out.items()}
    return out


def to_stems(d: dict, start_id: int = 0) -> list[Stem]:
    return [Stem(start_id + i, float(d["x"][i]), float(d["y"][i]), float(d["dbh"][i]),
                 float(d["height"][i]), -1, float(d["confidence"][i]), float(d["xy_std"][i]),
                 float(d["dbh_std"][i]), float(d["height_std"][i])) for i in range(len(d["x"]))]

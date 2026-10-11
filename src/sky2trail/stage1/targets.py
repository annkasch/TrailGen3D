"""Training targets: what stage 1 should output for a window with known stems.

Stems are given in pixel coordinates of the window: (v, u) = (rows down from the top edge,
columns from the left edge), continuous, so pixel (r, c) covers v in [r, r+1), u in [c, c+1).

- heat (2, H, W): a Gaussian blob per stem, exactly 1 at the stem's pixel. Channel 0 holds
  stems >= small_dbh, channel 1 thinner stems.
- offset (2, H, W): stem position minus pixel centre (dv, du), at stem pixels.
- size (2, H, W): log(dbh in m) and height / height_scale, at stem pixels.
- size_mask (2, H, W): 1 where a size value is known (field data may lack heights).
- reg_mask (H, W): 1 at stem pixels (where offset and size are trained).
- weight (2, H, W): where the heatmap loss counts. 0 outside the labelled area, and in the
  small-stem channel for sources that do not map small stems (NEON: only >= 10 cm).
"""

from __future__ import annotations

import numpy as np

from ..config import Stage1Cfg
from ..grid import Grid


def world_to_pixel(x, y, grid: Grid) -> np.ndarray:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.column_stack([(grid.ymax - y) / grid.resolution, (x - grid.xmin) / grid.resolution])


def pixel_to_world(v, u, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    return grid.xmin + np.asarray(u) * grid.resolution, grid.ymax - np.asarray(v) * grid.resolution


def make_targets(stems: np.ndarray, shape: tuple[int, int], label_mask: np.ndarray,
                 small_labelled: bool, cfg: Stage1Cfg) -> dict:
    """stems: (N, 4) array of v, u, dbh (m, NaN if unknown), height (m, NaN if unknown)."""
    H, W = shape
    heat = np.zeros((2, H, W), np.float32)
    offset = np.zeros((2, H, W), np.float32)
    size = np.zeros((2, H, W), np.float32)
    size_mask = np.zeros((2, H, W), np.float32)
    reg_mask = np.zeros((H, W), np.float32)
    sigma = cfg.heatmap_sigma / cfg.resolution
    rad = int(np.ceil(3 * sigma))
    for v, u, dbh, height in np.asarray(stems, float).reshape(-1, 4):
        r, c = int(np.floor(v)), int(np.floor(u))
        if not (0 <= r < H and 0 <= c < W):
            continue
        ch = 1 if (np.isfinite(dbh) and dbh < cfg.small_dbh) else 0
        r0, r1, c0, c1 = max(0, r - rad), min(H, r + rad + 1), max(0, c - rad), min(W, c + rad + 1)
        rr, cc = np.mgrid[r0:r1, c0:c1]
        g = np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * sigma**2))
        heat[ch, r0:r1, c0:c1] = np.maximum(heat[ch, r0:r1, c0:c1], g)
        offset[:, r, c] = (v - r - 0.5, u - c - 0.5)
        if np.isfinite(dbh) and dbh > 0:
            size[0, r, c], size_mask[0, r, c] = np.log(dbh), 1.0
        if np.isfinite(height):
            size[1, r, c], size_mask[1, r, c] = height / cfg.height_scale, 1.0
        reg_mask[r, c] = 1.0
    m = np.asarray(label_mask, np.float32)
    weight = np.stack([m, m * float(small_labelled)])
    # Regression only where the stem lies in labelled area.
    reg_mask *= m
    return {"heat": heat, "offset": offset, "size": size, "size_mask": size_mask * reg_mask,
            "reg_mask": reg_mask, "weight": weight}

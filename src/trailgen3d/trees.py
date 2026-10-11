"""Canopy height model, treetop detection, crown segmentation and baseline trunks.

Baseline only: each crown gets one trunk directly under its treetop, with a diameter
from a height-diameter rule. The learned tree model (Phase 6) replaces this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import TreesCfg
from .grid import Grid, bin_stat


@dataclass
class Tree:
    id: int
    x: float
    y: float
    height: float
    crown_diameter: float
    dbh: float           # trunk diameter at breast height, meters
    n_points: int        # lidar returns inside the crown

    def as_row(self) -> dict:
        return {
            "id": self.id, "x": round(self.x, 2), "y": round(self.y, 2),
            "height": round(self.height, 2), "crown_diameter": round(self.crown_diameter, 2),
            "dbh": round(self.dbh, 3), "n_points": self.n_points,
        }


def _fill_empty(raster: np.ndarray, max_iter: int = 10) -> np.ndarray:
    """Fill NaN cells with the mean of their filled neighbours, growing inwards."""
    out = raster.copy()
    kernel = np.ones((3, 3))
    for _ in range(max_iter):
        empty = np.isnan(out)
        if not empty.any():
            break
        vals = np.where(empty, 0.0, out)
        s = ndimage.convolve(vals, kernel, mode="nearest")
        n = ndimage.convolve((~empty).astype(float), kernel, mode="nearest")
        fill = empty & (n > 0)
        out[fill] = s[fill] / n[fill]
    return np.nan_to_num(out, nan=0.0)


def canopy_height_model(grid: Grid, x, y, hag, min_height: float = 0.5) -> np.ndarray:
    """Highest return above ground per cell.

    Cells with returns but none above `min_height` are open ground (0). Cells with no
    return at all are unknown and filled from their neighbours: at QL2 density many
    0.5 m cells get no pulse, and treating them as 0 would split every crown.
    """
    chm = bin_stat(grid, x, y, np.maximum(hag, 0.0), "max")
    chm[chm < min_height] = 0.0
    chm = _fill_empty(chm)
    # Close pits: a cell much lower than all its neighbours is usually a missed hit.
    closed = ndimage.grey_closing(chm, size=3)
    pit = (closed - chm) > 2.0
    chm[pit] = closed[pit]
    return chm


def _same_crown(chm: np.ndarray, a: tuple[int, int], b: tuple[int, int], h_low: float,
                min_dip: float) -> bool:
    """True if the canopy stays high all the way between two peaks (no saddle between them)."""
    n = int(max(abs(a[0] - b[0]), abs(a[1] - b[1]))) + 1
    rr = np.linspace(a[0], b[0], n).round().astype(int)
    cc = np.linspace(a[1], b[1], n).round().astype(int)
    return chm[rr, cc].min() >= h_low - min_dip


def detect_treetops(grid: Grid, chm: np.ndarray, cfg: TreesCfg) -> list[tuple[int, int, float]]:
    """Local maxima, tallest first. A lower peak is dropped if it lies within a height-dependent
    radius of a taller treetop, or if the canopy between them has no saddle (same crown)."""
    smooth = ndimage.gaussian_filter(chm, sigma=0.6)
    cand = (smooth == ndimage.maximum_filter(smooth, size=3)) & (chm >= cfg.min_tree_height)
    rows, cols = np.nonzero(cand)
    heights = chm[rows, cols]
    order = np.argsort(-heights)
    kept: list[tuple[int, int, float]] = []
    kept_rc = np.empty((0, 2))
    search = cfg.max_crown_radius / grid.resolution
    for i in order:
        r, c, h = rows[i], cols[i], heights[i]
        radius = (cfg.window_base + cfg.window_slope * h) / grid.resolution
        if len(kept_rc):
            dist = np.hypot(kept_rc[:, 0] - r, kept_rc[:, 1] - c)
            if dist.min() < radius:
                continue
            min_dip = max(cfg.min_saddle_drop, cfg.min_saddle_fraction * h)
            near = np.nonzero(dist < search)[0]
            if any(_same_crown(smooth, (r, c), (int(kept_rc[k, 0]), int(kept_rc[k, 1])), smooth[r, c], min_dip)
                   for k in near):
                continue
        kept.append((int(r), int(c), float(h)))
        kept_rc = np.vstack([kept_rc, [r, c]])
    return kept


def segment_crowns(grid: Grid, chm: np.ndarray, tops, cfg: TreesCfg) -> np.ndarray:
    """Label each canopy cell with its tree (1..n) by watershed from the treetops; 0 = none."""
    if not tops:
        return np.zeros(chm.shape, dtype=np.int32)
    canopy = chm >= cfg.min_crown_height
    hmax = max(float(chm.max()), 1e-6)
    inv = np.round((1.0 - chm / hmax) * 65000).astype(np.uint16)
    markers = np.zeros(chm.shape, dtype=np.int32)
    markers[~canopy] = -1
    for i, (r, c, _) in enumerate(tops, start=1):
        markers[r, c] = i
    labels = ndimage.watershed_ift(inv, markers)
    labels[~canopy] = 0
    labels[labels < 0] = 0

    # Drop cells too far from their treetop or far below it (neighbouring understorey).
    yy, xx = np.indices(chm.shape)
    top_r = np.array([t[0] for t in tops])
    top_c = np.array([t[1] for t in tops])
    top_h = np.array([t[2] for t in tops])
    lab = labels - 1
    has = labels > 0
    dist = np.zeros(chm.shape)
    dist[has] = np.hypot(yy[has] - top_r[lab[has]], xx[has] - top_c[lab[has]]) * grid.resolution
    too_far = has & (dist > cfg.max_crown_radius)
    too_low = has & (chm < 0.3 * top_h[np.clip(lab, 0, None)])
    labels[too_far | too_low] = 0
    return labels


def dbh_from_height(height: np.ndarray | float, cfg: TreesCfg) -> np.ndarray | float:
    """Placeholder height-diameter rule (meters). Calibrate a, b per forest type."""
    return cfg.dbh_a * np.power(height, cfg.dbh_b) / 100.0


def extract_trees(grid: Grid, chm: np.ndarray, x, y, hag, cfg: TreesCfg) -> tuple[list[Tree], np.ndarray]:
    tops = detect_treetops(grid, chm, cfg)
    labels = segment_crowns(grid, chm, tops, cfg)
    cell_area = grid.resolution**2
    areas = np.bincount(labels.ravel(), minlength=len(tops) + 1) * cell_area

    # Lidar returns inside each crown: a direct measure of how much data describes each tree.
    row, col, ok = grid.index(x, y)
    pt_label = np.zeros(len(x), dtype=np.int64)
    pt_label[ok] = labels[row[ok], col[ok]]
    pt_label[hag < cfg.min_crown_height * 0.5] = 0
    counts = np.bincount(pt_label, minlength=len(tops) + 1)

    trees = []
    for i, (r, c, h) in enumerate(tops, start=1):
        cx = grid.xmin + (c + 0.5) * grid.resolution
        cy = grid.ymax - (r + 0.5) * grid.resolution
        diam = 2.0 * np.sqrt(areas[i] / np.pi)
        trees.append(Tree(i, cx, cy, h, float(diam), float(dbh_from_height(h, cfg)), int(counts[i])))
    return trees, labels

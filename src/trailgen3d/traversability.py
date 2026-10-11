"""Navigation cost map from a Sky2Trail output: ground surface, stems and low vegetation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sky2trail.grid import Grid, resample
from sky2trail.output import Sky2TrailOutput, Stem

from .config import WorldConfig


@dataclass
class CostMap:
    grid: Grid
    cost: np.ndarray        # 0 = free ... 1 = blocked
    blocked: np.ndarray     # hard limits: stems, too-steep slopes
    layers: dict            # individual cost terms, for inspection
    confidence: np.ndarray  # 0..1, from the Sky2Trail ground confidence


def rasterize_stems(grid: Grid, stems: list[Stem], inflation: float) -> np.ndarray:
    """Discs of radius dbh/2 + inflation around each stem."""
    mask = np.zeros(grid.shape, dtype=bool)
    res = grid.resolution
    for t in stems:
        radius = t.dbh / 2.0 + inflation
        c0 = int(np.floor((t.x - radius - grid.xmin) / res))
        c1 = int(np.ceil((t.x + radius - grid.xmin) / res))
        r0 = int(np.floor((grid.ymax - (t.y + radius)) / res))
        r1 = int(np.ceil((grid.ymax - (t.y - radius)) / res))
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, grid.nrows), min(c1, grid.ncols)
        if r0 >= r1 or c0 >= c1:
            continue
        rr, cc = np.mgrid[r0:r1, c0:c1]
        cx = grid.xmin + (cc + 0.5) * res
        cy = grid.ymax - (rr + 0.5) * res
        mask[r0:r1, c0:c1] |= np.hypot(cx - t.x, cy - t.y) <= radius
    return mask


def build_cost_map(s2t: Sky2TrailOutput, cfg: WorldConfig) -> CostMap:
    g = s2t.ground
    grid = g.grid.resample_like(cfg.grid.cost_resolution)
    r, w = cfg.robot, cfg.cost_weights

    slope = resample(g.grid, g.slope_deg(), grid)
    rough = resample(g.grid, g.roughness(), grid)
    shrub = resample(s2t.canopy.grid, np.nan_to_num(s2t.canopy.low_vegetation, nan=0.0), grid)
    conf = resample(g.grid, g.confidence, grid)

    c_slope = np.clip(slope / r.max_slope, 0, 1)
    c_rough = np.clip(rough / r.max_roughness, 0, 1)
    c_shrub = np.clip(shrub, 0, 1)

    stems = rasterize_stems(grid, s2t.stems, r.half_width + r.safety_margin)
    too_steep = slope > r.max_slope
    blocked = stems | too_steep

    # Independent penalties combined so that no single term exceeds 1.
    cost = 1.0 - (1 - w.slope * c_slope) * (1 - w.roughness * c_rough) * (1 - w.shrub * c_shrub)
    cost[blocked] = 1.0
    layers = {"slope_deg": slope, "roughness": rough, "shrub": shrub, "stems": stems}
    return CostMap(grid, cost, blocked, layers, np.clip(conf, 0, 1))

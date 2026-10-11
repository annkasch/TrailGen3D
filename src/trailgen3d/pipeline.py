"""Build a navigation world from a Sky2Trail output.

TrailGen3D never reads aerial lidar itself: it consumes whatever Sky2Trail method produced
the ground-level data (v0 rules now, trained models later).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sky2trail.io import write_raster
from sky2trail.output import Sky2TrailOutput

from . import __version__
from .config import WorldConfig
from .export import write_nav2_map
from .traversability import CostMap, build_cost_map


@dataclass
class World:
    costmap: CostMap
    summary: dict


def build_world(s2t: Sky2TrailOutput, cfg: WorldConfig) -> World:
    cm = build_cost_map(s2t, cfg)
    summary = {
        "trailgen3d_version": __version__,
        "sky2trail_method": s2t.method,
        "blocked_fraction": round(float(cm.blocked.mean()), 3),
        "mean_cost": round(float(np.nanmean(cm.cost)), 3),
        "low_confidence_fraction": round(float((cm.confidence < 0.3).mean()), 3),
        "config": cfg.to_dict(),
    }
    return World(cm, summary)


def save_world(world: World, out_dir: str | Path, crs_wkt: str | None = None, preview: bool = True) -> Path:
    out = Path(out_dir)
    (out / "rasters").mkdir(parents=True, exist_ok=True)
    cm = world.costmap
    files = {
        "cost": write_raster(out / "rasters/cost", cm.cost, cm.grid, crs_wkt),
        "confidence": write_raster(out / "rasters/confidence", cm.confidence, cm.grid, crs_wkt),
        "nav2_map": write_nav2_map(out / "nav2", cm.cost, cm.grid),
    }
    if preview:
        from .viz import save_world_preview
        files["preview"] = save_world_preview(out / "preview.png", cm)
    summary = dict(world.summary)
    summary["files"] = {k: str(Path(v).relative_to(out)) for k, v in files.items()}
    summary["local_frame_origin"] = {"x": cm.grid.xmin, "y": cm.grid.ymin}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return out

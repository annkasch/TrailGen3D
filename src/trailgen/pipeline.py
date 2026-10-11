"""Phase 1 baseline: one point cloud in, terrain + trees + traversability out."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config
from .grid import Grid
from .io import PointCloud, write_nav2_map, write_raster, write_trees_csv
from .quality import density_report
from .terrain import Terrain, build_terrain
from .traversability import CostMap, build_cost_map
from .trees import Tree, canopy_height_model, extract_trees
from .vegetation import low_vegetation_fraction


@dataclass
class Result:
    terrain: Terrain
    chm_grid: Grid
    chm: np.ndarray
    crowns: np.ndarray
    trees: list[Tree]
    shrub: np.ndarray
    costmap: CostMap
    summary: dict


def process(pc: PointCloud, cfg: Config) -> Result:
    terrain = build_terrain(pc, cfg.grid.dem_resolution, cfg.ground)
    hag = terrain.height_above_ground(pc.x, pc.y, pc.z)

    chm_grid = terrain.grid.resample_like(cfg.grid.chm_resolution)
    chm = canopy_height_model(chm_grid, pc.x, pc.y, hag)
    trees, crowns = extract_trees(chm_grid, chm, pc.x, pc.y, hag, cfg.trees)
    shrub = low_vegetation_fraction(chm_grid, pc.x, pc.y, hag, cfg.vegetation)
    costmap = build_cost_map(terrain, trees, shrub, chm_grid, cfg)

    pts_per_tree = np.array([t.n_points for t in trees]) if trees else np.array([0])
    summary = {
        "trailgen_version": __version__,
        "source": pc.source,
        "tile": density_report(pc),
        "ground_filter": "lowest-point fallback" if terrain.used_fallback else "delivered ground class",
        "trees": {
            "count": len(trees),
            "height_m": _stats([t.height for t in trees]),
            "crown_diameter_m": _stats([t.crown_diameter for t in trees]),
            "lidar_points_per_crown": _stats(pts_per_tree),
            "trunks": "baseline: one trunk under each treetop, diameter from placeholder height rule",
        },
        "traversability": {
            "blocked_fraction": round(float(costmap.blocked.mean()), 3),
            "mean_cost": round(float(np.nanmean(costmap.cost)), 3),
            "low_confidence_fraction": round(float((costmap.confidence < 0.3).mean()), 3),
        },
        "config": cfg.to_dict(),
    }
    return Result(terrain, chm_grid, chm, crowns, trees, shrub, costmap, summary)


def _stats(values) -> dict | None:
    v = np.asarray(values, float)
    if v.size == 0:
        return None
    return {"min": round(float(v.min()), 2), "median": round(float(np.median(v)), 2),
            "max": round(float(v.max()), 2)}


def save(result: Result, out_dir: str | Path, crs_wkt: str | None = None, preview: bool = True) -> Path:
    out = Path(out_dir)
    (out / "rasters").mkdir(parents=True, exist_ok=True)
    t, cm = result.terrain, result.costmap
    files = {
        "dem": write_raster(out / "rasters/dem", t.dem, t.grid, crs_wkt),
        "slope_deg": write_raster(out / "rasters/slope_deg", t.slope_deg(), t.grid, crs_wkt),
        "roughness": write_raster(out / "rasters/roughness", t.roughness(), t.grid, crs_wkt),
        "chm": write_raster(out / "rasters/chm", result.chm, result.chm_grid, crs_wkt),
        "low_vegetation": write_raster(out / "rasters/low_vegetation", result.shrub, result.chm_grid, crs_wkt),
        "cost": write_raster(out / "rasters/cost", cm.cost, cm.grid, crs_wkt),
        "confidence": write_raster(out / "rasters/confidence", cm.confidence, cm.grid, crs_wkt),
        "trees": write_trees_csv(out / "trees.csv", [tr.as_row() for tr in result.trees]),
        "nav2_map": write_nav2_map(out / "nav2", cm.cost, cm.grid),
    }
    if preview:
        from .viz import save_preview
        files["preview"] = save_preview(
            out / "preview.png", t.grid, t.dem, result.chm_grid, result.chm, result.trees,
            result.shrub, cm.grid, cm.cost, cm.confidence)
    summary = dict(result.summary)
    summary["files"] = {k: str(Path(v).relative_to(out)) for k, v in files.items()}
    summary["local_frame_origin"] = {"x": cm.grid.xmin, "y": cm.grid.ymin}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return out

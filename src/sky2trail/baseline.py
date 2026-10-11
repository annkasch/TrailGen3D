"""Sky2Trail v0: fixed rules, no learning.

The baseline every trained Sky2Trail model must beat, and a reference implementation of
the output format. Measured parts come straight from the aerial lidar; the stems under
each crown are guessed:
  - one stem per detected crown, directly below the treetop,
  - diameter from height: dbh_cm = a * height_m ** b (placeholder coefficients),
  - stem drawn as a vertical cylinder from the ground to an assumed crown base.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Sky2TrailConfig
from .io import PointCloud
from .output import (COMPONENT_CROWN, COMPONENT_GROUND, COMPONENT_STEM, COMPONENT_UNDERSTORY,
                     SOURCE_GENERATED, SOURCE_MEASURED, Canopy, CompletedCloud, GroundSurface,
                     Sky2TrailOutput, Stem)
from .quality import density_report
from .terrain import Terrain, build_terrain
from .trees import Crown, canopy_height_model, extract_crowns
from .vegetation import low_vegetation_fraction

METHOD = "v0-rules"


def dbh_from_height(height, a: float, b: float):
    """Placeholder height-diameter rule, meters."""
    return a * np.power(height, b) / 100.0


def guess_stems(crowns: list[Crown], cfg: Sky2TrailConfig) -> list[Stem]:
    s = cfg.stems_v0
    stems = []
    for c in crowns:
        # Confidence here only reflects how much lidar data describes the crown; it is not a
        # calibrated probability that a stem of this size stands at this spot.
        conf = float(min(1.0, c.n_points / 50.0))
        stems.append(Stem(c.id, c.x, c.y, float(dbh_from_height(c.height, s.dbh_a, s.dbh_b)),
                          c.height, c.id, conf))
    return stems


def stem_points(stems: list[Stem], ground: GroundSurface, crown_base_fraction: float,
                spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points on a vertical cylinder per stem, from the ground to the assumed crown base."""
    xs, ys, zs = [], [], []
    if not stems:
        return np.array([]), np.array([]), np.array([])
    z0 = ground.elevation_at([s.x for s in stems], [s.y for s in stems])
    for s, base in zip(stems, z0):
        r = s.dbh / 2.0
        top = crown_base_fraction * s.height
        n_around = max(6, int(np.ceil(2 * np.pi * r / spacing)))
        n_up = max(2, int(np.ceil(top / spacing)))
        ang = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
        hgt = np.linspace(0, top, n_up)
        A, H = np.meshgrid(ang, hgt)
        xs.append(s.x + r * np.cos(A).ravel())
        ys.append(s.y + r * np.sin(A).ravel())
        zs.append(base + H.ravel())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)


@dataclass
class Measured:
    """What the aerial lidar measures directly; shared by v0 and trained methods."""

    terrain: Terrain
    hag: np.ndarray          # height above ground of every input point
    ground: GroundSurface
    canopy: Canopy


def measure(pc: PointCloud, cfg: Sky2TrailConfig) -> Measured:
    terrain = build_terrain(pc, cfg.grid.dem_resolution, cfg.ground)
    hag = terrain.height_above_ground(pc.x, pc.y, pc.z)
    ground = GroundSurface(terrain.grid, terrain.dem, np.clip(terrain.ground_support(), 0, 1))
    chm_grid = terrain.grid.resample_like(cfg.grid.chm_resolution)
    chm = canopy_height_model(chm_grid, pc.x, pc.y, hag)
    crowns, _ = extract_crowns(chm_grid, chm, pc.x, pc.y, hag, cfg.trees)
    low_veg = low_vegetation_fraction(chm_grid, pc.x, pc.y, hag, cfg.vegetation)
    return Measured(terrain, hag, ground, Canopy(chm_grid, chm, low_veg, crowns))


def assemble(pc: PointCloud, cfg: Sky2TrailConfig, m: Measured, stems: list[Stem], method: str,
             extra_summary: dict | None = None) -> Sky2TrailOutput:
    """Sky2Trail output from measured layers plus a stem list (guessed or predicted)."""
    comp = np.where(m.hag < cfg.trees.min_crown_height, COMPONENT_UNDERSTORY, COMPONENT_CROWN).astype(np.uint8)
    comp[pc.classification == cfg.ground.classification] = COMPONENT_GROUND
    sx, sy, sz = stem_points(stems, m.ground, cfg.stems_v0.crown_base_fraction, cfg.stems_v0.point_spacing)
    points = CompletedCloud(
        x=np.concatenate([pc.x, sx]), y=np.concatenate([pc.y, sy]), z=np.concatenate([pc.z, sz]),
        source=np.concatenate([np.full(len(pc), SOURCE_MEASURED, np.uint8), np.full(len(sx), SOURCE_GENERATED, np.uint8)]),
        component=np.concatenate([comp, np.full(len(sx), COMPONENT_STEM, np.uint8)]),
    )
    crowns = m.canopy.crowns
    summary = {
        "source": pc.source,
        "tile": density_report(pc),
        "ground_filter": "lowest-point fallback" if m.terrain.used_fallback else "delivered ground class",
        "ground_measured_fraction": round(float((m.terrain.ground_count > 0).mean()), 3),
        "crowns": len(crowns),
        "stems": len(stems),
        "lidar_points_per_crown_median": float(np.median([c.n_points for c in crowns])) if crowns else None,
    }
    summary.update(extra_summary or {})
    summary["config"] = cfg.to_dict()
    return Sky2TrailOutput(method, m.ground, stems, points, m.canopy, summary)


def run_v0(pc: PointCloud, cfg: Sky2TrailConfig) -> Sky2TrailOutput:
    m = measure(pc, cfg)
    stems = guess_stems(m.canopy.crowns, cfg)
    return assemble(pc, cfg, m, stems, METHOD, {
        "stem_rule": "one stem per crown under the treetop; dbh from placeholder height rule"})

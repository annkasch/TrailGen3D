"""Synthetic forest + simulated airborne lidar, with known ground truth.

Lets you develop and test the pipeline before downloading real tiles, and measures
how well trees are recovered at a given pulse density (e.g. QL2 vs QL1).

By default the forest is easy: well-spaced, upright trees with round crowns. The options
`understory_trees`, `max_lean` and `trunk_hit_prob` add what makes real forests hard and
what a trained model should learn from: small trees under big crowns, crowns that are not
above their stems, and the occasional pulse that hits a trunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .io import PointCloud


@dataclass
class SyntheticScene:
    points: PointCloud
    trees: np.ndarray   # columns: x, y (stem base), height, crown_radius, dbh
    ground_fn: object   # callable (x, y) -> ground elevation
    crown_xy: np.ndarray | None = None  # crown centres (differ from stem bases if trees lean)
    size: float = 0.0


def _ground(x, y, base=400.0):
    return (base + 0.08 * x + 0.03 * y
            + 1.5 * np.sin(x / 18.0) * np.cos(y / 23.0)
            + 0.3 * np.sin(x / 3.1 + y / 4.7))


def _dbh(h, rng):
    return 0.015 * h**1.05 * rng.uniform(0.8, 1.2)  # meters, loosely tied to height


def make_scene(size: float = 120.0, n_trees: int = 60, pulses_per_m2: float = 2.0,
               shrub_patches: int = 12, ground_misclass: float = 0.03,
               seed: int = 0, understory_trees: int = 0, max_lean: float = 0.0,
               trunk_hit_prob: float = 0.0, penetration: float = 0.35) -> SyntheticScene:
    """
    understory_trees: small trees (3-9 m) placed under the crowns of bigger ones.
    max_lean: crown centre offset from the stem base, up to this many meters.
    trunk_hit_prob: chance that a pulse passing within a trunk's footprint returns from it.
    penetration: chance that a pulse through two crowns also returns from the lower one
        (only used when understory_trees > 0).
    """
    rng = np.random.default_rng(seed)

    # Trees with a minimum spacing, so crowns overlap only partly.
    trees = []
    attempts = 0
    while len(trees) < n_trees and attempts < 20000:
        attempts += 1
        x, y = rng.uniform(5, size - 5, 2)
        h = rng.uniform(8, 32)
        rad = 0.12 * h + rng.uniform(0.5, 1.5)
        if all(np.hypot(x - t[0], y - t[1]) > 0.8 * (rad + t[3]) for t in trees):
            trees.append((x, y, h, rad, _dbh(h, rng)))

    # Understory: small trees under the crowns of the trees above.
    n_big = len(trees)
    attempts = 0
    while understory_trees and len(trees) < n_big + understory_trees and attempts < 20000:
        attempts += 1
        host = trees[rng.integers(n_big)]
        ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 0.9) * host[3]
        x, y = host[0] + dist * np.cos(ang), host[1] + dist * np.sin(ang)
        if not (2 < x < size - 2 and 2 < y < size - 2):
            continue
        if any(np.hypot(x - t[0], y - t[1]) < 1.5 for t in trees):
            continue
        h = rng.uniform(3, 9)
        trees.append((x, y, h, 0.12 * h + rng.uniform(0.3, 0.8), _dbh(h, rng)))
    trees = np.array(trees)

    crown_xy = trees[:, :2].copy()
    if max_lean > 0:
        ang = rng.uniform(0, 2 * np.pi, len(trees))
        off = rng.uniform(0, max_lean, len(trees))
        crown_xy += np.column_stack([off * np.cos(ang), off * np.sin(ang)])

    shrubs = np.column_stack([
        rng.uniform(0, size, shrub_patches), rng.uniform(0, size, shrub_patches),
        rng.uniform(2, 6, shrub_patches), rng.uniform(0.6, 1.6, shrub_patches),
    ])  # x, y, radius, height

    n = rng.poisson(pulses_per_m2 * size * size)
    px, py = rng.uniform(0, size, n), rng.uniform(0, size, n)
    gz = _ground(px, py)

    # Crown surface: paraboloid from tree top down to 60 % crown depth.
    dx = px[:, None] - crown_xy[None, :, 0]
    dy = py[:, None] - crown_xy[None, :, 1]
    rr = np.hypot(dx, dy) / trees[None, :, 3]
    inside = rr < 1.0
    surf = np.where(inside, trees[None, :, 2] * (1 - 0.6 * rr**2), -np.inf)
    crown_top = surf.max(axis=1)
    crown_idx = surf.argmax(axis=1)
    in_crown = np.isfinite(crown_top)
    crown_base = np.where(in_crown, 0.4 * trees[crown_idx, 2], 0.0)
    layered = understory_trees > 0
    if layered and surf.shape[1] > 1:
        second = np.partition(surf, -2, axis=1)[:, -2]
    else:
        second = np.full(n, -np.inf)
    del dx, dy, rr, surf, inside

    # Trunk footprint: pulses that pass within a stem radius (+ beam footprint) of a stem.
    trunk_hit = np.zeros(n, bool)
    trunk_top = np.zeros(n)
    if trunk_hit_prob > 0:
        d, j = cKDTree(trees[:, :2]).query(np.column_stack([px, py]))
        near = d < trees[j, 4] / 2 + 0.15
        trunk_hit = near & (rng.random(n) < trunk_hit_prob)
        trunk_top = 0.4 * trees[j, 2]

    sx = px[:, None] - shrubs[None, :, 0]
    sy = py[:, None] - shrubs[None, :, 1]
    in_shrub_all = np.hypot(sx, sy) < shrubs[None, :, 2]
    in_shrub = in_shrub_all.any(axis=1)
    shrub_h = np.where(in_shrub_all, shrubs[None, :, 3], 0).max(axis=1)

    xs, ys, zs, cls, rn, nr = [], [], [], [], [], []
    for i in range(n):
        rets = []
        if in_crown[i] and rng.random() < 0.85:
            rets.append(crown_top[i] - rng.uniform(0, 0.4))
            if rng.random() < 0.5:  # a deeper hit inside the crown
                rets.append(rng.uniform(crown_base[i], crown_top[i]))
        if layered and np.isfinite(second[i]) and second[i] < crown_top[i] - 1.0 \
                and rng.random() < penetration:
            rets.append(second[i] - rng.uniform(0, 0.4))  # a lower crown seen through a gap
        if trunk_hit[i]:
            rets.append(rng.uniform(0.5, trunk_top[i]))
        if in_shrub[i] and rng.random() < (0.35 if rets else 0.6):
            rets.append(rng.uniform(0.3, shrub_h[i]))
        ground_p = 0.45 if in_crown[i] else (0.6 if in_shrub[i] else 1.0)
        has_ground = rng.random() < ground_p
        k = len(rets) + int(has_ground)
        if k == 0:
            continue
        for j, hgt in enumerate(sorted(rets, reverse=True)):
            xs.append(px[i]); ys.append(py[i]); zs.append(gz[i] + hgt)
            cls.append(2 if rng.random() < ground_misclass * 0.2 else 1)
            rn.append(j + 1); nr.append(k)
        if has_ground:
            xs.append(px[i]); ys.append(py[i]); zs.append(gz[i] + rng.normal(0, 0.05))
            cls.append(1 if rng.random() < ground_misclass else 2)
            rn.append(k); nr.append(k)

    pc = PointCloud(
        np.array(xs), np.array(ys), np.array(zs), np.array(cls, dtype=np.uint8),
        np.array(rn, dtype=np.uint8), np.array(nr, dtype=np.uint8), None, "synthetic",
    )
    return SyntheticScene(pc, trees, _ground, crown_xy, float(size))


def make_hard_scene(size: float = 100.0, pulses_per_m2: float = 2.0, seed: int = 0) -> SyntheticScene:
    """The forest stage 1 trains and is scored on: understory, leaning trees, trunk hits."""
    rng = np.random.default_rng(10_000 + seed)
    area_ha = size * size / 10_000
    return make_scene(size=size, n_trees=int(rng.uniform(30, 55) * area_ha),
                      pulses_per_m2=pulses_per_m2, seed=seed,
                      shrub_patches=int(10 * area_ha),
                      understory_trees=int(rng.uniform(10, 25) * area_ha),
                      max_lean=1.5, trunk_hit_prob=0.5)

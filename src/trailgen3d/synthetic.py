"""Synthetic forest + simulated airborne lidar, with known ground truth.

Lets you develop and test the pipeline before downloading real tiles, and measures
how well trees are recovered at a given pulse density (e.g. QL2 vs QL1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import PointCloud


@dataclass
class SyntheticScene:
    points: PointCloud
    trees: np.ndarray   # columns: x, y, height, crown_radius, dbh
    ground_fn: object   # callable (x, y) -> ground elevation


def _ground(x, y, base=400.0):
    return (base + 0.08 * x + 0.03 * y
            + 1.5 * np.sin(x / 18.0) * np.cos(y / 23.0)
            + 0.3 * np.sin(x / 3.1 + y / 4.7))


def make_scene(size: float = 120.0, n_trees: int = 60, pulses_per_m2: float = 2.0,
               shrub_patches: int = 12, ground_misclass: float = 0.03,
               seed: int = 0) -> SyntheticScene:
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
            dbh = 0.015 * h**1.05 * rng.uniform(0.8, 1.2)  # meters, loosely tied to height
            trees.append((x, y, h, rad, dbh))
    trees = np.array(trees)

    shrubs = np.column_stack([
        rng.uniform(0, size, shrub_patches), rng.uniform(0, size, shrub_patches),
        rng.uniform(2, 6, shrub_patches), rng.uniform(0.6, 1.6, shrub_patches),
    ])  # x, y, radius, height

    n = rng.poisson(pulses_per_m2 * size * size)
    px, py = rng.uniform(0, size, n), rng.uniform(0, size, n)
    gz = _ground(px, py)

    # Crown surface: paraboloid from tree top down to 60 % crown depth.
    dx = px[:, None] - trees[None, :, 0]
    dy = py[:, None] - trees[None, :, 1]
    rr = np.hypot(dx, dy) / trees[None, :, 3]
    inside = rr < 1.0
    surf = np.where(inside, trees[None, :, 2] * (1 - 0.6 * rr**2), -np.inf)
    crown_top = surf.max(axis=1)
    crown_idx = surf.argmax(axis=1)
    in_crown = np.isfinite(crown_top)
    crown_base = np.where(in_crown, 0.4 * trees[crown_idx, 2], 0.0)

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
    return SyntheticScene(pc, trees, _ground)

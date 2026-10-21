"""Pair building from a registered scan: one pair per tree near the scanner, inside the unit box."""

import numpy as np

from sky2trail.config import TreesCfg
from sky2trail.fg3d.build import BuildCfg, plot_pairs, split_plots
from sky2trail.fg3d.synthetic import als_view, make_tree
from sky2trail.register import Registration, apply, make_transform


def _grid_stand(rng, spacing=9.0, n=4):
    parts = []
    for i in range(n):
        for j in range(n):
            pts, _ = make_tree(rng, n_dense=8000)
            parts.append(pts[pts[:, 2] > 0.1] + [(i - (n - 1) / 2) * spacing, (j - (n - 1) / 2) * spacing, 0])
    g = rng.uniform(-25, 25, (60_000, 2))
    return np.concatenate(parts + [np.c_[g, rng.normal(0, 0.02, len(g))]])


def test_plot_pairs_on_synthetic_stand():
    rng = np.random.default_rng(1)
    world = _grid_stand(rng) + [431000.0, 3548000.0, 10.0]
    als = als_view(world, rng, pulses_per_m2=15)
    als = np.c_[als, np.where(als[:, 2] < 10.1, 2, 1)]
    T = make_transform(30.0, [431000.0, 3548000.0, 11.5])
    scan = apply(np.linalg.inv(T), world[np.hypot(world[:, 0] - 431000, world[:, 1] - 3548000) < 25])
    reg = Registration(T.tolist(), 30.0, T[:3, 3].tolist(), 0.9, 0.3, 0.1, 0.9, True)

    cfg = BuildCfg(max_dist=18, min_tls=500, min_height=3)
    pairs, info = plot_pairs(scan, als, reg, "P1", "pre", cfg, TreesCfg(max_crown_radius=5))
    assert 10 <= len(pairs) <= 16, [i["height_als"] for i in info]
    for p, i in zip(pairs, info):
        assert p.tls.shape == (cfg.n_tls, 3)
        assert p.tls.min() > -0.1 and p.tls.max() < 1.1
        assert i["tls_in_als_hull"] > 0.6
    assert len({p.tree for p in pairs}) == len(pairs)


def test_split_plots_keeps_plots_whole():
    s = split_plots([f"P{i}" for i in range(12)])
    allp = s["train"] + s["val"] + s["test"]
    assert sorted(allp) == sorted(f"P{i}" for i in range(12)) and len(set(allp)) == 12
    assert len(s["test"]) == 2 and len(s["val"]) == 2

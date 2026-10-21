"""TLS -> ALS registration: recover a known heading and shift on a synthetic stand."""

import numpy as np

from sky2trail.fg3d.synthetic import als_view, make_tree
from sky2trail.register import RegCfg, apply, make_transform, register_scan


def _stand(seed=0, size=36.0, n_trees=14):
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(n_trees):
        pts, _ = make_tree(rng, n_dense=6000)
        parts.append(pts[pts[:, 2] > 0.1] + [*rng.uniform(-size / 2, size / 2, 2), 0])
    g = rng.uniform(-size / 2 - 4, size / 2 + 4, (40_000, 2))
    ground = np.c_[g, 0.02 * g[:, 0] + rng.normal(0, 0.02, len(g))]   # gentle slope
    trees = np.concatenate(parts)
    trees[:, 2] += 0.02 * trees[:, 0]
    return np.concatenate([trees, ground]), len(ground), rng


def test_recovers_heading_and_shift():
    scene, n_ground, rng = _stand()
    world = scene + [431000.0, 3548000.0, 27.0]
    als = als_view(world, rng, pulses_per_m2=10)
    cls = np.ones(len(als))
    cls[als[:, 2] - (27 + 0.02 * (als[:, 0] - 431000)) < 0.1] = 2
    als = np.c_[als, cls]

    true = make_transform(57.0, [431001.5, 3547999.0, 28.4])     # scanner at (1.5, -1, 1.4) in the stand
    tls_world = world[np.hypot(world[:, 0] - 431001.5, world[:, 1] - 3547999.0) < 14]
    tls = apply(np.linalg.inv(true), tls_world)                     # into the scanner frame

    cfg = RegCfg(tls_radius=12, search=5, heading_step=2)
    reg = register_scan(tls, als, gps=(431000.0, 3548000.0), cfg=cfg)
    err_h = abs((reg.heading_deg - 57.0 + 180) % 360 - 180)
    err_t = np.abs(np.array(reg.shift) - true[:3, 3])
    assert err_h < 2.0, reg
    assert err_t[:2].max() < 0.2 and err_t[2] < 0.1, (err_t, reg)
    assert reg.ok

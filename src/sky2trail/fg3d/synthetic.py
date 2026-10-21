"""Synthetic ALS/TLS tree pairs for developing and testing the ForestGen3D pipeline.

A tree is a tapered stem cylinder, a few branches, a crown (points filling an ellipsoid or cone
envelope, denser near the surface) and a patch of ground. The TLS-like cloud samples all of it,
thinning with height as a ground scanner sees less of the upper crown. The ALS-like cloud comes
from top-down pulses: per pulse the first hit is the highest point under the beam, with a chance
of further returns below and of reaching the ground through gaps. So, as in real data, ALS sees
the crown top well and the stem hardly at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pairs import Pair, box_norm, save_pair, subsample, to_unit


def make_tree(rng: np.random.Generator, n_dense: int = 40_000) -> tuple[np.ndarray, dict]:
    """Dense surface points (n, 3) of one tree in metres, base at the origin, and its parameters."""
    h = rng.uniform(8, 30)
    dbh = np.clip(0.012 * h * rng.lognormal(0, 0.2), 0.08, 0.9)
    base_frac = rng.uniform(0.3, 0.65)
    crown_r = np.clip(h * rng.uniform(0.12, 0.25), 1.0, 6.0)
    conifer = rng.random() < 0.5
    lean = rng.normal(0, 0.03, 2)                 # stem lean, m per m of height
    cb = base_frac * h

    n_stem, n_branch, n_ground = int(0.25 * n_dense), int(0.1 * n_dense), int(0.1 * n_dense)
    n_crown = n_dense - n_stem - n_branch - n_ground

    z = rng.uniform(0, h * 0.95, n_stem)
    r = dbh / 2 * (1 - 0.8 * z / h)
    a = rng.uniform(0, 2 * np.pi, n_stem)
    stem = np.c_[r * np.cos(a) + lean[0] * z, r * np.sin(a) + lean[1] * z, z]

    nb = rng.integers(6, 16)
    bz = rng.uniform(cb, h * 0.9, nb)
    ba = rng.uniform(0, 2 * np.pi, nb)
    bl = crown_r * rng.uniform(0.4, 0.9, nb)
    k = rng.integers(0, nb, n_branch)
    s = rng.random(n_branch)
    branch = np.c_[lean[0] * bz[k] + s * bl[k] * np.cos(ba[k]),
                   lean[1] * bz[k] + s * bl[k] * np.sin(ba[k]),
                   bz[k] + s * bl[k] * 0.3]

    u = rng.random(n_crown) ** 0.3                # radial fraction, biased to the envelope
    t = rng.uniform(0, 1, n_crown)                # height fraction within the crown
    zc = cb + t * (h - cb)
    if conifer:
        env = crown_r * (1 - t)
    else:
        env = crown_r * np.sqrt(np.clip(1 - (2 * t - 1) ** 2, 0, None))
    a = rng.uniform(0, 2 * np.pi, n_crown)
    crown = np.c_[lean[0] * zc + u * env * np.cos(a), lean[1] * zc + u * env * np.sin(a), zc]

    gr = crown_r * 1.2 * np.sqrt(rng.random(n_ground))
    a = rng.uniform(0, 2 * np.pi, n_ground)
    ground = np.c_[gr * np.cos(a), gr * np.sin(a), rng.normal(0, 0.03, n_ground)]

    pts = np.concatenate([stem, branch, crown, ground])
    return pts, dict(height=h, dbh=dbh, crown_base=cb, crown_radius=crown_r, conifer=conifer)


def tls_view(pts: np.ndarray, rng: np.random.Generator, height: float) -> np.ndarray:
    """Keep points with a probability falling with height (occlusion from below)."""
    keep = rng.random(len(pts)) < np.clip(1.0 - 0.6 * pts[:, 2] / height, 0.2, 1.0)
    return pts[keep]


def als_view(pts: np.ndarray, rng: np.random.Generator, pulses_per_m2: float = 15.0,
             beam: float = 0.25, p_second: float = 0.4) -> np.ndarray:
    """Top-down returns from random pulse positions over the tree's footprint."""
    lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
    n = rng.poisson(pulses_per_m2 * np.prod(hi - lo))
    xy = rng.uniform(lo, hi, (n, 2))
    from scipy.spatial import cKDTree
    tree = cKDTree(pts[:, :2])
    out = []
    for q, nb in zip(xy, tree.query_ball_point(xy, beam)):
        if not nb:
            continue
        nb = np.asarray(nb)
        order = nb[np.argsort(-pts[nb, 2])]
        out.append(pts[order[0]])
        if len(order) > 3 and rng.random() < p_second:
            out.append(pts[order[rng.integers(len(order) // 3, len(order))]])
    return np.array(out).reshape(-1, 3)


def make_pair(seed: int, pulses_per_m2: float = 15.0, n_tls: int = 2048) -> tuple[Pair, dict]:
    rng = np.random.default_rng(seed)
    pts, info = make_tree(rng)
    tls = tls_view(pts, rng, info["height"])
    als = als_view(pts, rng, pulses_per_m2)
    lo, hi = pts.min(0), pts.max(0)
    norm = box_norm(lo[0], lo[1], hi[0], hi[1], lo[2], hi[2])
    tls_u = to_unit(tls, norm)
    tls_u = tls_u[subsample(len(tls_u), n_tls, rng)]
    return Pair(to_unit(als, norm), tls_u, norm, "synth", f"{seed:05d}", "synth"), info


def write_pairs(out: str | Path, n: int, seed: int = 0, pulses_per_m2: float = 15.0,
                val_frac: float = 0.1, test_frac: float = 0.1, log=print) -> list[Path]:
    """n synthetic pairs plus split.json (train / val / test)."""
    import json

    out = Path(out)
    paths = []
    for i in range(n):
        p, _ = make_pair(seed + i, pulses_per_m2)
        paths.append(save_pair(p, out / f"synth_{seed + i:05d}.npz"))
        if log and (i + 1) % 100 == 0:
            log(f"{i + 1}/{n} pairs")
    names = [q.name for q in paths]
    n_val, n_test = int(round(val_frac * n)), int(round(test_frac * n))
    split = {"test": names[:n_test], "val": names[n_test:n_test + n_val], "train": names[n_test + n_val:]}
    (out / "split.json").write_text(json.dumps(split, indent=1))
    return paths

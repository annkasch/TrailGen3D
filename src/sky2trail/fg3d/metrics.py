"""Point cloud metrics for generated trees, as in ForestGen3D, plus tree measurements.

- Chamfer distance (CD): mean squared nearest-neighbour distance, both directions, summed
  (the PDR / ForestGen3D convention on normalised clouds).
- Earth mover's distance (EMD): mean distance under the optimal one-to-one matching of two
  equal-size clouds; exact (Hungarian, scipy) or approximate (Sinkhorn, torch).
- ALS envelope containment (EPC): fraction of generated points inside the convex hull of the
  ALS points.
- DBH by a circle fit to a stem slice 1.2-1.4 m above ground, and tree height.

numpy/scipy functions work without PyTorch; `chamfer_torch` and `emd_sinkhorn` need it.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, cKDTree


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    da = cKDTree(b).query(a)[0]
    db = cKDTree(a).query(b)[0]
    return float(np.mean(da ** 2) + np.mean(db ** 2))


def emd(a: np.ndarray, b: np.ndarray) -> float:
    """Exact EMD for equal-size clouds (O(n^3); fine for 2048 points, ~1 s)."""
    if len(a) != len(b):
        raise ValueError("EMD needs equal-size clouds")
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    r, c = linear_sum_assignment(d)
    return float(d[r, c].mean())


def envelope_containment(points: np.ndarray, als: np.ndarray) -> float:
    """Fraction of points inside the convex hull of the ALS points (EPC)."""
    if len(als) < 4:
        return float("nan")
    try:
        hull = Delaunay(als)
    except Exception:             # degenerate (flat) ALS set
        return float("nan")
    return float(np.mean(hull.find_simplex(points) >= 0))


def fit_circle(xy: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit: centre x, centre y, radius."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)[0]
    cx, cy = sol[0], sol[1]
    return float(cx), float(cy), float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))


def ransac_circle(xy: np.ndarray, tol: float, iters: int = 200, seed: int = 0) -> tuple[float, float, float]:
    """Circle with the most points within tol of it, refitted on those points."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        cx, cy, r = fit_circle(xy[rng.choice(len(xy), 3, replace=False)])
        n = int(np.sum(np.abs(np.linalg.norm(xy - [cx, cy], axis=1) - r) <= tol))
        if n > best_n:
            best, best_n = (cx, cy, r), n
    cx, cy, r = best
    inl = np.abs(np.linalg.norm(xy - [cx, cy], axis=1) - r) <= tol
    return fit_circle(xy[inl]) if inl.sum() >= 3 else best


def dbh(points: np.ndarray, ground: float = 0.0, lo: float = 1.2, hi: float = 1.4,
        centre: tuple[float, float] | None = None, search: float = 1.0, min_points: int = 8,
        tol: float = 0.02) -> float:
    """Stem diameter (m) from points in metres: circle fit to the slice lo..hi above ground.

    centre: approximate stem position; points farther than `search` from it are ignored
    (default: median of the slice). RANSAC with inlier tolerance `tol` (m) ignores branches and
    understory in the slice. Returns nan with too few points or no plausible circle (radius
    beyond `search`, e.g. a near-straight line of diffuse points).
    """
    z = points[:, 2] - ground
    s = points[(z >= lo) & (z <= hi), :2]
    if len(s) < min_points:
        return float("nan")
    c = np.median(s, axis=0) if centre is None else np.asarray(centre)
    s = s[np.linalg.norm(s - c, axis=1) <= search]
    if len(s) < min_points:
        return float("nan")
    cx, cy, r = ransac_circle(s, tol)
    return 2 * r if np.isfinite(r) and r <= search else float("nan")


def tree_height(points: np.ndarray, ground: float = 0.0, q: float = 99.5) -> float:
    """Height above ground of the q-th percentile point (robust to a few outliers)."""
    return float(np.percentile(points[:, 2], q) - ground)


def tree_metrics(gen: np.ndarray, ref: np.ndarray, als: np.ndarray, exact_emd: bool = True) -> dict:
    """CD, EMD and EPC for one normalised tree."""
    out = {"cd": chamfer(gen, ref), "epc": envelope_containment(gen, als)}
    if exact_emd and len(gen) == len(ref):
        out["emd"] = emd(gen, ref)
    return out


def chamfer_torch(a, b):
    """Batched Chamfer distance (B,) for a (B, N, 3) and b (B, M, 3) tensors."""
    from .ops import square_distance

    d = square_distance(a.float(), b.float())
    return d.min(2).values.mean(1) + d.min(1).values.mean(1)


def emd_sinkhorn(a, b, eps: float = 1e-3, iters: int = 200):
    """Approximate EMD (B,) by entropic optimal transport with uniform weights (torch)."""
    import torch

    from .ops import square_distance

    cost = square_distance(a.float(), b.float()).sqrt()
    B, n, m = cost.shape
    log_k = -cost / eps
    log_mu = torch.full((B, n), -np.log(n), device=a.device)
    log_nu = torch.full((B, m), -np.log(m), device=a.device)
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(log_k + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(log_k + u.unsqueeze(2), dim=1)
    plan = torch.exp(log_k + u.unsqueeze(2) + v.unsqueeze(1))
    return (plan * cost).sum((1, 2)) / plan.sum((1, 2))

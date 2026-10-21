"""PointNet++ building blocks in plain PyTorch: sampling, grouping, interpolation.

Batched tensors, points as (B, N, 3), features as (B, N, C). These replace the custom CUDA ops
of the original PointNet++ / PDR code so the model runs on CUDA, Apple MPS and CPU alike. They
use dense (B, M, N) distance matrices, fine for the ~2-4k points per tree we work with.
"""

from __future__ import annotations

import torch


def square_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared distances (B, M, N) between a (B, M, 3) and b (B, N, 3).

    Uses |a|^2 - 2ab + |b|^2 (no (B, M, N, 3) temporary); both sets are centred first to limit
    float32 cancellation. Expect absolute errors around 1e-7 on unit-scale clouds.
    """
    c = b.mean(1, keepdim=True)
    a, b = a - c, b - c
    d =(a * a).sum(-1, keepdim=True) - 2 * a @ b.transpose(1, 2) + (b * b).sum(-1).unsqueeze(1)
    return d.clamp_min_(0)


def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (B, N, C) indexed by idx (B, ...) -> (B, ..., C)."""
    B = x.shape[0]
    flat = idx.reshape(B, -1)
    out = torch.gather(x, 1, flat.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return out.reshape(*idx.shape, x.shape[-1])


def farthest_point_sample(xyz: torch.Tensor, m: int, start: torch.Tensor | None = None) -> torch.Tensor:
    """Indices (B, m) of m points spread by farthest point sampling.

    start: (B,) index of the first point (default 0, deterministic).
    """
    B, N, _ = xyz.shape
    idx = torch.zeros(B, m, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device, dtype=xyz.dtype)
    cur = torch.zeros(B, dtype=torch.long, device=xyz.device) if start is None else start.to(xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(m):
        idx[:, i] = cur
        c = xyz[ar, cur].unsqueeze(1)
        dist = torch.minimum(dist, ((xyz - c) ** 2).sum(-1))
        cur = dist.argmax(-1)
    return idx


def knn(query: torch.Tensor, points: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """k nearest points for each query: (indices (B, M, k), squared distances (B, M, k)), nearest first."""
    d = square_distance(query, points)
    dist, idx = d.topk(min(k, points.shape[1]), dim=-1, largest=False)
    return idx, dist


def ball_query(query: torch.Tensor, points: torch.Tensor, radius: float, k: int) -> torch.Tensor:
    """Up to k point indices (B, M, k) within `radius` of each query.

    Like the PointNet++ op: neighbours are taken nearest first, and slots without a neighbour in
    the ball repeat the nearest one (which is always used, even if outside the ball, so every
    query has a valid group).
    """
    idx, dist = knn(query, points, k)
    inside = dist <= radius * radius
    first = idx[..., :1].expand_as(idx)
    return torch.where(inside, idx, first)


def group(query: torch.Tensor, points: torch.Tensor, feats: torch.Tensor | None, idx: torch.Tensor,
          ) -> torch.Tensor:
    """Grouped neighbourhoods (B, M, k, 3 [+C]): offsets to the query point, plus features."""
    rel = gather(points, idx) - query.unsqueeze(2)
    if feats is None:
        return rel
    return torch.cat([rel, gather(feats, idx)], dim=-1)


def three_nn_interpolate(query: torch.Tensor, points: torch.Tensor, feats: torch.Tensor,
                         k: int = 3, eps: float = 1e-8) -> torch.Tensor:
    """Features (B, M, C) at query points, inverse-distance weighted from the k nearest points."""
    idx, dist = knn(query, points, k)
    w = 1.0 / (dist + eps)
    w = w / w.sum(-1, keepdim=True)
    return (gather(feats, idx) * w.unsqueeze(-1)).sum(2)

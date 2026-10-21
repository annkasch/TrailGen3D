"""ForestGen3D denoiser: a PointNet++ U-Net on the noisy points, conditioned on the ALS points.

Follows CGNet from Point Diffusion-Refinement (Lyu et al. 2021), which ForestGen3D builds on:

- `ConditionNet` encodes the ALS points of a tree with PointNet++ set abstraction, giving point
  sets and features at 5 levels (the input points plus 4 abstraction levels).
- `PointNet2Denoiser` path: set abstraction (SA) down 4 levels on the noisy points x_t, then feature
  propagation (FP) back up to every point and a linear head predicting the noise epsilon.
  The timestep embedding is added to the features of every SA and FP module.
- `FeatureMapper` at every level: each noisy-level point attends over its k nearest condition
  points of the same level and the result modulates its features (FiLM: h * (1 + gamma) + beta).
  The FP stage reuses the attended condition features of its level for a second FiLM.

Pure PyTorch (see `ops`), so it runs on CUDA, MPS and CPU. Coordinates are expected in the
normalised tree box ([0, 1]^3 for clean points); radii are in those units.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from . import ops
from .diffusion import timestep_embedding


@dataclass
class ModelCfg:
    npoints: tuple = (1024, 256, 64, 16)          # SA centres per level (for N = 2048 points)
    radii: tuple = (0.1, 0.2, 0.4, 0.8)
    nsample: int = 32
    in_dim: int = 32                              # per-point features before the first SA
    sa_mlps: tuple = ((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512))
    fp_mlps: tuple = ((256, 256), (256, 256), (256, 128), (128, 128, 128))   # deepest first
    cond_npoints: tuple = (256, 128, 64, 16)      # for M = 512 ALS points
    cond_in_dim: int = 32
    cond_mlps: tuple = ((32, 32), (64, 64), (64, 64), (128, 128))
    cond_k: int = 8                               # condition neighbours per point in the mapper
    t_dim: int = 128
    fps: bool = True                              # False: take the first points as centres (faster)


def mlp(dims) -> nn.Sequential:
    """Shared per-point MLP on the last dimension: Linear, LayerNorm, SiLU per layer."""
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU()]
    return nn.Sequential(*layers)


def _centres(xyz: torch.Tensor, m: int, fps: bool) -> torch.Tensor:
    m = min(m, xyz.shape[1])
    if fps:
        return ops.gather(xyz, ops.farthest_point_sample(xyz, m))
    return xyz[:, :m]


class SetAbstraction(nn.Module):
    """Sample centres, group neighbours in a ball, shared MLP, max-pool."""

    def __init__(self, npoint: int, radius: float, nsample: int, in_ch: int, widths, t_dim: int = 0,
                 fps: bool = True):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.fps = npoint, radius, nsample, fps
        self.mlp = mlp([in_ch + 3, *widths])
        self.out_ch = widths[-1]
        self.t = nn.Linear(t_dim, self.out_ch) if t_dim else None

    def forward(self, xyz, feats, temb=None):
        centres = _centres(xyz, self.npoint, self.fps)
        idx = ops.ball_query(centres, xyz, self.radius, self.nsample)
        g = ops.group(centres, xyz, feats, idx)
        g = torch.cat([g[..., :3] / self.radius, g[..., 3:]], -1)
        h = self.mlp(g).max(2).values
        if self.t is not None:
            h = h + self.t(temb).unsqueeze(1)
        return centres, h


class FeaturePropagation(nn.Module):
    """Interpolate coarse features to the fine points, concatenate the skip features, MLP."""

    def __init__(self, coarse_ch: int, fine_ch: int, widths, t_dim: int):
        super().__init__()
        self.mlp = mlp([coarse_ch + fine_ch, *widths])
        self.out_ch = widths[-1]
        self.t = nn.Linear(t_dim, self.out_ch)

    def forward(self, xyz_fine, feat_fine, xyz_coarse, feat_coarse, temb):
        up = ops.three_nn_interpolate(xyz_fine, xyz_coarse, feat_coarse)
        return self.mlp(torch.cat([up, feat_fine], -1)) + self.t(temb).unsqueeze(1)


class FeatureMapper(nn.Module):
    """Attend from each noisy-level point over its k nearest condition points; FiLM its features."""

    def __init__(self, q_ch: int, c_ch: int, k: int, scale: float):
        super().__init__()
        self.k, self.scale = k, scale
        self.hidden = hidden = max(c_ch, 32)
        self.val = mlp([3 + c_ch, hidden, hidden])
        self.score = nn.Sequential(nn.Linear(hidden + q_ch, hidden), nn.SiLU(), nn.Linear(hidden, 1, bias=False))
        self.film = nn.Linear(hidden, 2 * q_ch)

    def forward(self, q_xyz, q_feat, c_xyz, c_feat):
        idx, _ = ops.knn(q_xyz, c_xyz, self.k)
        g = ops.group(q_xyz, c_xyz, c_feat, idx)
        g = torch.cat([g[..., :3] / self.scale, g[..., 3:]], -1)
        v = self.val(g)                                                   # (B, M, k, H)
        q = q_feat.unsqueeze(2).expand(-1, -1, v.shape[2], -1)
        w = torch.softmax(self.score(torch.cat([v, q], -1)), dim=2)
        c = (v * w).sum(2)                                                # (B, M, H)
        gamma, beta = self.film(c).chunk(2, -1)
        return q_feat * (1 + gamma) + beta, c


class ConditionNet(nn.Module):
    """PointNet++ encoder of the ALS points: [(xyz, feats)] for levels 0..4."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.inp = mlp([3, cfg.cond_in_dim])
        chans = [cfg.cond_in_dim]
        self.sa = nn.ModuleList()
        for npoint, r, widths in zip(cfg.cond_npoints, cfg.radii, cfg.cond_mlps):
            sa = SetAbstraction(npoint, r, cfg.nsample, chans[-1], widths, fps=True)
            self.sa.append(sa)
            chans.append(sa.out_ch)
        self.chans = chans

    def forward(self, cond: torch.Tensor):
        levels = [(cond, self.inp(cond))]
        for sa in self.sa:
            levels.append(sa(*levels[-1]))
        return levels


class Fg3dNet(nn.Module):
    """eps = net(x_t, t, cond): x_t (B, N, 3) noisy points, t (B,), cond (B, M, 3) ALS points.

    `cond` may also be the output of `encode_cond`, so sampling encodes the ALS only once.
    """

    def __init__(self, cfg: ModelCfg | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelCfg()
        self.temb = nn.Sequential(nn.Linear(cfg.t_dim, cfg.t_dim), nn.SiLU(), nn.Linear(cfg.t_dim, cfg.t_dim))
        self.cond_net = ConditionNet(cfg)
        cc = self.cond_net.chans
        scales = (cfg.radii[0] / 2, *cfg.radii)

        self.inp = nn.Linear(3, cfg.in_dim)
        self.inp_t = nn.Linear(cfg.t_dim, cfg.in_dim)
        chans = [cfg.in_dim]
        self.sa = nn.ModuleList()
        for npoint, r, widths in zip(cfg.npoints, cfg.radii, cfg.sa_mlps):
            sa = SetAbstraction(npoint, r, cfg.nsample, chans[-1], widths, cfg.t_dim, cfg.fps)
            self.sa.append(sa)
            chans.append(sa.out_ch)
        self.mappers = nn.ModuleList(FeatureMapper(chans[i], cc[i], cfg.cond_k, scales[i])
                                     for i in range(len(chans)))

        self.fp = nn.ModuleList()
        self.fp_film = nn.ModuleList()
        coarse = chans[-1]
        for j, widths in enumerate(cfg.fp_mlps):
            lvl = len(chans) - 2 - j
            fp = FeaturePropagation(coarse, chans[lvl], widths, cfg.t_dim)
            self.fp.append(fp)
            self.fp_film.append(nn.Linear(self.mappers[lvl].hidden, 2 * fp.out_ch))
            coarse = fp.out_ch
        self.head = nn.Linear(coarse, 3)

    def encode_cond(self, cond: torch.Tensor):
        return self.cond_net(cond)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond) -> torch.Tensor:
        levels_c = cond if isinstance(cond, list) else self.encode_cond(cond)
        temb = self.temb(timestep_embedding(t, self.cfg.t_dim))

        h = self.inp(x) + self.inp_t(temb).unsqueeze(1)
        h, c = self.mappers[0](x, h, *levels_c[0])
        levels = [(x, h)]
        ctx = [c]
        for i, sa in enumerate(self.sa, start=1):
            xyz, h = sa(*levels[-1], temb)
            h, c = self.mappers[i](xyz, h, *levels_c[i])
            levels.append((xyz, h))
            ctx.append(c)

        xyz_c, h = levels[-1]
        for j, (fp, film) in enumerate(zip(self.fp, self.fp_film)):
            lvl = len(levels) - 2 - j
            xyz_f, h_f = levels[lvl]
            h = fp(xyz_f, h_f, xyz_c, h, temb)
            gamma, beta = film(ctx[lvl]).chunk(2, -1)
            h = h * (1 + gamma) + beta
            xyz_c = xyz_f
        return self.head(h)


def tiny_cfg() -> ModelCfg:
    """Small widths for tests and quick checks (~1/10 of the parameters)."""
    return ModelCfg(npoints=(64, 32, 16, 8), nsample=8, in_dim=8,
                    sa_mlps=((16,), (16,), (32,), (32,)), fp_mlps=((32,), (16,), (16,), (16,)),
                    cond_npoints=(32, 16, 8, 4), cond_in_dim=8, cond_mlps=((8,), (8,), (16,), (16,)),
                    cond_k=4, t_dim=16)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

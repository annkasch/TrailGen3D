"""Stage 1 network (PyTorch): a small U-Net with three output heads.

Input (B, C, H, W) image of a window. Outputs at the same resolution:
  heat   (B, 2, H, W) logits of the stem heatmaps (>= small_dbh, < small_dbh)
  offset (B, 2, H, W) stem position minus pixel centre, in pixels
  size   (B, 2, H, W) log(dbh in m), height / height_scale
With 4 down-sampling steps each output pixel sees ~20 m around it, about one large crown.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


def _head(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cin, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(cin, cout, 1))


class StemUNet(nn.Module):
    def __init__(self, in_channels: int, base: int = 32, depth: int = 4):
        super().__init__()
        widths = [base * 2 ** min(i, 3) for i in range(depth + 1)]  # 32, 64, 128, 256, 256
        self.depth = depth
        self.down = nn.ModuleList([_block(in_channels, widths[0])] +
                                  [_block(widths[i], widths[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2)
                                 for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_block(2 * widths[i], widths[i]) for i in reversed(range(depth))])
        self.heat = _head(widths[0], 2)
        self.offset = _head(widths[0], 2)
        self.size = _head(widths[0], 2)
        # Start with a low stem probability everywhere (CenterNet trick, p = 0.1).
        nn.init.constant_(self.heat[-1].bias, -math.log((1 - 0.1) / 0.1))

    def forward(self, x: torch.Tensor) -> dict:
        H, W = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (-H) % m, (-W) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        out = {"heat": self.heat(x), "offset": self.offset(x), "size": self.size(x)}
        return {k: v[..., :H, :W] for k, v in out.items()}


def focal_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
               alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet focal loss, counted only where weight > 0, normalised by the number of stems."""
    logits, target, weight = logits.float(), target.float(), weight.float()
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 1.0).float() * weight
    neg = (target < 1.0).float() * weight
    pos_loss = -torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = -torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred, target, mask = pred.float(), target.float(), mask.float()
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1.0)


def stage1_loss(out: dict, batch: dict, w_offset: float = 1.0, w_size: float = 1.0) -> dict:
    reg = batch["reg_mask"].unsqueeze(1).expand_as(batch["offset"])
    losses = {
        "heat": focal_loss(out["heat"], batch["heat"], batch["weight"]),
        "offset": masked_l1(out["offset"], batch["offset"], reg),
        "size": masked_l1(out["size"], batch["size"], batch["size_mask"]),
    }
    losses["total"] = losses["heat"] + w_offset * losses["offset"] + w_size * losses["size"]
    return losses


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

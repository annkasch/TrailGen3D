"""Preview of a navigation world."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .traversability import CostMap


def save_world_preview(path: str | Path, cm: CostMap) -> Path:
    ext = (0, cm.grid.xmax - cm.grid.xmin, 0, cm.grid.ymax - cm.grid.ymin)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    ax = axes[0]
    im = ax.imshow(cm.cost, cmap="RdYlGn_r", vmin=0, vmax=1, extent=ext)
    fig.colorbar(im, ax=ax, label="cost (1 = blocked)")
    ax.set_title("Traversability")

    ax = axes[1]
    im = ax.imshow(cm.confidence, cmap="RdYlGn", vmin=0, vmax=1, extent=ext)
    fig.colorbar(im, ax=ax, label="share of ground measured nearby")
    ax.set_title("Ground confidence")

    for ax in axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    path = Path(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

"""Preview figure of a processed tile."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

from .grid import Grid


def _extent(grid: Grid):
    return (0, grid.xmax - grid.xmin, 0, grid.ymax - grid.ymin)


def save_preview(path: str | Path, dem_grid: Grid, dem: np.ndarray, chm_grid: Grid, chm: np.ndarray,
                 trees, shrub: np.ndarray, cost_grid: Grid, cost: np.ndarray,
                 confidence: np.ndarray) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    tx = np.array([t.x - dem_grid.xmin for t in trees]) if trees else np.array([])
    ty = np.array([t.y - dem_grid.ymin for t in trees]) if trees else np.array([])

    ax = axes[0, 0]
    shade = LightSource(azdeg=315, altdeg=45).hillshade(dem, dx=dem_grid.resolution, dy=dem_grid.resolution)
    ax.imshow(shade, cmap="gray", extent=_extent(dem_grid))
    im = ax.imshow(dem, cmap="terrain", alpha=0.45, extent=_extent(dem_grid))
    fig.colorbar(im, ax=ax, label="ground elevation (m)")
    ax.set_title("Ground model (DEM)")

    ax = axes[0, 1]
    im = ax.imshow(chm, cmap="Greens", extent=_extent(chm_grid))
    ax.scatter(tx, ty, s=6, c="crimson", label=f"{len(trees)} treetops")
    fig.colorbar(im, ax=ax, label="canopy height (m)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Canopy height and detected trees")

    ax = axes[1, 0]
    im = ax.imshow(shrub, cmap="YlOrBr", vmin=0, vmax=max(0.3, np.nanmax(shrub) if np.isfinite(shrub).any() else 0.3),
                   extent=_extent(chm_grid))
    fig.colorbar(im, ax=ax, label="share of returns 0.3-2 m above ground")
    ax.set_title("Low vegetation")

    ax = axes[1, 1]
    im = ax.imshow(cost, cmap="RdYlGn_r", vmin=0, vmax=1, extent=_extent(cost_grid))
    low_conf = confidence < 0.3
    if low_conf.any():
        ax.contour(np.flipud(low_conf.astype(float)), levels=[0.5], colors="k", linewidths=0.6,
                   extent=_extent(cost_grid))
    fig.colorbar(im, ax=ax, label="cost (1 = blocked)")
    ax.set_title("Traversability (black outline: little ground data)")

    for ax in axes.ravel():
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    path = Path(path)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

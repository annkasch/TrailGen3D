"""Preview of a Sky2Trail output: top views plus a side view showing measured vs generated."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

from .output import SOURCE_GENERATED, Sky2TrailOutput


def _extent(grid):
    return (0, grid.xmax - grid.xmin, 0, grid.ymax - grid.ymin)


def save_preview(out: Sky2TrailOutput, path: str | Path, strip_width: float = 4.0) -> Path:
    g, c = out.ground, out.canopy
    x0, y0 = g.grid.xmin, g.grid.ymin
    fig = plt.figure(figsize=(12, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7])

    ax = fig.add_subplot(gs[0, 0])
    shade = LightSource(azdeg=315, altdeg=45).hillshade(g.elevation, dx=g.grid.resolution, dy=g.grid.resolution)
    ax.imshow(shade, cmap="gray", extent=_extent(g.grid))
    im = ax.imshow(g.confidence, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.35, extent=_extent(g.grid))
    fig.colorbar(im, ax=ax, label="ground confidence (share measured)")
    ax.set_title("Ground surface")

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(c.chm, cmap="Greens", extent=_extent(c.grid))
    if out.stems:
        sx = np.array([s.x for s in out.stems]) - x0
        sy = np.array([s.y for s in out.stems]) - y0
        sd = np.array([s.dbh for s in out.stems])
        ax.scatter(sx, sy, s=8 + 200 * sd, c="saddlebrown", edgecolors="white", linewidths=0.4,
                   label=f"{len(out.stems)} stems (size = diameter)")
        ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=ax, label="canopy height (m)")
    ax.set_title("Canopy (measured) and stems")
    ymid = (g.grid.ymax - g.grid.ymin) / 2
    ax.axhspan(ymid - strip_width / 2, ymid + strip_width / 2, color="k", alpha=0.12)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(c.low_vegetation, cmap="YlOrBr", vmin=0, extent=_extent(c.grid))
    fig.colorbar(im, ax=ax, label="share of returns 0.3-2 m above ground")
    ax.set_title("Low vegetation (measured)")

    ax = fig.add_subplot(gs[1, 1])
    p = out.points
    gen = p.source == SOURCE_GENERATED
    hag = p.z - g.elevation_at(p.x, p.y)
    ax.hist([hag[~gen], hag[gen]], bins=40, orientation="horizontal", stacked=True,
            color=["0.6", "crimson"], label=["measured", "generated"], log=True)
    ax.set_xlabel("points (log)")
    ax.set_ylabel("height above ground (m)")
    ax.legend(fontsize=8)
    ax.set_title("Where the points are: measured vs generated")

    ax = fig.add_subplot(gs[2, :])
    sel = np.abs(p.y - (y0 + ymid)) < strip_width / 2
    m, gsel = sel & ~gen, sel & gen
    ax.scatter(p.x[m] - x0, p.z[m], s=1.5, c="0.55", label="measured (aerial)")
    ax.scatter(p.x[gsel] - x0, p.z[gsel], s=1.5, c="crimson", label=f"generated ({out.method})")
    gx = np.linspace(g.grid.xmin, g.grid.xmax, 400)
    ax.plot(gx - x0, g.elevation_at(gx, np.full_like(gx, y0 + ymid)), c="k", lw=0.8, label="ground surface")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("elevation (m)")
    ax.legend(loc="upper right", fontsize=8, markerscale=4)
    ax.set_title(f"Side view of the shaded strip ({strip_width:.0f} m wide)")

    path = Path(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

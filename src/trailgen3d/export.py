"""Exports for simulators and navigation stacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sky2trail.grid import Grid


def write_nav2_map(directory: str | Path, cost: np.ndarray, grid: Grid, name: str = "map") -> Path:
    """Nav2 map_server map (PGM + YAML) in a local frame whose origin is the grid's SW corner.

    Pixel value 0 = blocked, 254 = free, in between = graded cost (map mode 'scale').
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    c = np.nan_to_num(np.clip(cost, 0.0, 1.0), nan=1.0)
    img = np.round((1.0 - c) * 254).astype(np.uint8)
    pgm = directory / f"{name}.pgm"
    with open(pgm, "wb") as fh:
        fh.write(f"P5\n{grid.ncols} {grid.nrows}\n255\n".encode())
        fh.write(img.tobytes())
    yaml = directory / f"{name}.yaml"
    yaml.write_text(
        f"image: {name}.pgm\n"
        f"mode: scale\n"
        f"resolution: {grid.resolution}\n"
        f"origin: [0.0, 0.0, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n"
        f"# Local frame: x = easting - {grid.xmin}, y = northing - {grid.ymin}\n"
    )
    return yaml

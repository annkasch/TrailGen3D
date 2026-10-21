"""Terrestrial lidar (TLS) scans: read Leica .ptx and keep them as compact .npz.

A .ptx file holds one or more scans, each a gridded panorama:

    columns
    rows
    tx ty tz              scanner position          (3 lines)
    r11 r12 r13 ...       scanner axes              (3 lines)
    4 x 4 transform       scan -> project frame     (4 lines, row vectors: p' = [p 1] @ M)
    x y z intensity [r g b]   one line per grid cell, columns * rows lines; 0 0 0 = no return

The WFSI Fort Stewart scans are single BLK360 scans with an identity transform: points are in the
scanner frame (origin at the scanner, z up, heading arbitrary), so they have to be registered to
the ALS before use. Text .ptx is ~5x larger than needed; `ptx_to_npz` stores float32 xyz relative
to the scanner, uint8 intensity and RGB, and can voxel-downsample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TlsScan:
    xyz: np.ndarray          # (N, 3) float32, m, scanner at the origin of `transform`'s frame
    intensity: np.ndarray    # (N,) float32, 0..1
    rgb: np.ndarray | None   # (N, 3) uint8
    transform: np.ndarray    # (4, 4) float64, scan -> project frame (column-vector convention)
    source: str = ""

    def __len__(self) -> int:
        return len(self.xyz)

    def world(self) -> np.ndarray:
        """Points in the project frame (float64)."""
        return self.xyz.astype(np.float64) @ self.transform[:3, :3].T + self.transform[:3, 3]


def _read_header(fh) -> tuple[int, int, np.ndarray]:
    cols = int(fh.readline())
    rows = int(fh.readline())
    for _ in range(4):           # scanner position and axes (also in the transform)
        fh.readline()
    m = np.array([[float(v) for v in fh.readline().split()] for _ in range(4)])
    return cols, rows, m.T       # file stores row-vector form; return column-vector form


def read_ptx(path: str | Path, chunk_rows: int = 2_000_000) -> list[TlsScan]:
    """All scans in a .ptx file, without the empty grid cells (no return)."""
    path = Path(path)
    scans = []
    with open(path) as fh:
        while True:
            pos = fh.tell()
            first = fh.readline()
            if not first.strip():
                break
            fh.seek(pos)
            cols, rows, transform = _read_header(fh)
            n = cols * rows
            parts = []
            ncol = None
            left = n
            while left > 0:
                k = min(chunk_rows, left)
                lines = [fh.readline() for _ in range(k)]
                a = np.loadtxt(lines, dtype=np.float32, ndmin=2)
                ncol = a.shape[1]
                parts.append(a[np.any(a[:, :3] != 0, axis=1)])
                left -= k
            a = np.concatenate(parts) if parts else np.zeros((0, 7), np.float32)
            rgb = a[:, 4:7].astype(np.uint8) if ncol and ncol >= 7 else None
            scans.append(TlsScan(a[:, :3].copy(), a[:, 3].copy(), rgb, transform, path.stem))
    return scans


def voxel_downsample(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of one point per occupied voxel (the first one met)."""
    if voxel <= 0 or len(xyz) == 0:
        return np.arange(len(xyz))
    key = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64)
    span = key.max(0) + 1
    flat = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    _, idx = np.unique(flat, return_index=True)
    return np.sort(idx)


def save_npz(scan: TlsScan, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"rgb": scan.rgb} if scan.rgb is not None else {}
    np.savez_compressed(path, xyz=scan.xyz.astype(np.float32),
                        intensity=np.round(scan.intensity * 255).clip(0, 255).astype(np.uint8),
                        transform=scan.transform, source=np.array(scan.source), **extra)
    return path


def load_npz(path: str | Path) -> TlsScan:
    d = np.load(path)
    return TlsScan(d["xyz"], d["intensity"].astype(np.float32) / 255.0,
                   d["rgb"] if "rgb" in d else None, d["transform"], str(d["source"]))


def ptx_to_npz(src: str | Path, dest: str | Path, voxel: float = 0.0, max_range: float = 0.0) -> TlsScan:
    """Convert a single-scan .ptx to .npz, optionally voxel-downsampled and range-limited (m)."""
    scans = read_ptx(src)
    if len(scans) != 1:
        raise ValueError(f"{src}: {len(scans)} scans, expected 1")
    s = scans[0]
    keep = np.ones(len(s), bool)
    if max_range > 0:
        keep &= np.linalg.norm(s.xyz[:, :2], axis=1) <= max_range
    idx = np.flatnonzero(keep)
    idx = idx[voxel_downsample(s.xyz[idx], voxel)]
    s = TlsScan(s.xyz[idx], s.intensity[idx], None if s.rgb is None else s.rgb[idx], s.transform, s.source)
    save_npz(s, dest)
    return s

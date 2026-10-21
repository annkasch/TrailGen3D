"""Per-tree ALS/TLS pairs: file format, normalisation, and the training dataset.

One pair = one tree box. Points are stored normalised: the box footprint centre and the ground
height move to (0.5, 0.5, 0) and the largest box dimension is scaled to 1, so a tree fills
[0, 1]^3 (as in ForestGen3D). ALS and TLS share the transform; `norm` = (cx, cy, z0, scale)
maps back to world coordinates: world = (p - (0.5, 0.5, 0)) * scale + (cx, cy, z0).

File (.npz): als (M, 3) float32, tls (N, 3) float32 (N = 2048, random subset), norm (4,) float64,
plus the strings plot, tree, flight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

OFFSET = np.array([0.5, 0.5, 0.0])


@dataclass
class Pair:
    als: np.ndarray
    tls: np.ndarray
    norm: np.ndarray                  # (cx, cy, z0, scale)
    plot: str = ""
    tree: str = ""
    flight: str = ""


def box_norm(x0: float, y0: float, x1: float, y1: float, z0: float, z1: float) -> np.ndarray:
    """Normalisation (cx, cy, z0, scale) of a box from (x0, y0, z0) to (x1, y1, z1)."""
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, z0, max(x1 - x0, y1 - y0, z1 - z0)])


def to_unit(xyz: np.ndarray, norm: np.ndarray) -> np.ndarray:
    return ((np.asarray(xyz, np.float64) - norm[:3]) / norm[3] + OFFSET).astype(np.float32)


def to_world(p: np.ndarray, norm: np.ndarray) -> np.ndarray:
    return (np.asarray(p, np.float64) - OFFSET) * norm[3] + norm[:3]


def fps_numpy(xyz: np.ndarray, m: int, seed: int | None = 0) -> np.ndarray:
    """Indices of m farthest-point samples (all indices, shuffled, if there are fewer points)."""
    n = len(xyz)
    rng = np.random.default_rng(seed)
    if n <= m:
        return rng.permutation(n)
    idx = np.empty(m, np.int64)
    idx[0] = rng.integers(n)
    d = np.full(n, np.inf)
    for i in range(1, m):
        d = np.minimum(d, ((xyz - xyz[idx[i - 1]]) ** 2).sum(1))
        idx[i] = d.argmax()
    return idx


def subsample(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of a random subset of m of n points (all, shuffled, if n <= m).

    Random rather than farthest-point sampling: FPS spreads points evenly in space and leaves a
    thin, densely scanned stem with almost none, which would lose the DBH information.
    """
    return rng.permutation(n) if n <= m else rng.choice(n, m, replace=False)


def resample(xyz: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Exactly m points: a random subset, or all points plus random repeats."""
    n = len(xyz)
    if n == 0:
        return np.zeros((m, 3), np.float32)
    if n >= m:
        return xyz[rng.choice(n, m, replace=False)]
    return np.concatenate([xyz, xyz[rng.integers(0, n, m - n)]])


def save_pair(p: Pair, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, als=p.als.astype(np.float32), tls=p.tls.astype(np.float32),
                        norm=np.asarray(p.norm, np.float64), plot=np.array(p.plot),
                        tree=np.array(p.tree), flight=np.array(p.flight))
    return path


def load_pair(path: str | Path) -> Pair:
    d = np.load(path)
    return Pair(d["als"], d["tls"], d["norm"], str(d["plot"]), str(d["tree"]), str(d["flight"]))


def list_pairs(root: str | Path, split: str | None = None) -> list[Path]:
    """Pair files under root; with a split name, only those listed in root/split.json."""
    root = Path(root)
    if split is None:
        return sorted(root.glob("**/*.npz"))
    import json
    names = json.loads((root / "split.json").read_text())[split]
    return [root / n for n in names]


class PairDataset:
    """Torch dataset of pairs -> (tls (N, 3), als (M, 3)) float32 tensors.

    augment: random rotation about the vertical tree axis, and point jitter.
    thin: ALS keep fraction range (lo, hi) to simulate sparser lidar (e.g. (0.2, 1.0)).
    """

    def __init__(self, files, n_als: int = 512, n_tls: int = 2048, augment: bool = False,
                 jitter: float = 0.002, thin: tuple[float, float] | None = None, seed: int = 0,
                 cache: bool = True):
        self.files = [Path(f) for f in files]
        self.n_als, self.n_tls = n_als, n_tls
        self.augment, self.jitter, self.thin = augment, jitter, thin
        self.seed = seed
        self.epoch = 0
        self._cache = {} if cache else None

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, i):
        if self._cache is None:
            return load_pair(self.files[i])
        if i not in self._cache:
            self._cache[i] = load_pair(self.files[i])
        return self._cache[i]

    def __getitem__(self, i):
        import torch

        p = self._load(i)
        if self.augment or self.thin:
            rng = np.random.default_rng()       # per-worker seeding via worker_init_fn
        else:
            rng = np.random.default_rng(self.seed + i)
        als, tls = p.als, p.tls
        if self.thin:
            keep = rng.random(len(als)) < rng.uniform(*self.thin)
            if keep.sum() >= 8:
                als = als[keep]
        als = resample(als, self.n_als, rng)
        tls = resample(tls, self.n_tls, rng)
        if self.augment:
            a = rng.uniform(0, 2 * np.pi)
            rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32)
            als = (als - OFFSET) @ rot.T + OFFSET
            tls = (tls - OFFSET) @ rot.T + OFFSET
            if self.jitter:
                als = als + rng.normal(0, self.jitter, als.shape)
                tls = tls + rng.normal(0, self.jitter, tls.shape)
        return torch.as_tensor(tls, dtype=torch.float32), torch.as_tensor(als, dtype=torch.float32)

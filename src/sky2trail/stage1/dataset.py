"""Training samples: one per patch, stored compactly.

A sample holds the input image of the window, the known stems inside the window (pixel
coordinates), the labelled area, and whether small stems are labelled. Targets are built
on the fly after augmentation, so the files stay small and augmentation stays exact.

Synthetic data comes from `make_hard_scene`; real plots (USGS Sierra, NEON) will produce
the same sample format from their own loaders.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..config import Sky2TrailConfig, Stage1Cfg
from ..io import PointCloud
from ..patches import Patch, tile_patches
from ..terrain import build_terrain
from .augment import random_transform
from .features import channel_names, rasterize, window_grid
from .targets import make_targets, world_to_pixel


def window_samples(pc: PointCloud, hag: np.ndarray, patches: list[Patch], truth: np.ndarray | None,
                   label_fn, small_labelled: bool, cfg: Sky2TrailConfig, source: str = "") -> list[dict]:
    """Samples for the given patches.

    truth: (N, 4) world x, y, dbh, height of known stems (None for prediction only).
    label_fn(x, y) -> bool array: where stems are fully labelled (field plot, synthetic scene).
    """
    s1 = cfg.stage1
    is_ground = pc.classification == cfg.ground.classification
    out = []
    for p in patches:
        sel = p.in_input(pc.x, pc.y)
        grid = window_grid(p.input_bounds, s1.resolution)
        image = rasterize(pc.x[sel], pc.y[sel], hag[sel], is_ground[sel], pc.return_number[sel], grid, s1)
        xs, ys = grid.cell_centers()
        mask = label_fn(xs, ys) if label_fn is not None else np.zeros(grid.shape, bool)
        if truth is not None and len(truth):
            t = truth[p.in_input(truth[:, 0], truth[:, 1])]
            stems = np.column_stack([world_to_pixel(t[:, 0], t[:, 1], grid), t[:, 2], t[:, 3]])
        else:
            stems = np.zeros((0, 4))
        m = int(round(p.margin / s1.resolution))
        n = int(round(p.core_size / s1.resolution))
        out.append({"image": image.astype(np.float16), "stems": stems.astype(np.float32),
                    "mask": mask.astype(bool), "small_labelled": bool(small_labelled),
                    "core_px": np.array([m, m, m + n, m + n], np.int32),
                    "bounds": np.array(p.input_bounds), "source": source})
    return out


def scene_samples(scene, cfg: Sky2TrailConfig, source: str) -> list[dict]:
    """All patches of a synthetic scene; everything inside the scene is labelled."""
    pc = scene.points
    terrain = build_terrain(pc, cfg.grid.dem_resolution, cfg.ground)
    hag = terrain.height_above_ground(pc.x, pc.y, pc.z)
    size = scene.size
    patches = tile_patches((0, 0, size, size), cfg.patches.core_size, cfg.patches.margin)
    t = scene.trees
    truth = np.column_stack([t[:, 0], t[:, 1], t[:, 4], t[:, 2]])

    def inside(x, y):
        return (x >= 0) & (x <= size) & (y >= 0) & (y <= size)

    return window_samples(pc, hag, patches, truth, inside, True, cfg, source)


def save_samples(samples: list[dict], path: str | Path, cfg: Stage1Cfg, info: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.array([len(s["stems"]) for s in samples])
    meta = {"stage1": asdict(cfg), "channels": channel_names(cfg), **(info or {})}
    np.savez_compressed(
        path,
        image=np.stack([s["image"] for s in samples]),
        stems=np.concatenate([s["stems"] for s in samples]) if counts.sum() else np.zeros((0, 4), np.float32),
        stem_index=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        mask=np.stack([s["mask"] for s in samples]),
        small_labelled=np.array([s["small_labelled"] for s in samples]),
        core_px=np.stack([s["core_px"] for s in samples]),
        bounds=np.stack([s["bounds"] for s in samples]),
        source=np.array([s["source"] for s in samples]),
        meta=np.array(json.dumps(meta)),
    )
    return path


class StemSamples:
    """Samples from a .npz written by save_samples; get(i) returns image and targets."""

    def __init__(self, path: str | Path):
        d = np.load(path)
        self.image = d["image"]
        self.stems = d["stems"]
        self.stem_index = d["stem_index"]
        self.mask = d["mask"]
        self.small_labelled = d["small_labelled"]
        self.core_px = d["core_px"]
        self.bounds = d["bounds"]
        self.source = d["source"]
        self.meta = json.loads(str(d["meta"]))
        self.cfg = Stage1Cfg(**self.meta["stage1"])

    def __len__(self) -> int:
        return len(self.image)

    @property
    def n_channels(self) -> int:
        return self.image.shape[1]

    def stems_of(self, i: int) -> np.ndarray:
        return self.stems[self.stem_index[i]:self.stem_index[i + 1]]

    def get(self, i: int, rng: np.random.Generator | None = None) -> dict:
        image = self.image[i].astype(np.float32)
        stems = self.stems_of(i)
        mask = self.mask[i]
        if rng is not None:
            image, stems, (mask,) = random_transform(rng, image, stems, [mask])
        t = make_targets(stems, image.shape[1:], mask, bool(self.small_labelled[i]), self.cfg)
        t["image"] = image
        return t


def make_synthetic_dataset(out_dir: str | Path, cfg: Sky2TrailConfig, n_train: int = 20, n_val: int = 4,
                           size: float = 100.0, min_density: float = 2.0, max_density: float = 15.0,
                           seed: int = 0, log=print) -> dict:
    """Train/val files from hard synthetic scenes at random densities (log-uniform).

    Scene seeds: train seed..seed+n_train-1, val 1000+seed..; held-out test scenes for
    evaluation use 2000+ (see predict.evaluate_synthetic), so the three never overlap.
    """
    from ..synthetic import make_hard_scene

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    paths = {}
    for split, n, base in (("train", n_train, seed), ("val", n_val, 1000 + seed)):
        samples, densities = [], []
        for k in range(n):
            d = float(np.exp(rng.uniform(np.log(min_density), np.log(max_density))))
            scene = make_hard_scene(size=size, pulses_per_m2=d, seed=base + k)
            samples += scene_samples(scene, cfg, f"synthetic seed {base + k}")
            densities.append(round(d, 2))
            log(f"{split}: scene {k + 1}/{n}, {d:.1f} pulses/m2, {len(scene.trees)} trees")
        paths[split] = save_samples(samples, out_dir / f"{split}.npz", cfg.stage1,
                                    {"split": split, "densities": densities, "scene_size": size})
        log(f"wrote {paths[split]} ({len(samples)} samples)")
    return paths

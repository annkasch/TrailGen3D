"""Build per-tree ALS/TLS pairs from registered WFSI scans.

Per plot: segment crowns in the ALS canopy height model (`trees.extract_crowns`), and for each
crown with its treetop near the scanner take the ALS and the registered TLS points over the
crown footprint (label cells, grown by `grow` m so the stem and crown edge stay in), from the
ground to the treetop. Points are normalised to the footprint's bounding box (`pairs.box_norm`).
ForestGen3D uses bounding boxes from a BEV detector; footprints keep fewer points of neighbours.

    sky2trail fg3d pairs --root data/fort_stewart --out data/fg3d_pairs
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import TreesCfg
from ..grid import Grid
from ..register import Registration, apply, ground_surface, sample_raster
from ..trees import canopy_height_model, extract_crowns
from .metrics import envelope_containment
from .pairs import Pair, box_norm, save_pair, subsample, to_unit


# Crown segmentation for tall, touching southern pines: the stage-1 defaults (15 % saddle,
# 10 m radius) merge neighbours into 11-18 m crowns; these match the survey count on E023 (22).
PINE_TREES = TreesCfg(max_crown_radius=6.0, min_saddle_drop=0.8, min_saddle_fraction=0.06)


@dataclass
class BuildCfg:
    resolution: float = 0.5          # CHM cell size, m
    max_dist: float = 20.0           # treetop within this distance of the scanner, m
    tls_radius: float = 25.0         # TLS points used, m from the scanner
    grow: float = 0.5                # footprint growth, m
    min_height: float = 5.0          # trees lower than this are skipped, m
    min_tls: int = 3000              # TLS points per tree (before subsampling)
    min_als: int = 30
    n_tls: int = 2048
    seed: int = 0


def plot_pairs(scan_xyz: np.ndarray, als: np.ndarray, reg: Registration, plot: str, flight: str,
               cfg: BuildCfg | None = None, trees_cfg: TreesCfg | None = None) -> tuple[list[Pair], list[dict]]:
    """Pairs for one registered scan; plus per-tree info (height, point counts, containment)."""
    cfg = cfg or BuildCfg()
    trees_cfg = trees_cfg or PINE_TREES
    rng = np.random.default_rng(cfg.seed)
    T = reg.matrix()
    c = T[:3, 3]
    tls = scan_xyz[np.hypot(scan_xyz[:, 0], scan_xyz[:, 1]) <= cfg.tls_radius]
    tls = apply(T, tls)

    half = cfg.max_dist + 10
    a = als[np.hypot(als[:, 0] - c[0], als[:, 1] - c[1]) <= half * 1.5]
    gmask = a[:, 3] == 2 if (a[:, 3] == 2).mean() > 0.01 else None
    ground = ground_surface(a[:, :3], c[:2], half, cfg.resolution, ground_mask=gmask)
    grid = Grid.from_bounds(c[0] - half, c[1] - half, c[0] + half, c[1] + half, cfg.resolution)
    # ground raster -> the Grid's north-up layout (row 0 = north)
    gx, gy = grid.cell_centers()
    dem = sample_raster(ground, np.c_[gx.ravel(), gy.ravel()], c[:2], half, cfg.resolution).reshape(grid.shape)
    hag_a = a[:, 2] - grid.sample(dem, a[:, 0], a[:, 1])
    chm = canopy_height_model(grid, a[:, 0], a[:, 1], hag_a)
    crowns, labels = extract_crowns(grid, chm, a[:, 0], a[:, 1], hag_a, trees_cfg)

    grow = max(1, int(round(cfg.grow / cfg.resolution)))
    lab_a = _labels_at(grid, labels, a[:, 0], a[:, 1])
    lab_t = _labels_at(grid, labels, tls[:, 0], tls[:, 1])
    pairs, info = [], []
    for cr in crowns:
        if cr.height < cfg.min_height or np.hypot(cr.x - c[0], cr.y - c[1]) > cfg.max_dist:
            continue
        foot = ndimage.binary_dilation(labels == cr.id, iterations=grow)
        rows, cols = np.nonzero(foot)
        x0 = grid.xmin + cols.min() * cfg.resolution
        x1 = grid.xmin + (cols.max() + 1) * cfg.resolution
        y1 = grid.ymax - rows.min() * cfg.resolution
        y0 = grid.ymax - (rows.max() + 1) * cfg.resolution
        z0 = float(dem[foot].min())
        in_a = _in_foot(foot, grid, a[:, 0], a[:, 1], lab_a, cr.id)
        in_t = _in_foot(foot, grid, tls[:, 0], tls[:, 1], lab_t, cr.id)
        pa = a[in_a & (a[:, 2] >= z0 - 0.5), :3]
        pt = tls[in_t & (tls[:, 2] >= z0 - 0.5)]
        z1 = float(max(pa[:, 2].max() if len(pa) else z0, z0 + cr.height))
        pt = pt[pt[:, 2] <= z1 + 1.0]
        if len(pt) < cfg.min_tls or len(pa) < cfg.min_als:
            continue
        norm = box_norm(x0, y0, x1, y1, z0, z1)
        als_u = to_unit(pa, norm)
        tls_u = to_unit(pt, norm)
        epc = envelope_containment(tls_u[subsample(len(tls_u), 5000, rng)], als_u)
        tls_u = tls_u[subsample(len(tls_u), cfg.n_tls, rng)]
        tree = f"{plot}_{flight}_{cr.id:03d}"
        pairs.append(Pair(als_u, tls_u, norm, plot, tree, flight))
        info.append({"tree": tree, "plot": plot, "flight": flight, "height_als": round(cr.height, 2),
                     "height_tls": round(float(np.percentile(pt[:, 2], 99.5) - z0), 2),
                     "dist": round(float(np.hypot(cr.x - c[0], cr.y - c[1])), 1),
                     "n_als": int(len(pa)), "n_tls": int(len(pt)), "tls_in_als_hull": round(epc, 3)})
    return pairs, info


def _labels_at(grid: Grid, labels: np.ndarray, x, y) -> np.ndarray:
    row, col, ok = grid.index(x, y)
    out = np.zeros(len(x), np.int64)
    out[ok] = labels[row[ok], col[ok]]
    return out


def _in_foot(foot: np.ndarray, grid: Grid, x, y, lab, crown_id: int) -> np.ndarray:
    """Points over the grown footprint, excluding those inside another crown's label."""
    row, col, ok = grid.index(x, y)
    m = np.zeros(len(x), bool)
    m[ok] = foot[row[ok], col[ok]]
    return m & ((lab == 0) | (lab == crown_id))


def split_plots(plots: list[str], val: float = 0.15, test: float = 0.15, seed: int = 0) -> dict:
    """Assign whole plots to train / val / test (never split a plot)."""
    rng = np.random.default_rng(seed)
    p = list(rng.permutation(sorted(set(plots))))
    n_test = max(1, int(round(test * len(p)))) if len(p) >= 3 else 0
    n_val = max(1, int(round(val * len(p)))) if len(p) >= 3 else 0
    return {"test": sorted(p[:n_test]), "val": sorted(p[n_test:n_test + n_val]),
            "train": sorted(p[n_test + n_val:])}


def build_pairs(root: str | Path, out: str | Path, flights=("pre",), cfg: BuildCfg | None = None,
                only_ok: bool = True, trees_cfg: TreesCfg | None = None, log=print) -> dict:
    """Pairs for every registered scan under root; writes pair files, split.json and trees.json."""
    from ..tls import load_npz

    cfg = cfg or BuildCfg()
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = {p["name"]: p for p in json.loads((root / "manifest.json").read_text())["plots"]}
    files: dict[str, list[str]] = {}
    infos = []
    for flight in flights:
        for rj in sorted((root / "registration" / flight).glob("*.json")):
            reg = Registration.load(rj)
            plot = next((n for n in table if rj.stem.startswith(f"FTST{n}")), rj.stem)
            if only_ok and not reg.ok:
                log(f"{plot} {flight}: registration flagged, skipped")
                continue
            als = np.load(root / "als_crops" / flight / f"{plot}.npy")
            scan = load_npz(root / "tls_npz" / flight / f"{rj.stem}.npz")
            pairs, info = plot_pairs(scan.xyz, als, reg, plot, flight, cfg, trees_cfg)
            for p in pairs:
                name = f"{p.tree}.npz"
                save_pair(p, out / name)
                files.setdefault(plot, []).append(name)
            infos += info
            epc = np.nanmean([i["tls_in_als_hull"] for i in info]) if info else float("nan")
            log(f"{plot} {flight}: {len(pairs)} trees, TLS in ALS hull {epc:.2f}")
    split_by_plot = split_plots(list(files), seed=cfg.seed)
    split = {k: [n for pl in v for n in files[pl]] for k, v in split_by_plot.items()}
    (out / "split.json").write_text(json.dumps(split | {"plots": split_by_plot}, indent=1))
    (out / "trees.json").write_text(json.dumps({"cfg": asdict(cfg), "trees": infos}, indent=1))
    return {"trees": len(infos), **{k: len(v) for k, v in split.items()}}

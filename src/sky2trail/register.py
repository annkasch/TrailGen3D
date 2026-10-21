"""Register single terrestrial scans (TLS) to airborne lidar (ALS).

The Fort Stewart BLK360 scans are in the scanner frame: levelled (the scanner's tilt sensor makes
z vertical) but with an unknown heading and position. The plot GPS point gives the position to
within a few metres. So registration solves 4 degrees of freedom, heading and a 3D shift:

1. Height above ground for both clouds: ALS from its ground class, TLS from the lowest points
   per cell (the scanner sees the ground all around it).
2. Coarse heading and xy: canopy-height images (highest point above ground per 0.5 m cell) of the
   TLS, rotated in 1 degree steps, are matched to the ALS image by masked normalised
   cross-correlation (FFT), over shifts within `search` m of the GPS point.
3. Vertical shift from the ground surfaces, then point-to-plane ICP (heading + shift only) of
   TLS points against ALS points, with a shrinking correspondence distance.

Result: a 4x4 transform, scanner frame -> UTM (column vectors: world = T @ [p, 1]).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .tls import voxel_downsample


@dataclass
class RegCfg:
    resolution: float = 0.5          # raster cell size for ground and canopy images, m
    tls_radius: float = 25.0         # TLS points used, m from the scanner
    search: float = 10.0             # xy search radius around the GPS point, m
    heading_step: float = 1.0        # degrees
    voxel: float = 0.2               # TLS / ALS thinning for ICP, m
    icp_iters: int = 30
    icp_dist: tuple = (2.0, 1.0, 0.5, 0.3)   # correspondence distance schedule, m
    min_height: float = 1.0          # canopy image: heights below are 0


@dataclass
class Registration:
    transform: list                  # 4x4, scanner frame -> world
    heading_deg: float
    shift: list                      # x, y, z of the scanner in world coordinates
    ncc: float                       # canopy-image correlation at the coarse solution
    ncc_second: float                # best correlation at headings > 20 deg away (ambiguity)
    icp_rms: float                   # point-to-plane RMS of inliers at the last ICP step, m
    inlier_fraction: float           # TLS points within 0.3 m of the ALS surface
    ok: bool

    def matrix(self) -> np.ndarray:
        return np.asarray(self.transform)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Registration":
        return cls(**json.loads(Path(path).read_text()))


def rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def make_transform(heading_deg: float, shift) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot_z(heading_deg)
    T[:3, 3] = shift
    return T


def apply(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return np.asarray(xyz, np.float64) @ T[:3, :3].T + T[:3, 3]


# --- rasters -------------------------------------------------------------------------------

def _raster(xy: np.ndarray, values: np.ndarray, centre, half: float, res: float, stat: str) -> np.ndarray:
    """Square raster of side 2*half around centre; 'max' / 'min' per cell, NaN where empty."""
    n = int(round(2 * half / res))
    ij = np.floor((xy - (np.asarray(centre) - half)) / res).astype(np.int64)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    flat = ij[ok, 1] * n + ij[ok, 0]          # row = y, col = x
    out = np.full(n * n, np.nan)
    v = values[ok]
    if stat == "max":
        order = np.argsort(v)
    else:
        order = np.argsort(-v)
    out[flat[order]] = v[order]               # last write wins: max (or min)
    return out.reshape(n, n)


def _fill(r: np.ndarray) -> np.ndarray:
    known = np.isfinite(r)
    if known.all() or not known.any():
        return np.nan_to_num(r)
    idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    return r[tuple(idx)]


def ground_surface(xyz: np.ndarray, centre, half: float, res: float, max_rise: float = 0.5,
                   ground_mask: np.ndarray | None = None) -> np.ndarray:
    """Ground elevation raster: class-2 points if given, else lowest points per cell, filtered."""
    pts = xyz if ground_mask is None else xyz[ground_mask]
    low = _raster(pts[:, :2], pts[:, 2], centre, half, res, "min")
    # drop cells well above the local floor (low vegetation with no ground under it)
    floor = ndimage.minimum_filter(_fill(low), size=7, mode="nearest")
    ref = ndimage.uniform_filter(floor, size=7, mode="nearest")
    low[(low - ref) > max_rise] = np.nan
    return ndimage.median_filter(_fill(low), size=3, mode="nearest")


def sample_raster(r: np.ndarray, xy: np.ndarray, centre, half: float, res: float) -> np.ndarray:
    col = (xy[:, 0] - (centre[0] - half)) / res - 0.5
    row = (xy[:, 1] - (centre[1] - half)) / res - 0.5
    return ndimage.map_coordinates(r, [row, col], order=1, mode="nearest")


def canopy_image(xy: np.ndarray, hag: np.ndarray, centre, half: float, res: float, min_height: float):
    """Highest point above ground per cell (0 below min_height) and a mask of observed cells."""
    chm = _raster(xy, hag, centre, half, res, "max")
    seen = np.isfinite(chm)
    chm = np.where(seen, chm, 0.0)
    chm[chm < min_height] = 0.0
    return chm, seen


# --- coarse search ---------------------------------------------------------------------------

def _masked_ncc(A: np.ndarray, T: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of template T (valid where M) at every offset in A.

    'valid'-style output: entry (i, j) compares T with A[i:i+h, j:j+w].
    """
    from scipy.signal import fftconvolve

    M = M.astype(float)
    n = M.sum()
    T0 = np.where(M > 0, T - T[M > 0].mean(), 0.0)
    tn = np.sqrt((T0 ** 2).sum())
    flip = lambda x: x[::-1, ::-1]
    num = fftconvolve(A, flip(T0), mode="valid")
    s1 = fftconvolve(A, flip(M), mode="valid")
    s2 = fftconvolve(A * A, flip(M), mode="valid")
    var = np.maximum(s2 - s1 ** 2 / n, 1e-9)
    return num / (tn * np.sqrt(var) + 1e-12)


def coarse_search(tls_xy: np.ndarray, tls_hag: np.ndarray, als_xy: np.ndarray, als_hag: np.ndarray,
                  gps, cfg: RegCfg) -> tuple[float, np.ndarray, float, float]:
    """Best (heading, scanner xy in world, ncc, best ncc at headings > 20 deg away)."""
    res, R = cfg.resolution, cfg.tls_radius
    A, _ = canopy_image(als_xy, als_hag, gps, R + cfg.search, res, cfg.min_height)
    results = {}

    def at(h):
        h = float(h) % 360
        if h not in results:
            results[h] = _try(h)
        return results[h]

    def _try(h):
        xy = tls_xy @ rot_z(h)[:2, :2].T
        T, seen = canopy_image(xy, tls_hag, (0.0, 0.0), R, res, cfg.min_height)
        disk = np.hypot(*np.meshgrid((np.arange(T.shape[1]) + 0.5) * res - R,
                                     (np.arange(T.shape[0]) + 0.5) * res - R)) <= R
        ncc = _masked_ncc(A, T, seen & disk)
        i, j = np.unravel_index(np.argmax(ncc), ncc.shape)
        # offset (i, j) puts the template's corner at A's cell (i, j); template centre = scanner
        x = gps[0] - (R + cfg.search) + j * res + R
        y = gps[1] - (R + cfg.search) + i * res + R
        return (h, float(ncc[i, j]), np.array([x, y]))

    # coarse 5 degree sweep, then refine around the 3 best headings
    coarse = max(cfg.heading_step, 5.0)
    for h in np.arange(0, 360, coarse):
        at(h)
    for h0, _, _ in sorted(results.values(), key=lambda r: -r[1])[:3]:
        for h in np.arange(h0 - coarse, h0 + coarse + 1e-9, cfg.heading_step):
            at(h)
    best = max(results.values(), key=lambda r: r[1])
    far = [r[1] for r in results.values() if abs((r[0] - best[0] + 180) % 360 - 180) > 20]
    return best[0], best[2], best[1], max(far) if far else float("nan")


# --- ICP -------------------------------------------------------------------------------------

def normals(xyz: np.ndarray, k: int = 12) -> np.ndarray:
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=min(k, len(xyz)))
    nb = xyz[idx] - xyz[idx].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vec = np.linalg.eigh(cov)
    return vec[:, :, 0]


def icp_4dof(src: np.ndarray, dst: np.ndarray, dst_n: np.ndarray, T0: np.ndarray, cfg: RegCfg
             ) -> tuple[np.ndarray, float]:
    """Point-to-plane ICP over heading and shift. src in scanner frame, dst in world."""
    tree = cKDTree(dst)
    T = T0.copy()
    rms = float("nan")
    per = max(1, cfg.icp_iters // len(cfg.icp_dist))
    for dmax in cfg.icp_dist:
        for _ in range(per):
            p = apply(T, src)
            d, j = tree.query(p, distance_upper_bound=dmax)
            ok = np.isfinite(d)
            if ok.sum() < 50:
                break
            p, q, n = p[ok], dst[j[ok]], dst_n[j[ok]]
            c = T[:2, 3]
            # rotation about the vertical through the scanner: dp = dθ * (-(y - cy), x - cx, 0)
            J = np.c_[n[:, 0] * -(p[:, 1] - c[1]) + n[:, 1] * (p[:, 0] - c[0]), n]
            r = ((q - p) * n).sum(1)
            x = np.linalg.lstsq(J, r, rcond=None)[0]
            dT = make_transform(np.rad2deg(x[0]), [0, 0, 0])
            # rotate about the scanner position, then shift
            T[:3, :3] = dT[:3, :3] @ T[:3, :3]
            T[:3, 3] = T[:3, 3] + x[1:]
            rms = float(np.sqrt(np.mean((r - J @ x) ** 2)))
            if abs(x[0]) < 1e-5 and np.abs(x[1:]).max() < 1e-4:
                break
    return T, rms


# --- driver ----------------------------------------------------------------------------------

def register_scan(tls_xyz: np.ndarray, als: np.ndarray, gps, cfg: RegCfg | None = None) -> Registration:
    """Register a levelled TLS scan (scanner frame) to ALS points near the plot GPS point.

    als: (N, 3+) x, y, z [, classification]; class 2 = ground if present.
    """
    cfg = cfg or RegCfg()
    gps = np.asarray(gps, float)[:2]
    res, R = cfg.resolution, cfg.tls_radius

    tls = tls_xyz[np.hypot(tls_xyz[:, 0], tls_xyz[:, 1]) <= R].astype(np.float64)
    tls = tls[voxel_downsample(tls, 0.05)]
    t_ground = ground_surface(tls, (0, 0), R, res)
    t_hag = tls[:, 2] - sample_raster(t_ground, tls[:, :2], (0, 0), R, res)

    half = R + cfg.search + 5
    ax = als[:, :3].astype(np.float64)
    gmask = als[:, 3] == 2 if als.shape[1] > 3 and (als[:, 3] == 2).mean() > 0.01 else None
    a_ground = ground_surface(ax, gps, half, res, ground_mask=gmask)
    a_hag = ax[:, 2] - sample_raster(a_ground, ax[:, :2], gps, half, res)

    heading, xy, ncc, ncc2 = coarse_search(tls[:, :2], t_hag, ax[:, :2], a_hag, gps, cfg)

    # vertical: median difference of the ground surfaces under TLS ground points
    T = make_transform(heading, [xy[0], xy[1], 0.0])
    g = tls[np.abs(t_hag) < 0.15]
    gw = apply(T, g)
    dz = float(np.median(sample_raster(a_ground, gw[:, :2], gps, half, res) - gw[:, 2]))
    T[2, 3] = dz

    src = tls[voxel_downsample(tls, cfg.voxel)]
    near = np.hypot(ax[:, 0] - xy[0], ax[:, 1] - xy[1]) <= R + 5
    dst = ax[near]
    dst = dst[voxel_downsample(dst, cfg.voxel / 2)]
    T, rms = icp_4dof(src, dst, normals(dst), T, cfg)

    d, _ = cKDTree(dst).query(apply(T, src), distance_upper_bound=0.3)
    # only TLS points that ALS can see (ground and upper canopy) can be inliers; report the fraction
    inl = float(np.isfinite(d).mean())
    heading = float(np.rad2deg(np.arctan2(T[1, 0], T[0, 0])) % 360)
    ok = bool(ncc > 0.3 and (not np.isfinite(ncc2) or ncc - ncc2 > 0.05) and rms < 0.25)
    return Registration(T.tolist(), heading, T[:3, 3].tolist(), ncc, ncc2, rms, inl, ok)


def save_overlay(tls_xyz: np.ndarray, als: np.ndarray, reg: Registration, path: str | Path,
                 radius: float = 25.0) -> Path:
    """Top view (points 1-4 m above the lowest ALS, i.e. stems and low crowns) and a side slice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = reg.matrix()
    c = T[:3, 3]
    t = tls_xyz[np.hypot(tls_xyz[:, 0], tls_xyz[:, 1]) <= radius]
    t = apply(T, t[voxel_downsample(t, 0.1)])
    a = als[np.hypot(als[:, 0] - c[0], als[:, 1] - c[1]) <= radius, :3]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
    res = 0.5
    chm, _ = canopy_image(a[:, :2], a[:, 2] - np.percentile(a[:, 2], 2), c[:2], radius, res, 1.0)
    ax[0].imshow(chm, origin="lower", cmap="Greens", extent=(-radius, radius, -radius, radius))
    gz = np.percentile(t[:, 2], 2)
    stems = t[(t[:, 2] > gz + 1) & (t[:, 2] < gz + 3)]
    ax[0].scatter(stems[:, 0] - c[0], stems[:, 1] - c[1], s=0.2, c="k")
    ax[0].set(aspect="equal", title=f"top: ALS canopy height, TLS points 1-3 m (heading {reg.heading_deg:.1f}°)")
    sl = lambda q: np.abs(q[:, 1] - c[1]) < 1.5
    ax[1].scatter(t[sl(t), 0] - c[0], t[sl(t), 2], s=0.1, c="0.5")
    ax[1].scatter(a[sl(a), 0] - c[0], a[sl(a), 2], s=2.0, c="r")
    ax[1].set(aspect="equal", title=f"side slice |dy| < 1.5 m  (NCC {reg.ncc:.2f}, ICP RMS {reg.icp_rms:.2f} m,"
                                    f" {'ok' if reg.ok else 'CHECK'})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return Path(path)


def crop_als(laz_files, centre, radius: float, chunk: int = 2_000_000) -> np.ndarray:
    """ALS points within radius of centre from LAZ flightlines: (N, 5) x, y, z, class, return."""
    import laspy

    parts = []
    for f in laz_files:
        with laspy.open(f) as fh:
            h = fh.header
            if (h.maxs[0] < centre[0] - radius or h.mins[0] > centre[0] + radius
                    or h.maxs[1] < centre[1] - radius or h.mins[1] > centre[1] + radius):
                continue
            for pts in fh.chunk_iterator(chunk):
                x, y = np.asarray(pts.x), np.asarray(pts.y)
                ok = (x - centre[0]) ** 2 + (y - centre[1]) ** 2 <= radius ** 2
                if ok.any():
                    parts.append(np.c_[x[ok], y[ok], np.asarray(pts.z)[ok],
                                      np.asarray(pts.classification)[ok], np.asarray(pts.return_number)[ok]])
    return np.concatenate(parts) if parts else np.zeros((0, 5))


def register_plots(root: str | Path, flight: str = "pre", plots: list[str] | None = None,
                   cfg: RegCfg | None = None, log=print) -> dict:
    """Register every converted scan of a WFSI download (`sky2trail wfsi fetch` layout).

    Writes <root>/registration/<flight>/<scan>.json and .png, and caches ALS crops in
    <root>/als_crops/<flight>/<plot>.npy. Returns {plot: Registration}.
    """
    from . import wfsi
    from .tls import load_npz

    cfg = cfg or RegCfg()
    root = Path(root)
    table = {p["name"]: p for p in json.loads((root / "manifest.json").read_text())["plots"]}
    out = root / "registration" / flight
    out.mkdir(parents=True, exist_ok=True)
    laz = sorted((root / "als" / flight).glob("*.laz"))
    regs = {}
    for npz in sorted((root / "tls_npz" / flight).glob("*.npz")):
        plot = next((n for n in table if wfsi.scan_name(n, flight).removesuffix(".ptx") == npz.stem), None)
        if plot is None or (plots and plot not in plots):
            continue
        p = table[plot]
        crop = root / "als_crops" / flight / f"{plot}.npy"
        if crop.exists():
            als = np.load(crop)
        else:
            als = crop_als(laz, (p["x"], p["y"]), cfg.tls_radius + cfg.search + 10)
            crop.parent.mkdir(parents=True, exist_ok=True)
            np.save(crop, als)
        scan = load_npz(npz)
        reg = register_scan(scan.xyz, als, (p["x"], p["y"]), cfg)
        reg.save(out / f"{npz.stem}.json")
        save_overlay(scan.xyz, als, reg, out / f"{npz.stem}.png", cfg.tls_radius)
        regs[plot] = reg
        log(f"{plot}: heading {reg.heading_deg:6.1f}  offset from GPS "
            f"{np.hypot(reg.shift[0] - p['x'], reg.shift[1] - p['y']):4.1f} m  NCC {reg.ncc:.2f}"
            f" (2nd {reg.ncc_second:.2f})  ICP RMS {reg.icp_rms:.2f} m  {'ok' if reg.ok else 'CHECK'}")
    return regs

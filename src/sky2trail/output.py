"""The Sky2Trail output format: what every Sky2Trail method (v0 rules, trained models) produces.

On disk (`Sky2TrailOutput.save`):
  points.npz / points.laz   completed point cloud; per point `source` (measured / generated)
                            and `component` (ground / stem / understory / crown)
  stems.csv                 one row per stem: position, diameter, height, confidence
  crowns.csv                measured crowns the stems belong to
  ground/elevation, ground/confidence   ground surface and how much of it was measured
  canopy/chm, canopy/low_vegetation     measured canopy layers
  summary.json              method, tile quality, counts
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from .grid import Grid
from .io import write_raster
from .terrain import roughness, slope_deg
from .trees import Crown

SOURCE_MEASURED = 0
SOURCE_GENERATED = 1
SOURCE_NAMES = {SOURCE_MEASURED: "measured", SOURCE_GENERATED: "generated"}

COMPONENT_UNKNOWN = 0
COMPONENT_GROUND = 1
COMPONENT_STEM = 2
COMPONENT_UNDERSTORY = 3
COMPONENT_CROWN = 4
COMPONENT_NAMES = {0: "unknown", 1: "ground", 2: "stem", 3: "understory", 4: "crown"}


@dataclass
class Stem:
    id: int
    x: float
    y: float
    dbh: float              # diameter at breast height (1.3 m), meters
    height: float           # tree height, meters
    crown_id: int           # measured crown it belongs to (0 = none, e.g. hidden understorey tree)
    confidence: float       # 0..1; method-specific, see summary["method"]

    def as_row(self) -> dict:
        return {"id": self.id, "x": round(self.x, 2), "y": round(self.y, 2), "dbh": round(self.dbh, 3),
                "height": round(self.height, 2), "crown_id": self.crown_id,
                "confidence": round(self.confidence, 3)}


@dataclass
class GroundSurface:
    grid: Grid
    elevation: np.ndarray   # meters
    confidence: np.ndarray  # 0..1, share of the surface backed by ground returns nearby

    def slope_deg(self) -> np.ndarray:
        return slope_deg(self.grid, self.elevation)

    def roughness(self, window_m: float = 3.0) -> np.ndarray:
        return roughness(self.grid, self.elevation, window_m)

    def elevation_at(self, x, y) -> np.ndarray:
        return self.grid.sample(self.elevation, np.asarray(x, float), np.asarray(y, float))


@dataclass
class CompletedCloud:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    source: np.ndarray      # SOURCE_* per point
    component: np.ndarray   # COMPONENT_* per point

    def __len__(self) -> int:
        return len(self.x)

    def counts(self) -> dict:
        out = {}
        for s, sname in SOURCE_NAMES.items():
            for c, cname in COMPONENT_NAMES.items():
                n = int(((self.source == s) & (self.component == c)).sum())
                if n:
                    out[f"{sname}/{cname}"] = n
        return out


@dataclass
class Canopy:
    grid: Grid
    chm: np.ndarray             # canopy height above ground
    low_vegetation: np.ndarray  # share of returns 0.3-2 m above ground
    crowns: list[Crown] = field(default_factory=list)


@dataclass
class Sky2TrailOutput:
    method: str                 # e.g. "v0-rules", or a model name + version
    ground: GroundSurface
    stems: list[Stem]
    points: CompletedCloud
    canopy: Canopy
    summary: dict = field(default_factory=dict)

    def save(self, out_dir: str | Path, crs_wkt: str | None = None) -> Path:
        out = Path(out_dir)
        for sub in ("ground", "canopy"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        files = {
            "ground_elevation": write_raster(out / "ground/elevation", self.ground.elevation, self.ground.grid, crs_wkt),
            "ground_confidence": write_raster(out / "ground/confidence", self.ground.confidence, self.ground.grid, crs_wkt),
            "chm": write_raster(out / "canopy/chm", self.canopy.chm, self.canopy.grid, crs_wkt),
            "low_vegetation": write_raster(out / "canopy/low_vegetation", self.canopy.low_vegetation, self.canopy.grid, crs_wkt),
            "stems": _write_csv(out / "stems.csv", [s.as_row() for s in self.stems],
                                ["id", "x", "y", "dbh", "height", "crown_id", "confidence"]),
            "crowns": _write_csv(out / "crowns.csv", [_crown_row(c) for c in self.canopy.crowns],
                                 ["id", "x", "y", "height", "crown_diameter", "n_points"]),
        }
        files.update(write_completed_cloud(out / "points", self.points, crs_wkt))
        summary = {"sky2trail_version": __version__, "method": self.method, **self.summary}
        summary["points"] = {"total": len(self.points), **self.points.counts()}
        summary["files"] = {k: str(Path(v).relative_to(out)) for k, v in files.items()}
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
        return out


def _crown_row(c: Crown) -> dict:
    return {"id": c.id, "x": round(c.x, 2), "y": round(c.y, 2), "height": round(c.height, 2),
            "crown_diameter": round(c.crown_diameter, 2), "n_points": c.n_points}


def _write_csv(path: Path, rows: list[dict], header: list[str]) -> Path:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(type(o))


def write_completed_cloud(path: str | Path, pc: CompletedCloud, crs_wkt: str | None = None) -> dict:
    """Always .npz; also .laz (LAS 1.4 with extra dims `source`, `component`) if laspy is installed."""
    path = Path(path)
    files = {}
    npz = path.with_suffix(".npz")
    np.savez_compressed(npz, x=pc.x, y=pc.y, z=pc.z, source=pc.source, component=pc.component)
    files["points_npz"] = npz
    try:
        import laspy
    except ImportError:
        return files

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="source", type=np.uint8,
                                                description="0 measured, 1 generated"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="component", type=np.uint8,
                                                # LAS limits descriptions to 32 bytes.
                                                description="0unk 1grnd 2stem 3under 4crown"))
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([float(pc.x.min()), float(pc.y.min()), float(pc.z.min())]) if len(pc) else np.zeros(3)
    if crs_wkt:
        try:
            import pyproj
            header.add_crs(pyproj.CRS.from_wkt(crs_wkt))
        except Exception:
            pass
    las = laspy.LasData(header)
    las.x, las.y, las.z = pc.x, pc.y, pc.z
    # ASPRS classes for tools that ignore extra dims: 2 ground, 3 low vegetation, 5 high vegetation.
    cls = np.ones(len(pc), dtype=np.uint8)
    cls[pc.component == COMPONENT_GROUND] = 2
    cls[pc.component == COMPONENT_UNDERSTORY] = 3
    cls[(pc.component == COMPONENT_CROWN) | (pc.component == COMPONENT_STEM)] = 5
    las.classification = cls
    las.source = pc.source.astype(np.uint8)
    las.component = pc.component.astype(np.uint8)
    laz = path.with_suffix(".laz")
    las.write(laz)
    files["points_laz"] = laz
    return files


def load_completed_cloud(path: str | Path) -> CompletedCloud:
    d = np.load(Path(path).with_suffix(".npz"))
    return CompletedCloud(d["x"], d["y"], d["z"], d["source"], d["component"])


def load_stems(path: str | Path) -> list[Stem]:
    with open(path, newline="") as fh:
        return [Stem(int(r["id"]), float(r["x"]), float(r["y"]), float(r["dbh"]), float(r["height"]),
                     int(r["crown_id"]), float(r["confidence"])) for r in csv.DictReader(fh)]

"""Reading point clouds and writing rasters, maps and tables."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grid import Grid

US_SURVEY_FOOT = 1200.0 / 3937.0
INTERNATIONAL_FOOT = 0.3048


@dataclass
class PointCloud:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    classification: np.ndarray
    return_number: np.ndarray
    number_of_returns: np.ndarray
    crs_wkt: str | None = None
    source: str = ""

    def __len__(self) -> int:
        return len(self.x)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return float(self.x.min()), float(self.y.min()), float(self.x.max()), float(self.y.max())

    def subset(self, mask: np.ndarray) -> "PointCloud":
        return PointCloud(
            self.x[mask], self.y[mask], self.z[mask], self.classification[mask],
            self.return_number[mask], self.number_of_returns[mask], self.crs_wkt, self.source,
        )


def _unit_scale(unit_name: str | None) -> float:
    if not unit_name:
        return 1.0
    name = unit_name.lower()
    if "survey" in name and "foot" in name:
        return US_SURVEY_FOOT
    if "foot" in name or name in ("ft", "feet"):
        return INTERNATIONAL_FOOT
    return 1.0


def _crs_scales(crs) -> tuple[float, float]:
    """Horizontal and vertical unit-to-meter factors from a pyproj CRS."""
    if crs is None:
        return 1.0, 1.0
    horiz = vert = None
    parts = crs.sub_crs_list if crs.is_compound else [crs]
    for part in parts:
        for axis in part.axis_info:
            if axis.direction in ("east", "north") and horiz is None:
                horiz = axis.unit_name
            if axis.direction == "up" and vert is None:
                vert = axis.unit_name
    h = _unit_scale(horiz)
    # A horizontal-only CRS usually means heights share its unit.
    v = _unit_scale(vert) if vert else h
    return h, v


def read_points(path: str | Path, units: str = "auto") -> PointCloud:
    """Read a LAS/LAZ file (needs laspy) or a TrailGen .npz, returning meters.

    units: 'auto' reads the unit from the file's CRS; 'm', 'us-ft' or 'ft' force it.
    """
    path = Path(path)
    if path.suffix == ".npz":
        d = np.load(path)
        return PointCloud(
            d["x"], d["y"], d["z"], d["classification"], d["return_number"],
            d["number_of_returns"], None, str(path),
        )

    try:
        import laspy
    except ImportError as exc:
        raise ImportError("Reading LAS/LAZ needs laspy: pip install 'laspy[lazrs]' pyproj") from exc

    las = laspy.read(path)
    crs = None
    try:
        crs = las.header.parse_crs()
    except Exception:  # missing or unparsable CRS records
        crs = None

    if units == "auto":
        hs, vs = _crs_scales(crs)
    else:
        hs = vs = {"m": 1.0, "us-ft": US_SURVEY_FOOT, "ft": INTERNATIONAL_FOOT}[units]

    return PointCloud(
        x=np.asarray(las.x, dtype=float) * hs,
        y=np.asarray(las.y, dtype=float) * hs,
        z=np.asarray(las.z, dtype=float) * vs,
        classification=np.asarray(las.classification, dtype=np.uint8),
        return_number=np.asarray(las.return_number, dtype=np.uint8),
        number_of_returns=np.asarray(las.number_of_returns, dtype=np.uint8),
        crs_wkt=crs.to_wkt() if crs is not None else None,
        source=str(path),
    )


def save_points_npz(pc: PointCloud, path: str | Path) -> None:
    np.savez_compressed(
        path, x=pc.x, y=pc.y, z=pc.z, classification=pc.classification,
        return_number=pc.return_number, number_of_returns=pc.number_of_returns,
    )


def write_raster(path: str | Path, raster: np.ndarray, grid: Grid, crs_wkt: str | None = None,
                 nodata: float = np.nan) -> Path:
    """GeoTIFF if rasterio is installed, otherwise .npy plus a JSON sidecar with the grid."""
    path = Path(path)
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError:
        npy = path.with_suffix(".npy")
        np.save(npy, raster.astype(np.float32))
        npy.with_suffix(".json").write_text(json.dumps(
            {"transform": grid.transform, "shape": grid.shape, "crs_wkt": crs_wkt}, indent=2))
        return npy

    tif = path.with_suffix(".tif")
    a, b, c, d, e, f = grid.transform
    with rasterio.open(
        tif, "w", driver="GTiff", height=grid.nrows, width=grid.ncols, count=1,
        dtype="float32", crs=crs_wkt, transform=Affine(a, b, c, d, e, f),
        nodata=nodata, compress="deflate",
    ) as dst:
        dst.write(raster.astype(np.float32), 1)
    return tif


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


def write_trees_csv(path: str | Path, trees: list[dict]) -> Path:
    path = Path(path)
    if not trees:
        path.write_text("id,x,y,height,crown_diameter,dbh\n")
        return path
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(trees[0].keys()))
        writer.writeheader()
        writer.writerows(trees)
    return path

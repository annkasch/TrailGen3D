"""Download a 3DEP lidar point cloud for a small lon/lat box, as LAZ in UTM meters.

Uses the public USGS 3DEP Entwine Point Tile (EPT) data on AWS through PDAL, so only the
requested box is read, not whole project tiles.

Requires the PDAL command line tool (e.g. `conda install -c conda-forge pdal`).
Not yet tested end to end: check the printed project list and the output on first use.

Examples:
    python scripts/fetch_3dep.py --list --bbox -119.30 37.05 -119.29 37.06
    python scripts/fetch_3dep.py --bbox -119.30 37.05 -119.29 37.06 --out data/pilot.laz
    python scripts/fetch_3dep.py --bbox ... --project CA_SierraNevada_8_2022 --out data/pilot.laz
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# Index of 3DEP EPT projects with their footprints (maintained by the EPT data publisher).
RESOURCES_URL = "https://raw.githubusercontent.com/hobu/usgs-lidar/master/boundaries/resources.geojson"
EPT_URL = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{name}/ept.json"


def lonlat_to_3857(lon: float, lat: float) -> tuple[float, float]:
    r = 6378137.0
    x = math.radians(lon) * r
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * r
    return x, y


def utm_epsg(lon: float) -> int:
    """NAD83 / UTM zone N, the usual frame for US work (California: zones 10 and 11)."""
    zone = int((lon + 180) // 6) + 1
    return 26900 + zone


def find_projects(bbox) -> list[dict]:
    with urllib.request.urlopen(RESOURCES_URL, timeout=60) as fh:
        data = json.load(fh)
    lon0, lat0, lon1, lat1 = bbox
    hits = []
    for feat in data["features"]:
        props = feat.get("properties", {})
        coords = feat["geometry"]["coordinates"]
        flat = list(_iter_points(coords))
        if not flat:
            continue
        xs, ys = zip(*flat)
        if min(xs) <= lon1 and max(xs) >= lon0 and min(ys) <= lat1 and max(ys) >= lat0:
            hits.append({"name": props.get("name"), "count": props.get("count"), "url": props.get("url")})
    return hits


def _iter_points(coords):
    if isinstance(coords[0], (int, float)):
        yield coords[0], coords[1]
    else:
        for c in coords:
            yield from _iter_points(c)


def fetch(bbox, project: str, out: Path) -> None:
    if shutil.which("pdal") is None:
        sys.exit("PDAL not found. Install it, e.g.: conda install -c conda-forge pdal")
    lon0, lat0, lon1, lat1 = bbox
    x0, y0 = lonlat_to_3857(lon0, lat0)
    x1, y1 = lonlat_to_3857(lon1, lat1)
    epsg = utm_epsg((lon0 + lon1) / 2)
    pipeline = [
        {"type": "readers.ept", "filename": EPT_URL.format(name=project),
         "bounds": f"([{x0}, {x1}], [{y0}, {y1}])"},
        {"type": "filters.reprojection", "out_srs": f"EPSG:{epsg}"},
        {"type": "writers.las", "filename": str(out), "compression": "true",
         "minor_version": 4, "a_srs": f"EPSG:{epsg}"},
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(pipeline, fh)
        path = fh.name
    print(f"Reading {project} for box {bbox} -> {out} (EPSG:{epsg})")
    subprocess.run(["pdal", "pipeline", path], check=True)
    print("Done. Next: trailgen inspect", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    p.add_argument("--project", help="3DEP project name; default: newest-looking match")
    p.add_argument("--out", type=Path, help="output .laz path")
    p.add_argument("--list", action="store_true", help="only list matching projects")
    args = p.parse_args()

    projects = find_projects(args.bbox)
    if not projects:
        sys.exit("No 3DEP EPT project covers this box.")
    print("Projects covering the box:")
    for pr in projects:
        print(f"  {pr['name']}  ({pr['count']} points)")
    if args.list:
        return
    if args.out is None:
        sys.exit("Give --out to download.")
    # Project names usually end in the survey year; prefer the most recent.
    name = args.project or sorted(projects, key=lambda pr: pr["name"][-4:])[-1]["name"]
    fetch(args.bbox, name, args.out)


if __name__ == "__main__":
    main()

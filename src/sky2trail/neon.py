"""NEON training data: mapped stems (vegetation structure) and the matching airborne lidar.

Downloads from the NEON Data API (https://data.neonscience.org/data-api/) into a local
folder, so anyone with the repository can rebuild the same training data:

    sky2trail neon fetch --site SJER --out data/neon

Since June 2026 NEON needs a free API token for downloads: create an account at
data.neonscience.org, open My Account, click "Get API token", then
`export NEON_API_TOKEN=...` (or pass --token). Metadata and locations need no token.

Products:
  DP1.10098.001  vegetation structure. Stems of woody plants >= 10 cm DBH in each plot,
                 mapped as distance and azimuth from a plot reference point
                 (vst_mappingandtagging), DBH and height per survey (vst_apparentindividual).
  DP1.30003.001  discrete-return lidar point cloud, 1 km x 1 km LAZ tiles (classified).

Stem positions: reference point UTM + distance * (sin az, cos az), as in NEON's geoNEON
package. NEON gives ~0.6 m position uncertainty for mapped stems.

Downloads are pinned to a NEON release (default below) so results are reproducible; every
downloaded file is listed with its checksum in manifest.json.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

API = "https://data.neonscience.org/api/v0/"
VST = "DP1.10098.001"
LIDAR = "DP1.30003.001"
RELEASE = "RELEASE-2026"
D17_SITES = ("SJER", "SOAP", "TEAK")
USER_AGENT = "trailgen3d-neon-loader"


class NeonError(RuntimeError):
    pass


def token_from_env(token: str | None = None) -> str | None:
    return token or os.environ.get("NEON_API_TOKEN") or None


def _get(url: str, token: str | None = None, retries: int = 3) -> bytes:
    headers = {"User-Agent": USER_AGENT, "accept": "application/json"}
    if token:
        headers["X-API-Token"] = token
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise NeonError(
                    f"NEON refused {url} (403). Data downloads need a NEON API token: "
                    "data.neonscience.org > My Account > Get API token, then set NEON_API_TOKEN."
                ) from e
            if e.code == 429 or e.code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise NeonError(f"giving up on {url}")


def get_json(path: str, token: str | None = None, **params) -> dict:
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    d = json.loads(_get(API + path + (f"?{q}" if q else ""), token))
    if d.get("error"):
        raise NeonError(f"{path}: {d['error']}")
    return d["data"]


def available_months(product: str, site: str, release: str = RELEASE) -> list[str]:
    """Months with data for a product at a site in the given release (no token needed)."""
    d = get_json(f"products/{product}")
    for s in d["siteCodes"]:
        if s["siteCode"] == site:
            for r in s.get("availableReleases", []):
                if r["release"] == release:
                    return sorted(r["availableMonths"])
    return []


def month_files(product: str, site: str, month: str, token: str | None, release: str = RELEASE) -> list[dict]:
    d = get_json(f"data/{product}/{site}/{month}", token, release=release, package="basic")
    return d["files"]


def download(f: dict, dest: Path, token: str | None) -> Path:
    """Download one file entry (name, url, size) unless it is already there with the right size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = int(f.get("size") or 0)
    if dest.exists() and (not size or dest.stat().st_size == size):
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(_get(f["url"], token))
    tmp.rename(dest)
    return dest


# ---------------------------------------------------------------------------
# Vegetation structure -> stem table


@dataclass
class NeonStem:
    site: str
    plot: str
    individual: str
    x: float              # UTM easting, m
    y: float              # UTM northing, m
    utm_zone: str
    dbh: float            # m (NaN if not measured in this survey)
    height: float         # m (NaN if not measured)
    date: str             # survey date of dbh/height
    status: str           # plantStatus, e.g. "Live"
    growth_form: str      # e.g. "single bole tree"
    xy_uncertainty: float  # m


def _read_tables(zips: list[Path], table: str) -> list[dict]:
    """Rows of one NEON table (e.g. 'vst_mappingandtagging') from all downloaded month zips/csvs."""
    rows = []
    for p in zips:
        if p.suffix == ".csv":
            if f".{table}." in p.name:
                with open(p, newline="", encoding="utf-8") as fh:
                    rows += list(csv.DictReader(fh))
            continue
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if f".{table}." in name and name.endswith(".csv"):
                    rows += list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8")))
    return rows


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


class _Points:
    """Cached UTM coordinates of plot reference points (locations API, no token needed)."""

    def __init__(self, cache: Path):
        self.cache = cache
        self.d = json.loads(cache.read_text()) if cache.exists() else {}

    def get(self, plot: str, point: str):
        name = f"{plot}.basePlot.vst.{point}"
        if name not in self.d:
            try:
                loc = get_json(f"locations/{name}")
                unc = next((_f(p["locationPropertyValue"]) for p in loc.get("locationProperties", [])
                            if p["locationPropertyName"] == "Value for Coordinate uncertainty"), float("nan"))
                self.d[name] = [loc["locationUtmEasting"], loc["locationUtmNorthing"],
                                f"{loc['locationUtmZone']}{loc['locationUtmHemisphere']}", unc]
            except (NeonError, urllib.error.HTTPError):
                self.d[name] = None
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            self.cache.write_text(json.dumps(self.d))
        return self.d[name]


def build_stems(site_dir: Path, site: str) -> list[NeonStem]:
    """One row per mapped individual and survey, from the downloaded vegetation structure files."""
    files = sorted((site_dir / "vst").glob("*"))
    mapping = _read_tables(files, "vst_mappingandtagging")
    measured = _read_tables(files, "vst_apparentindividual")
    # Latest mapping record per individual.
    where = {}
    for r in sorted(mapping, key=lambda r: r.get("date", "")):
        if r.get("pointID") and r.get("stemDistance") and r.get("stemAzimuth"):
            where[r["individualID"]] = r
    points = _Points(site_dir / "reference_points.json")
    out = []
    for r in measured:
        m = where.get(r.get("individualID"))
        if m is None:
            continue
        ref = points.get(m["plotID"], m["pointID"])
        if ref is None:
            continue
        e, n, zone, unc = ref
        dist, az = _f(m["stemDistance"]), math.radians(_f(m["stemAzimuth"]))
        out.append(NeonStem(
            site=site, plot=m["plotID"], individual=m["individualID"],
            x=e + dist * math.sin(az), y=n + dist * math.cos(az), utm_zone=zone,
            dbh=_f(r.get("stemDiameter")) / 100.0, height=_f(r.get("height")),
            date=r.get("date", ""), status=r.get("plantStatus", ""),
            growth_form=r.get("growthForm", ""),
            xy_uncertainty=(unc if unc == unc else 0.0) + 0.6,
        ))
    return out


def save_stems(stems: list[NeonStem], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(NeonStem.__dataclass_fields__)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        for s in stems:
            w.writerow(asdict(s))
    return path


def load_stems(path: Path) -> list[NeonStem]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    num = {"x", "y", "dbh", "height", "xy_uncertainty"}
    return [NeonStem(**{k: (float(v) if k in num else v) for k, v in r.items()}) for r in rows]


# ---------------------------------------------------------------------------
# Lidar tiles


def tile_of(x: float, y: float) -> tuple[int, int]:
    """South-west corner of the 1 km NEON AOP tile containing (x, y)."""
    return int(math.floor(x / 1000.0) * 1000), int(math.floor(y / 1000.0) * 1000)


def lidar_files(site: str, month: str, token: str | None, release: str = RELEASE) -> dict:
    """Classified point cloud LAZ files of one flight, keyed by tile corner."""
    tiles = {}
    for f in month_files(LIDAR, site, month, token, release):
        name = f["name"]
        if not name.endswith(".laz") or "classified_point_cloud" not in name.lower():
            continue
        parts = name.split("_")
        try:
            i = next(k for k, p in enumerate(parts) if p.isdigit() and len(p) == 6)
            tiles[(int(parts[i]), int(parts[i + 1]))] = f
        except (StopIteration, IndexError, ValueError):
            continue
    return tiles


def nearest_flight(survey_month: str, flight_months: list[str]) -> str | None:
    """Flight closest in time to a survey month ('YYYY-MM')."""
    if not flight_months:
        return None
    def idx(m):
        y, mo = m.split("-")
        return int(y) * 12 + int(mo)
    return min(flight_months, key=lambda m: abs(idx(m) - idx(survey_month)))


# ---------------------------------------------------------------------------
# Fetch everything for a site


def fetch_site(site: str, out: str | Path, token: str | None = None, release: str = RELEASE,
               months: list[str] | None = None, max_plots: int | None = None, log=print) -> dict:
    """Download stem tables and the lidar tiles covering the plots, for one site.

    Layout: OUT/SITE/vst/*.zip, OUT/SITE/stems.csv, OUT/SITE/lidar/YYYY-MM/*.laz,
    OUT/SITE/manifest.json (release, files, checksums, survey-to-flight pairing).
    max_plots: only the first N plots (sorted by name), for a quick look.
    """
    token = token_from_env(token)
    site_dir = Path(out) / site
    manifest = {"site": site, "release": release, "products": {VST: {}, LIDAR: {}}}

    vst_months = months or available_months(VST, site, release)
    log(f"{site}: vegetation structure, {len(vst_months)} months in {release}")
    for m in vst_months:
        for f in month_files(VST, site, m, token, release):
            if any(f".{t}." in f["name"] for t in ("vst_mappingandtagging", "vst_apparentindividual")):
                download(f, site_dir / "vst" / f["name"], token)
                manifest["products"][VST][f["name"]] = {"month": m, "size": f.get("size"),
                                                        "crc32c": f.get("crc32c") or f.get("md5")}
    stems = build_stems(site_dir, site)
    if max_plots:
        keep = sorted({s.plot for s in stems})[:max_plots]
        stems = [s for s in stems if s.plot in keep]
    save_stems(stems, site_dir / "stems.csv")
    plots = sorted({s.plot for s in stems})
    log(f"{site}: {len(stems)} stem records, {len({s.individual for s in stems})} individuals, "
        f"{len(plots)} plots")

    flights = available_months(LIDAR, site, release)
    survey_months = sorted({s.date[:7] for s in stems if s.date})
    pairing = {m: nearest_flight(m, flights) for m in survey_months}
    manifest["survey_to_flight"] = pairing
    need = {}
    for s in stems:
        fl = pairing.get(s.date[:7])
        if fl:
            need.setdefault(fl, set()).add(tile_of(s.x, s.y))
    for fl, tiles in sorted(need.items()):
        avail = lidar_files(site, fl, token, release)
        for t in sorted(tiles):
            f = avail.get(t)
            if f is None:
                log(f"{site} {fl}: no lidar tile at {t}")
                continue
            download(f, site_dir / "lidar" / fl / f["name"], token)
            manifest["products"][LIDAR][f"{fl}/{f['name']}"] = {"size": f.get("size"),
                                                                 "crc32c": f.get("crc32c") or f.get("md5")}
        log(f"{site} {fl}: {len(tiles)} lidar tiles")
    (site_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def summary(site_dir: Path) -> dict:
    """Counts for training: plots, individuals, stem records by size and survey year."""
    stems = load_stems(site_dir / "stems.csv")
    live = [s for s in stems if s.status.lower().startswith("live")]
    dbh = np.array([s.dbh for s in live if s.dbh == s.dbh])
    years = {}
    for s in live:
        years[s.date[:4]] = years.get(s.date[:4], 0) + 1
    return {
        "plots": len({s.plot for s in stems}),
        "individuals": len({s.individual for s in stems}),
        "records": len(stems),
        "live_records": len(live),
        "live_with_dbh": int(len(dbh)),
        "dbh_ge_10cm": int((dbh >= 0.10).sum()),
        "dbh_median_m": round(float(np.median(dbh)), 3) if len(dbh) else None,
        "live_records_by_year": dict(sorted(years.items())),
        "lidar_tiles": len(list((site_dir / "lidar").glob("*/*.laz"))),
    }

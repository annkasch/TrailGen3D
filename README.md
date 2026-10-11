# TrailGen

Navigation-ready simulation worlds of natural California terrain, generated from public
aerial lidar (USGS 3DEP) and validated against ground truth from a Unitree Go2.

This repository currently holds **Phase 1, the baseline world generator**: one lidar tile
in, a ground model, tree list, low-vegetation map and traversability cost map out,
including a map that Nav2 can load directly.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[geo,dev]"          # geo: laspy, pyproj, rasterio (real tiles, GeoTIFFs)
pytest                              # synthetic end-to-end tests

# 1. Try it without any data: a synthetic forest with known trees, scored automatically
trailgen synth --out outputs/synth_ql2 --density 2     # QL2-like
trailgen synth --out outputs/synth_ql1 --density 8     # QL1-like

# 2. Get a real tile (needs PDAL: conda install -c conda-forge pdal)
python scripts/fetch_3dep.py --list --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX
python scripts/fetch_3dep.py --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX --out data/pilot.laz

# 3. Check what the tile can resolve, then process it
trailgen inspect data/pilot.laz
trailgen run data/pilot.laz --out outputs/pilot
```

Keep boxes small at first (a few hundred meters); one square kilometer of QL1 data is
roughly ten million points.

## What `trailgen run` produces

| Output | What it is |
| --- | --- |
| `rasters/dem` | Ground elevation, gaps under canopy interpolated |
| `rasters/slope_deg`, `rasters/roughness` | Terrain layers from the DEM |
| `rasters/chm` | Canopy height above ground |
| `rasters/low_vegetation` | Share of returns 0.3–2 m above ground (shrubs, low branches) |
| `rasters/cost` | Traversability, 0 = free, 1 = blocked |
| `rasters/confidence` | Share of the ground model backed by real ground returns |
| `trees.csv` | Tree positions, height, crown diameter, trunk diameter, lidar points per crown |
| `nav2/map.pgm`, `nav2/map.yaml` | Cost map in Nav2 map_server format (local frame) |
| `summary.json` | Point density, estimated quality level, tree statistics, all parameters |
| `preview.png` | Four-panel overview |

Rasters are GeoTIFFs when `rasterio` is installed, otherwise `.npy` with a JSON sidecar.

## How it works

1. **Ground:** delivered ground-class points, minus outliers that sit above the local ground
   surface, averaged per cell; gaps filled by interpolation. Falls back to a lowest-point
   filter if a tile has no usable ground class.
2. **Canopy:** highest return above ground per cell; cells with no return are filled from
   neighbours, so sparse QL2 data doesn't split crowns.
3. **Trees:** local maxima with a height-dependent search radius, plus a saddle test
   (two peaks are separate trees only if the canopy dips between them); crowns by watershed.
4. **Trunks (baseline):** one trunk under each treetop, diameter from a height rule.
   **The coefficients in `configs/default.toml` are placeholders** to be calibrated per
   forest type (e.g. NEON plots) and later replaced by the learned tree model.
5. **Traversability:** slope, roughness and low vegetation as graded costs; trunks
   (inflated by the robot's half width and a margin) and slopes above the robot's limit as
   blocked.

## Baseline accuracy on synthetic forests

From `trailgen synth` (120 m × 120 m, 60 trees, seeds 0–2):

| Pulse density | Trunks found within 1 m | Detected trunks that are real |
| --- | --- | --- |
| 2 / m² (QL2) | 92–97 % | 100 % |
| 8 / m² (QL1) | 95–97 % | 100 % |

Trees under 10 m are found least often (60–67 %), as expected from aerial data. The
synthetic forest is deliberately simple, so **real forests will score lower**; the real
benchmark comes from field ground truth (Phase 2 and Phase 5 of the build plan).

## Layout

```
configs/default.toml     all parameters (robot limits, thresholds, cost weights)
scripts/fetch_3dep.py    download a 3DEP point cloud for a lon/lat box
src/trailgen/
  io.py                  LAS/LAZ reading (unit handling), GeoTIFF, Nav2 map, CSV
  grid.py                aligned north-up grids, binning, resampling
  terrain.py             ground model, height above ground, slope, roughness
  trees.py               canopy height, treetops, crowns, baseline trunks
  vegetation.py          low-vegetation share
  traversability.py      cost map
  quality.py             point density and quality-level report
  evaluate.py            trunk matching metrics (recall, precision, errors)
  synthetic.py           synthetic forest + simulated airborne lidar
  pipeline.py, cli.py    orchestration and the `trailgen` command
tests/                   end-to-end tests on synthetic data
```

## Known limits

- Robot limits in the config are defaults; replace them with values measured on your Go2.
- Steps and drop-offs smaller than the DEM resolution (0.5–1 m) are not resolved.
- Low-vegetation values under dense canopy are underestimated (the canopy hides it).
- `scripts/fetch_3dep.py` has not yet been tested end to end.

## Project documents

Project description, proposal, build plan and detailed plans are kept as shared docs;
see `docs/README.md`.

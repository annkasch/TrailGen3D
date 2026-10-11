# TrailGen3D

Navigation-ready 3D simulation worlds of natural terrain, generated from public aerial
lidar (USGS 3DEP) and validated against ground truth from a Unitree Go2.

The repository holds two packages:

| Package | Command | Does |
| --- | --- | --- |
| **`sky2trail`** | `sky2trail` | Aerial lidar → ground-level data: what a scanner on the ground would see |
| **`trailgen3d`** | `trailgen` | Ground-level data → navigation world (cost map, Nav2 map, later simulator scenes) |

`trailgen3d` never reads aerial lidar itself; it builds worlds from whatever a Sky2Trail
method produced. The planned Go2 + 3DEP dataset of California trails is **Sky2Trail-CA**.

## Sky2Trail

**Goal:** a trained model that turns aerial lidar into ground-level structure. Every
Sky2Trail method produces the same output:

| Output | Contents |
| --- | --- |
| `points.npz` / `points.laz` | Completed point cloud; per point `source` (0 measured, 1 generated) and `component` (1 ground, 2 stem, 3 understory, 4 crown) |
| `stems.csv` | Per stem: position, diameter at breast height, tree height, crown id, confidence |
| `crowns.csv` | Measured crowns: treetop, height, crown diameter, lidar points per crown |
| `ground/elevation`, `ground/confidence` | Ground surface and the share of it backed by real ground returns |
| `canopy/chm`, `canopy/low_vegetation` | Measured canopy height and low-vegetation share |
| `summary.json` | Method, tile quality (density, quality level), counts |

**v0 (`sky2trail.baseline`)** fills this format with fixed rules and no learning. It's the
baseline trained models must beat:

1. Ground: ground-class points minus outliers, gaps under canopy interpolated.
2. Canopy: highest return per cell, empty cells filled from neighbours.
3. Crowns: canopy peaks with a height-dependent search radius and a saddle test, then watershed.
4. Stems (guessed): one per crown, under the treetop; diameter from height with
   **placeholder coefficients**; drawn as a cylinder up to an assumed crown base.

On a synthetic forest, v0 finds 92–97 % of stems within 1 m at QL2 and QL1 density with
no false stems. The synthetic forest is deliberately simple; real-forest numbers come from
field ground truth.

## Stage 1: learned stem map

`sky2trail.stage1` replaces v0's guessed stems with a trained network; ground and canopy stay
as measured. A small U-Net reads each 22 m input window as a 13-channel top-down image
(0.25 m pixels): returns per height band, canopy height, lowest return, share of later
returns, ground hits, pulse density. It predicts per pixel:

- a stem heatmap (two channels: stems >= 10 cm DBH and thinner ones),
- the stem's position inside the pixel,
- DBH and tree height.

Peaks in the heatmap become the stem list. Each window keeps only the stems in its 10 m core.

The loss counts only where stems are labelled. That excludes area outside a field plot, and
the small-stem channel on sources like NEON that map only stems >= 10 cm.

Until real plots are prepared, it trains on synthetic forests made harder on purpose:
small trees under big crowns, leaning trees, and occasional trunk hits. On these, v0
finds only ~40 % of stems within 1 m.

```bash
pip install torch          # CUDA build on the GPU machine: see pytorch.org for the right command
pytest tests/sky2trail/test_stage1_torch.py -v      # ~1 min smoke test of model, training, checkpoints

sky2trail stage1 data-synth --out data/stage1_synth                 # 40 train + 8 val scenes, ~1 min
sky2trail stage1 train --data data/stage1_synth --out runs/s1_synth # GPU picked automatically
sky2trail stage1 eval-synth --model runs/s1_synth/best.pt --out runs/s1_synth/eval   # vs v0
sky2trail stage1 run data/pilot.laz --model runs/s1_synth/best.pt --out outputs/pilot_s1
```

`train` writes `last.pt` every epoch (`--resume` continues from it), `best.pt` on the best
validation F1, and `metrics.jsonl` with one line per epoch.

## Quick start

```bash
conda create -n trailgen3d -c conda-forge python=3.11 pdal laspy lazrs-python rasterio pyproj numpy scipy matplotlib pytest
conda activate trailgen3d
pip install -e . --no-deps
pytest

# No data needed: synthetic forest with known stems, scored automatically
sky2trail synth --out outputs/s2t_synth --density 2      # 2 = QL2, 8 = QL1
trailgen synth --out outputs/world_synth

# Real tile (needs PDAL)
python scripts/fetch_3dep.py --list --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX
python scripts/fetch_3dep.py --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX --out data/pilot.laz
sky2trail inspect data/pilot.laz          # density, quality level, smallest reliable feature
sky2trail run data/pilot.laz --out outputs/pilot_s2t
trailgen run data/pilot.laz --out outputs/pilot   # Sky2Trail v0 + world
```

Keep boxes small at first (a few hundred meters); one square kilometer of QL1 data is
roughly ten million points. Rasters are GeoTIFFs when `rasterio` is installed, otherwise
`.npy` with a JSON sidecar.

## Layout

```
configs/default.toml        [sky2trail.*] and [trailgen3d.*] parameters
scripts/fetch_3dep.py       download a 3DEP point cloud for a lon/lat box
src/sky2trail/
  output.py                 the output format (stems, ground surface, completed cloud)
  baseline.py               v0 rules
  patches.py                patches for trained models: 10 m core, 22 m input window
  stage1/                   learned stem map: features, targets, augment, decode, dataset
                            (numpy); model, train, predict (PyTorch); cli
  terrain.py trees.py vegetation.py   measured ground, canopy, crowns, low vegetation
  io.py grid.py quality.py  lidar reading, grids, tile quality report
  synthetic.py evaluate.py  synthetic forests with known stems, stem-matching metrics
  viz.py cli.py
src/trailgen3d/
  traversability.py         cost map from a Sky2Trail output
  export.py                 Nav2 map
  pipeline.py viz.py cli.py
tests/sky2trail, tests/trailgen3d
```

## Known limits

- v0 stem diameters and crown base use placeholder rules; robot limits are Go2 defaults.
- Steps and drop-offs smaller than the ground grid (0.5–1 m) are not resolved.
- Low vegetation under dense canopy is underestimated (the canopy hides it).
- Reading LAZ, writing GeoTIFF/LAZ and `scripts/fetch_3dep.py` are not yet tested on real tiles.

# TrailGen3D

Tree stem detection from aerial lidar. Work in progress.

## Install

```bash
conda env create -f environment.yml
conda activate trailgen3d
pytest
```

## Usage

```bash
sky2trail stage1 data-synth --out data/stage1_synth
sky2trail stage1 train --data data/stage1_synth --out runs/s1
sky2trail stage1 eval-synth --model runs/s1/best.pt --out runs/s1/eval
sky2trail stage1 run tile.laz --model runs/s1/best.pt --out outputs/tile
```

`run` writes `stems.csv` with each stem's position, DBH, height, confidence and standard deviations.

## Training data

NEON stem maps and matching airborne lidar (needs a free NEON API token from
data.neonscience.org > My Account):

```bash
export NEON_API_TOKEN=...
sky2trail neon fetch --site SJER SOAP TEAK --out data/neon
sky2trail neon summary --out data/neon
```

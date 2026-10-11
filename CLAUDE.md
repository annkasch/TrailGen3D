# TrailGen3D: context for Claude Code

Navigation-ready 3D simulation worlds of natural California terrain from public aerial lidar
(USGS 3DEP), validated against ground truth from a Unitree Go2 Pro. Owner: Ann-Kathrin.

## Names and scope
- **Sky2Trail** (`src/sky2trail`, command `sky2trail`): aerial lidar -> ground-level data only
  (stems, ground surface, completed point cloud). The goal is a **trained model**; the rule-based
  v0 (`baseline.py`) is the baseline to beat and the reference implementation of the output format.
- **TrailGen3D** (`src/trailgen3d`, command `trailgen`): everything built on Sky2Trail output
  (cost maps, Nav2 maps, later simulator scenes). Never reads aerial lidar itself.
- **Sky2Trail-CA**: planned Go2 + 3DEP dataset of California trails.

## Rules
- Planning documents (description, proposal, plans) live in `docs/`, which is git-ignored. Never commit them.
- Commit after each working step with a clear message; push when a remote exists.
- Data, outputs, runs and checkpoints stay out of git (`.gitignore`).
- Run the tests before committing: `pytest` (numpy tests run anywhere; `tests/sky2trail/test_stage1_torch.py` needs PyTorch).

## Model design (decided)
- **Patches, no tree detector:** 10 m core (predicted area) inside a 22 m input window
  (6 m margin, about one Sierra crown radius). Cores tile a scene; each stem belongs to one core.
  10 m cores fit inside the 20 m USGS Sierra plots (1 each) and NEON plots (4 per 20 m, 16 per 40 m).
- **Stage 1 (implemented, `src/sky2trail/stage1`):** U-Net on a 13-channel top-down image
  (0.25 m pixels) -> stem heatmaps (>= / < 10 cm DBH), sub-pixel offset, log DBH, height
  (CenterNet-style). Loss only where stems are labelled; small-stem channel ignored for NEON.
  The PyTorch parts (model, train, predict) have not been run yet: run
  `pytest tests/sky2trail/test_stage1_torch.py -v` first on any new machine.
- **Stage 2 (later):** conditional point diffusion of the core (~4k points), conditioned on stage-1
  stems; backbone from CGNet/PDR (Lyu et al. 2021), the model ForestGen3D builds on.
- Training: density augmentation 2-15 pts/m2; split train/test by plot or site, never by patch.

## Data (planned)
- USGS Sierra Nevada 2025 lidar and fuels data: 133 TLS plots, 20 m circles (stems must be
  extracted from the TLS scans) -> stages 1 and 2.
- NEON D17 (SJER, SOAP, TEAK) vegetation structure: stem maps of trees >= 10 cm DBH,
  ~0.6 m position error, no ground scans -> stage 1 only. Check fire perimeters and survey dates.
- Go2 scans: mainly the test set.
- Synthetic: `make_hard_scene` (understory, leaning trees, trunk hits) for development;
  v0 finds ~40 % of stems within 1 m there.

## Compute
- Development on the Mac. Training on an RTX 3080 in a Windows PC (WSL2 Ubuntu), reachable
  over Tailscale + SSH (e.g. `ssh gpu-pc`). Run long jobs in tmux; check logs over SSH.
- Rent a GPU (Vast.ai / RunPod) only for stage-2 runs. NAIRR startup allocation is an option later.

## Commands
```bash
sky2trail synth --out outputs/s2t_synth --density 2           # v0 on a synthetic forest
sky2trail stage1 data-synth --out data/stage1_synth           # ~2 min, 4,000 windows
sky2trail stage1 train --data data/stage1_synth --out runs/s1_synth
sky2trail stage1 eval-synth --model runs/s1_synth/best.pt --out runs/s1_synth/eval
```

## Next steps
1. Set up the 3080 PC (WSL2, conda, PyTorch with CUDA, Tailscale + SSH) and a private GitHub repo.
2. Run the PyTorch smoke test, train stage 1 on synthetic data, compare with v0.
3. Real training data: pair Sierra / NEON plots with 3DEP, align, extract stem labels,
   coverage masks; score v0 on real plots.

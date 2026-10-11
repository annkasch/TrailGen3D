"""`sky2trail inspect | run | synth | stage1`: aerial lidar in, ground-level data out."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _cmd_inspect(args) -> int:
    from .io import read_points
    from .quality import density_report

    print(json.dumps(density_report(read_points(args.input, units=args.units)), indent=2))
    return 0


def _cmd_run(args) -> int:
    from .baseline import run_v0
    from .config import load_config
    from .io import read_points

    pc = read_points(args.input, units=args.units)
    print(f"Read {len(pc):,} points from {args.input}")
    out = run_v0(pc, load_config(args.config))
    out.save(args.out, pc.crs_wkt)
    if not args.no_preview:
        from .viz import save_preview
        save_preview(out, Path(args.out) / "preview.png")
    s = out.summary
    print(f"Tile: {s['tile']['first_returns_per_m2']} pulses/m2, {s['tile']['estimated_quality_level']}")
    print(f"Ground measured: {s['ground_measured_fraction']:.0%} of cells; crowns: {s['crowns']}; "
          f"stems: {s['stems']} ({out.method})")
    print(f"Outputs in {args.out}/")
    return 0


def _cmd_synth(args) -> int:
    from .baseline import run_v0
    from .config import load_config
    from .evaluate import match_trunks, recall_by_height
    from .io import save_points_npz
    from .synthetic import make_scene

    scene = make_scene(size=args.size, n_trees=args.trees, pulses_per_m2=args.density, seed=args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_points_npz(scene.points, out_dir / "aerial_points.npz")
    np.savetxt(out_dir / "truth_stems.csv", scene.trees, delimiter=",",
               header="x,y,height,crown_radius,dbh", comments="", fmt="%.3f")

    out = run_v0(scene.points, load_config(args.config))
    out.save(out_dir)
    if not args.no_preview:
        from .viz import save_preview
        save_preview(out, out_dir / "preview.png")

    pred_xy = np.array([[s.x, s.y] for s in out.stems]).reshape(-1, 2)
    pred_dbh = np.array([s.dbh for s in out.stems])
    t = scene.trees
    report = {
        "method": out.method,
        "density_pulses_per_m2": args.density,
        "stems_within_0.5m": match_trunks(pred_xy, pred_dbh, t[:, :2], t[:, 4], 0.5),
        "stems_within_1.0m": match_trunks(pred_xy, pred_dbh, t[:, :2], t[:, 4], 1.0),
        "recall_by_height_1.0m": recall_by_height(pred_xy, t[:, :2], t[:, 2], max_dist=1.0),
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Outputs in {out_dir}/")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sky2trail", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="point density and quality level of a tile")
    pi.add_argument("input")
    pi.add_argument("--units", default="auto", choices=["auto", "m", "us-ft", "ft"])
    pi.set_defaults(func=_cmd_inspect)

    pr = sub.add_parser("run", help="aerial tile -> ground surface, stems, completed point cloud (v0)")
    pr.add_argument("input")
    pr.add_argument("--out", required=True)
    pr.add_argument("--config", default=None, help="TOML overriding configs/default.toml")
    pr.add_argument("--units", default="auto", choices=["auto", "m", "us-ft", "ft"])
    pr.add_argument("--no-preview", action="store_true")
    pr.set_defaults(func=_cmd_run)

    ps = sub.add_parser("synth", help="synthetic forest with known stems: run v0 and score it")
    ps.add_argument("--out", required=True)
    ps.add_argument("--density", type=float, default=2.0, help="pulses per m2 (2 = QL2, 8 = QL1)")
    ps.add_argument("--size", type=float, default=120.0)
    ps.add_argument("--trees", type=int, default=60)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--config", default=None)
    ps.add_argument("--no-preview", action="store_true")
    ps.set_defaults(func=_cmd_synth)

    from .stage1.cli import add_parser as add_stage1
    add_stage1(sub)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

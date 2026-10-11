"""Command line: `trailgen inspect | run | synth`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _cmd_inspect(args) -> int:
    from .io import read_points
    from .quality import density_report

    pc = read_points(args.input, units=args.units)
    print(json.dumps(density_report(pc), indent=2))
    return 0


def _cmd_run(args) -> int:
    from .config import load_config
    from .io import read_points
    from .pipeline import process, save

    cfg = load_config(args.config)
    pc = read_points(args.input, units=args.units)
    print(f"Read {len(pc):,} points from {args.input}")
    result = process(pc, cfg)
    out = save(result, args.out, pc.crs_wkt, preview=not args.no_preview)
    s = result.summary
    print(f"Tile: {s['tile']['first_returns_per_m2']} pulses/m2, {s['tile']['estimated_quality_level']}")
    print(f"Trees: {s['trees']['count']}, lidar points per crown (median): "
          f"{(s['trees']['lidar_points_per_crown'] or {}).get('median')}")
    print(f"Blocked: {s['traversability']['blocked_fraction']:.1%}, "
          f"low ground confidence: {s['traversability']['low_confidence_fraction']:.1%}")
    print(f"Outputs in {out}/")
    return 0


def _cmd_synth(args) -> int:
    from .config import load_config
    from .evaluate import match_trunks, recall_by_height
    from .io import save_points_npz
    from .pipeline import process, save
    from .synthetic import make_scene

    scene = make_scene(size=args.size, n_trees=args.trees, pulses_per_m2=args.density, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_points_npz(scene.points, out / "points.npz")
    np.savetxt(out / "truth_trees.csv", scene.trees, delimiter=",",
               header="x,y,height,crown_radius,dbh", comments="", fmt="%.3f")

    result = process(scene.points, load_config(args.config))
    save(result, out, preview=not args.no_preview)
    pred_xy = np.array([[t.x, t.y] for t in result.trees]).reshape(-1, 2)
    pred_dbh = np.array([t.dbh for t in result.trees])
    report = {
        "density_pulses_per_m2": args.density,
        "trunks_within_0.5m": match_trunks(pred_xy, pred_dbh, scene.trees[:, :2], scene.trees[:, 4], 0.5),
        "trunks_within_1.0m": match_trunks(pred_xy, pred_dbh, scene.trees[:, :2], scene.trees[:, 4], 1.0),
        "recall_by_height_1.0m": recall_by_height(pred_xy, scene.trees[:, :2], scene.trees[:, 2], max_dist=1.0),
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Outputs in {out}/")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="trailgen", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="report point density and quality level of a tile")
    pi.add_argument("input")
    pi.add_argument("--units", default="auto", choices=["auto", "m", "us-ft", "ft"])
    pi.set_defaults(func=_cmd_inspect)

    pr = sub.add_parser("run", help="process a LAS/LAZ tile into terrain, trees and a cost map")
    pr.add_argument("input")
    pr.add_argument("--out", required=True)
    pr.add_argument("--config", default=None, help="TOML file overriding configs/default.toml")
    pr.add_argument("--units", default="auto", choices=["auto", "m", "us-ft", "ft"])
    pr.add_argument("--no-preview", action="store_true")
    pr.set_defaults(func=_cmd_run)

    ps = sub.add_parser("synth", help="synthetic forest with known trees: run the pipeline and score it")
    ps.add_argument("--out", required=True)
    ps.add_argument("--density", type=float, default=2.0, help="lidar pulses per m2 (2 = QL2, 8 = QL1)")
    ps.add_argument("--size", type=float, default=120.0)
    ps.add_argument("--trees", type=int, default=60)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--config", default=None)
    ps.add_argument("--no-preview", action="store_true")
    ps.set_defaults(func=_cmd_synth)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

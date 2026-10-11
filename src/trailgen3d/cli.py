"""`trailgen run | synth`: aerial tile -> Sky2Trail ground data -> navigation world."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _run(pc, out: Path, args, crs_wkt=None) -> int:
    from sky2trail.baseline import run_v0
    from sky2trail.config import load_config as load_s2t
    from sky2trail.viz import save_preview

    from .config import load_config
    from .pipeline import build_world, save_world

    s2t = run_v0(pc, load_s2t(args.config))
    s2t.save(out / "sky2trail", crs_wkt)
    if not args.no_preview:
        save_preview(s2t, out / "sky2trail" / "preview.png")
    world = build_world(s2t, load_config(args.config))
    save_world(world, out / "world", crs_wkt, preview=not args.no_preview)
    w = world.summary
    print(f"Sky2Trail ({s2t.method}): {len(s2t.stems)} stems, {len(s2t.points):,} points")
    print(f"World: {w['blocked_fraction']:.1%} blocked, {w['low_confidence_fraction']:.1%} low ground confidence")
    print(f"Outputs in {out}/sky2trail and {out}/world")
    return 0


def _cmd_run(args) -> int:
    from sky2trail.io import read_points

    pc = read_points(args.input, units=args.units)
    print(f"Read {len(pc):,} points from {args.input}")
    return _run(pc, Path(args.out), args, pc.crs_wkt)


def _cmd_synth(args) -> int:
    from sky2trail.synthetic import make_scene

    scene = make_scene(size=args.size, n_trees=args.trees, pulses_per_m2=args.density, seed=args.seed)
    return _run(scene.points, Path(args.out), args)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="trailgen", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="aerial tile -> navigation world (uses Sky2Trail v0)")
    pr.add_argument("input")
    pr.add_argument("--out", required=True)
    pr.add_argument("--config", default=None, help="TOML overriding configs/default.toml")
    pr.add_argument("--units", default="auto", choices=["auto", "m", "us-ft", "ft"])
    pr.add_argument("--no-preview", action="store_true")
    pr.set_defaults(func=_cmd_run)

    ps = sub.add_parser("synth", help="same, on a synthetic forest")
    ps.add_argument("--out", required=True)
    ps.add_argument("--density", type=float, default=2.0)
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

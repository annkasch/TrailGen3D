"""`sky2trail stage1 data-synth | train | eval-synth | run`."""

from __future__ import annotations

import json
from pathlib import Path


def _data_synth(args) -> int:
    from ..config import load_config
    from .dataset import make_synthetic_dataset

    make_synthetic_dataset(args.out, load_config(args.config), n_train=args.train_scenes,
                           n_val=args.val_scenes, size=args.size, min_density=args.min_density,
                           max_density=args.max_density, seed=args.seed)
    return 0


def _train(args) -> int:
    from .train import train

    res = train(args.data, args.out, epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                base=args.base, workers=args.workers, device=args.device, resume=args.resume,
                seed=args.seed)
    print(json.dumps(res))
    return 0


def _eval_synth(args) -> int:
    from ..config import load_config
    from .predict import evaluate_synthetic, format_report

    report = evaluate_synthetic(args.model, load_config(args.config), args.out, n_scenes=args.scenes,
                                densities=tuple(args.densities), size=args.size, device=args.device)
    print(format_report(report))
    print(f"Details and previews in {args.out}/")
    return 0


def _run(args) -> int:
    from ..config import load_config
    from ..io import read_points
    from .predict import Stage1Model, run_stage1

    pc = read_points(args.input, units=args.units)
    out = run_stage1(pc, load_config(args.config), Stage1Model(args.model, args.device))
    out.save(args.out, pc.crs_wkt)
    if not args.no_preview:
        from ..viz import save_preview
        save_preview(out, Path(args.out) / "preview.png")
    print(f"{len(out.stems)} stems ({out.method}); outputs in {args.out}/")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("stage1", help="learned stem map: build data, train, evaluate, run")
    s = p.add_subparsers(dest="stage1_cmd", required=True)

    d = s.add_parser("data-synth", help="training windows from hard synthetic forests")
    d.add_argument("--out", required=True)
    d.add_argument("--train-scenes", type=int, default=40)
    d.add_argument("--val-scenes", type=int, default=8)
    d.add_argument("--size", type=float, default=100.0, help="scene side, meters")
    d.add_argument("--min-density", type=float, default=2.0, help="pulses/m2")
    d.add_argument("--max-density", type=float, default=15.0)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--config", default=None)
    d.set_defaults(func=_data_synth)

    t = s.add_parser("train", help="train the stem U-Net (needs PyTorch)")
    t.add_argument("--data", required=True, help="folder with train.npz and val.npz")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--base", type=int, default=32, help="width of the first U-Net level")
    t.add_argument("--workers", type=int, default=2, help="data-loading processes")
    t.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    t.add_argument("--resume", action="store_true", help="continue from OUT/last.pt")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=_train)

    e = s.add_parser("eval-synth", help="stage 1 vs v0 on held-out synthetic forests")
    e.add_argument("--model", required=True, help="checkpoint, e.g. runs/x/best.pt")
    e.add_argument("--out", required=True)
    e.add_argument("--scenes", type=int, default=5)
    e.add_argument("--densities", type=float, nargs="+", default=[2.0, 8.0])
    e.add_argument("--size", type=float, default=100.0)
    e.add_argument("--device", default="auto")
    e.add_argument("--config", default=None)
    e.set_defaults(func=_eval_synth)

    r = s.add_parser("run", help="aerial tile -> Sky2Trail output with learned stems")
    r.add_argument("input")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="auto")
    r.add_argument("--config", default=None)
    r.add_argument("--units", default="auto", choices=["auto", "m", "us-ft", "ft"])
    r.add_argument("--no-preview", action="store_true")
    r.set_defaults(func=_run)

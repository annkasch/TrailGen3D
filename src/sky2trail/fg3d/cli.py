"""`sky2trail fg3d synth-pairs | register | pairs | train | sample | eval`."""

from __future__ import annotations

import json


def _synth_pairs(args) -> int:
    from .synthetic import write_pairs

    paths = write_pairs(args.out, args.n, seed=args.seed, pulses_per_m2=args.density)
    print(f"{len(paths)} pairs and split.json in {args.out}/")
    return 0


def _train(args) -> int:
    from .train import TrainCfg, train

    cfg = TrainCfg(epochs=args.epochs, batch_size=args.batch, lr=args.lr, workers=args.workers,
                   seed=args.seed, tiny=args.tiny, sample_every=args.sample_every,
                   augment=not args.no_augment, n_als=args.n_als, n_tls=args.n_tls,
                   thin=tuple(args.thin) if args.thin else None)
    res = train(args.data, args.out, cfg, device=args.device, resume=args.resume,
                train_split=args.train_split, val_split=args.val_split or None)
    print(json.dumps(res))
    return 0


def _sample(args) -> int:
    from .pairs import list_pairs
    from .sample import Fg3dModel, sample_pairs, save_preview

    files = list_pairs(args.data, args.split)
    if args.limit:
        files = files[:args.limit]
    model = Fg3dModel(args.model, args.device)
    sample_pairs(model, files, args.out, batch=args.batch, n_samples=args.samples,
                 sampler=args.sampler, steps=args.steps)
    print(f"Preview: {save_preview(args.out, f'{args.out}/preview.png')}")
    return 0


def _register(args) -> int:
    from ..register import register_plots

    regs = register_plots(args.root, args.flight, args.plots or None)
    print(f"{sum(r.ok for r in regs.values())}/{len(regs)} scans pass; "
          f"check {args.root}/registration/{args.flight}/*.png")
    return 0


def _pairs(args) -> int:
    from .build import BuildCfg, build_pairs

    cfg = BuildCfg(max_dist=args.max_dist, min_height=args.min_height, n_tls=args.n_tls, seed=args.seed)
    res = build_pairs(args.root, args.out, tuple(args.flights), cfg, only_ok=not args.all)
    print(json.dumps(res))
    return 0


def _eval(args) -> int:
    from .sample import evaluate

    print(json.dumps(evaluate(args.samples, exact_emd=not args.fast), indent=1))
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("fg3d", help="ForestGen3D: generate TLS-like trees from ALS (diffusion)")
    s = p.add_subparsers(dest="fg3d_cmd", required=True)

    d = s.add_parser("synth-pairs", help="synthetic ALS/TLS tree pairs for development")
    d.add_argument("--out", required=True)
    d.add_argument("--n", type=int, default=500)
    d.add_argument("--density", type=float, default=15.0, help="ALS pulses per m2")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=_synth_pairs)

    r = s.add_parser("register", help="register WFSI TLS scans to the ALS (heading + shift)")
    r.add_argument("--root", required=True, help="a `sky2trail wfsi fetch` folder with tls_npz/")
    r.add_argument("--flight", choices=["pre", "post"], default="pre")
    r.add_argument("--plots", nargs="*", help="plot names (default: all converted scans)")
    r.set_defaults(func=_register)

    b = s.add_parser("pairs", help="per-tree ALS/TLS pairs from registered scans (+ split.json)")
    b.add_argument("--root", required=True, help="folder with tls_npz/, als_crops/, registration/")
    b.add_argument("--out", required=True)
    b.add_argument("--flights", nargs="+", default=["pre"], choices=["pre", "post"])
    b.add_argument("--max-dist", type=float, default=20.0, help="treetop distance from the scanner, m")
    b.add_argument("--min-height", type=float, default=5.0, help="skip lower trees, m")
    b.add_argument("--n-tls", type=int, default=2048)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--all", action="store_true", help="also use scans whose registration is flagged")
    b.set_defaults(func=_pairs)

    t = s.add_parser("train", help="train the diffusion model on pairs (with split.json)")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--n-als", type=int, default=512)
    t.add_argument("--n-tls", type=int, default=2048)
    t.add_argument("--thin", type=float, nargs=2, metavar=("LO", "HI"),
                   help="ALS density augmentation: keep fraction drawn from [LO, HI]")
    t.add_argument("--no-augment", action="store_true")
    t.add_argument("--train-split", default="train")
    t.add_argument("--val-split", default="val", help="'' for none")
    t.add_argument("--sample-every", type=int, default=10, help="epochs between DDIM checks (0: never)")
    t.add_argument("--tiny", action="store_true", help="small model (quick checks)")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--device", default="auto")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", action="store_true")
    t.set_defaults(func=_train)

    g = s.add_parser("sample", help="generate trees for pairs with a checkpoint")
    g.add_argument("--model", required=True)
    g.add_argument("--data", required=True)
    g.add_argument("--split", default="test", help="split name in split.json, or 'all'")
    g.add_argument("--out", required=True)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--samples", type=int, default=1, help="samples per tree")
    g.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    g.add_argument("--steps", type=int, default=50, help="DDIM steps")
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--device", default="auto")
    g.set_defaults(func=_sample)

    e = s.add_parser("eval", help="CD / EMD / EPC, height and DBH of generated trees")
    e.add_argument("--samples", required=True)
    e.add_argument("--fast", action="store_true", help="skip the exact EMD")
    e.set_defaults(func=_eval)

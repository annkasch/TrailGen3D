"""Train the ForestGen3D diffusion model on tree pairs (PyTorch).

    sky2trail fg3d train --data data/fg3d_synth --out runs/fg3d_synth

Saves last.pt every epoch (model, EMA, optimizer, scaler, for --resume) and best.pt (EMA weights)
on the lowest validation loss. Validation loss uses fixed timesteps and noise, so epochs compare.
Every `sample_every` epochs a few validation trees are generated with DDIM and scored by Chamfer
distance. Runs on CUDA (with mixed precision), Apple MPS or CPU.
"""

from __future__ import annotations

import copy
import functools
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..stage1.train import _autocast, _grad_scaler, lr_factor, pick_device
from .diffusion import Diffusion
from .metrics import chamfer_torch
from .model import Fg3dNet, ModelCfg, count_parameters, tiny_cfg
from .pairs import PairDataset, list_pairs


CLIP = (-0.25, 1.25)        # predicted x_0 range while sampling: the unit tree box plus a margin


@dataclass
class TrainCfg:
    epochs: int = 100
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.0
    ema: float = 0.999
    grad_clip: float = 1.0
    n_als: int = 512
    n_tls: int = 2048
    augment: bool = True
    thin: tuple | None = None            # ALS keep-fraction range, e.g. (0.2, 1.0)
    steps: int = 1000                    # diffusion steps
    sample_every: int = 10               # epochs between DDIM sample checks (0: never)
    sample_trees: int = 8
    ddim_steps: int = 50
    workers: int = 2
    seed: int = 0
    tiny: bool = False                   # small widths (tests, quick checks)
    model: dict = field(default_factory=dict)   # ModelCfg overrides


def _model_cfg(cfg: TrainCfg) -> ModelCfg:
    base = tiny_cfg() if cfg.tiny else ModelCfg()
    for k, v in cfg.model.items():
        setattr(base, k, tuple(v) if isinstance(v, list) else v)
    # SA levels can't sample more centres than there are points
    base.npoints = tuple(min(p, cfg.n_tls) for p in base.npoints)
    base.cond_npoints = tuple(min(p, cfg.n_als) for p in base.cond_npoints)
    return base


def build_model(ck: dict) -> Fg3dNet:
    """Model from a checkpoint's stored ModelCfg (weights not loaded)."""
    mc = ModelCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["model_cfg"].items()})
    return Fg3dNet(mc)


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)


def run_info(device) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        commit = ""
    return {"git_commit": commit, "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "device": str(device), "host": platform.node()}


@torch.no_grad()
def val_loss(model, diff: Diffusion, loader, device, amp: bool, seed: int = 1234) -> float:
    """Epsilon MSE at fixed timesteps and noise."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    losses = []
    for tls, als in loader:
        t = torch.randint(0, diff.steps, (tls.shape[0],), generator=g)
        noise = torch.randn(tls.shape, generator=g)
        tls, als, t, noise = tls.to(device), als.to(device), t.to(device), noise.to(device)
        with _autocast(device, amp):
            eps = model(diff.q_sample(tls, t, noise), t, als)
        losses.append(float(torch.mean((eps.float() - noise) ** 2)))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def sample_check(model, diff: Diffusion, ds: PairDataset, n: int, device, ddim_steps: int) -> float:
    """Mean Chamfer distance of DDIM samples vs targets for the first n pairs."""
    model.eval()
    n = min(n, len(ds))
    if n == 0:
        return float("nan")
    tls, als = zip(*(ds[i] for i in range(n)))
    tls, als = torch.stack(tls).to(device), torch.stack(als).to(device)
    x = _ddim(model, diff, als, tls.shape, device, ddim_steps)
    return float(chamfer_torch(x, tls).mean())


def _ddim(model, diff: Diffusion, als: torch.Tensor, shape, device, steps: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(tuple(shape), generator=g).to(device)
    cond = model.encode_cond(als)
    return diff.sample_ddim(model, tuple(shape), cond, device=device, steps=steps, noise=noise, clip=CLIP)


def train(data_dir: str, out_dir: str, cfg: TrainCfg | None = None, device: str = "auto",
          resume: bool = False, train_split: str = "train", val_split: str = "val",
          log=None) -> dict:
    log = log or functools.partial(print, flush=True)
    cfg = cfg or TrainCfg()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)
    amp = dev.type == "cuda"

    train_files = list_pairs(data_dir, train_split)
    val_files = list_pairs(data_dir, val_split) if val_split else []
    train_ds = PairDataset(train_files, cfg.n_als, cfg.n_tls, augment=cfg.augment,
                           thin=tuple(cfg.thin) if cfg.thin else None, seed=cfg.seed)
    val_ds = PairDataset(val_files, cfg.n_als, cfg.n_tls, augment=False, seed=cfg.seed)
    pin = dev.type == "cuda"
    g = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(train_ds, batch_size=min(cfg.batch_size, len(train_ds)), shuffle=True,
                          num_workers=cfg.workers, pin_memory=pin, drop_last=True,
                          persistent_workers=cfg.workers > 0, worker_init_fn=_seed_worker, generator=g)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    mcfg = _model_cfg(cfg)
    model = Fg3dNet(mcfg).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    diff = Diffusion(cfg.steps)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = _grad_scaler(amp)
    meta = {"train_cfg": asdict(cfg), "model_cfg": asdict(mcfg), "data": str(data_dir), **run_info(dev)}

    start, best = 0, float("inf")
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        log(f"Resumed from epoch {start}")
    log(f"Device {dev} | {count_parameters(model) / 1e6:.2f} M parameters | "
        f"{len(train_ds)} train / {len(val_ds)} val pairs")
    (out / "config.json").write_text(json.dumps(meta, indent=2))

    steps_per_epoch = max(1, len(train_dl))
    total = cfg.epochs * steps_per_epoch
    rec = {}
    for epoch in range(start, cfg.epochs):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for step, (tls, als) in enumerate(train_dl):
            for grp in opt.param_groups:
                grp["lr"] = cfg.lr * lr_factor(epoch * steps_per_epoch + step, total)
            tls, als = tls.to(dev, non_blocking=True), als.to(dev, non_blocking=True)
            with _autocast(dev, amp):
                loss = diff.loss(model, tls, als)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            # EMA decay ramps up from 0, so early EMA weights follow the model (short runs work too)
            gstep = epoch * steps_per_epoch + step
            decay = min(cfg.ema, (1 + gstep) / (10 + gstep))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - decay)
            run += float(loss.detach())
            n += 1
        rec = {"epoch": epoch, "seconds": round(time.time() - t0, 1), "lr": opt.param_groups[0]["lr"],
               "train_loss": round(run / max(n, 1), 5)}
        if len(val_ds):
            rec["val_loss"] = round(val_loss(ema, diff, val_dl, dev, amp), 5)
        last = epoch + 1 == cfg.epochs
        if cfg.sample_every and ((epoch + 1) % cfg.sample_every == 0 or last):
            ds = val_ds if len(val_ds) else PairDataset(train_files, cfg.n_als, cfg.n_tls, seed=cfg.seed)
            rec["val_cd"] = sample_check(ema, diff, ds, cfg.sample_trees, dev, cfg.ddim_steps)
        with open(out / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        score = rec.get("val_loss", rec["train_loss"])
        improved = score < best
        best = min(best, score)
        ck = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
              "scaler": scaler.state_dict(), "epoch": epoch, "best": best, **meta}
        torch.save(ck, out / "last.pt")
        if improved:
            torch.save({"model": ema.state_dict(), "epoch": epoch, "metrics": rec, **meta}, out / "best.pt")
        log(f"epoch {epoch + 1:3d}/{cfg.epochs} {rec['seconds']:6.1f}s  loss {rec['train_loss']:.4f}"
            + (f"  val {rec['val_loss']:.4f}" if "val_loss" in rec else "")
            + (f"  CD {rec['val_cd']:.2e}" if "val_cd" in rec else "")
            + ("  *best" if improved else ""))
    return {"best": best, "out": str(out), **rec}

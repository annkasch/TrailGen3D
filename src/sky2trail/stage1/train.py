"""Train stage 1 (PyTorch). Saves last.pt every epoch and best.pt on the best validation F1.

    sky2trail stage1 train --data data/stage1_synth --out runs/stage1_synth

Validation decodes every validation window, keeps stems in its core, and matches them to the
known stems within 1 m (and 0.5 m), at several heatmap thresholds; the best threshold is
stored in the checkpoint and used for prediction. After the last epoch, the last-layer
Laplace approximation (laplace.py) is fitted on best.pt and stored in it.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..evaluate import match_trunks
from .dataset import StemSamples
from .decode import decode
from .features import window_grid
from .model import StemUNet, count_parameters, stage1_loss
from .targets import pixel_to_world

THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6)
KEYS = ("image", "heat", "offset", "size", "size_mask", "reg_mask", "weight")


class TorchSamples(Dataset):
    def __init__(self, samples: StemSamples, augment: bool):
        self.s, self.augment = samples, augment

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)))) if self.augment else None
        item = self.s.get(i, rng)
        out = {k: torch.from_numpy(np.ascontiguousarray(item[k])) for k in KEYS}
        out["index"] = i
        return out


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_factor(step: int, total: int, warmup: float = 0.05) -> float:
    """Linear warm-up over the first 5 % of steps, then cosine decay to 1 %.

    Depends only on the step number, so a resumed run (even with more epochs) continues
    smoothly without stored scheduler state.
    """
    w = max(1, int(warmup * total))
    if step < w:
        return (step + 1) / w
    t = min(1.0, (step - w) / max(1, total - w))
    return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * t))


def _autocast(device: torch.device, enabled: bool):
    """Mixed precision on CUDA only; plain float32 elsewhere (MPS/CPU)."""
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else contextlib.nullcontext()


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def core_truth(samples: StemSamples, i: int, grid) -> np.ndarray:
    """Known stems inside the core of sample i: world x, y, dbh, height."""
    st = samples.stems_of(i)
    r0, c0, r1, c1 = samples.core_px[i]
    sel = (st[:, 0] >= r0) & (st[:, 0] < r1) & (st[:, 1] >= c0) & (st[:, 1] < c1)
    # Only stems in labelled area count (all of them for synthetic scenes).
    st = st[sel]
    x, y = pixel_to_world(st[:, 0], st[:, 1], grid)
    return np.column_stack([x, y, st[:, 2], st[:, 3]])


def core_bounds(samples: StemSamples, i: int, grid) -> tuple:
    r0, c0, r1, c1 = samples.core_px[i]
    x0, y1 = pixel_to_world(r0, c0, grid)
    x1, y0 = pixel_to_world(r1, c1, grid)
    return float(x0), float(y0), float(x1), float(y1)


@torch.no_grad()
def validate(model, loader, samples: StemSamples, device, amp: bool) -> dict:
    model.eval()
    cfg = samples.cfg
    totals = {(t, d): [0, 0, 0] for t in THRESHOLDS for d in (0.5, 1.0)}  # matched, true, pred
    dbh_err = {t: [] for t in THRESHOLDS}
    losses = []
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "index"}
        with _autocast(device, amp):
            out = model(b["image"])
        losses.append(float(stage1_loss(out, b)["total"]))
        heat = torch.sigmoid(out["heat"].float()).cpu().numpy()
        off = out["offset"].float().cpu().numpy()
        size = out["size"].float().cpu().numpy()
        for j, i in enumerate(batch["index"].tolist()):
            grid = window_grid(samples.bounds[i], cfg.resolution)
            truth = core_truth(samples, i, grid)
            cb = core_bounds(samples, i, grid)
            for t in THRESHOLDS:
                d = decode(heat[j], off[j], size[j], grid, cfg, core_bounds=cb, threshold=t)
                pred = np.column_stack([d["x"], d["y"]])
                for dist in (0.5, 1.0):
                    m = match_trunks(pred, d["dbh"], truth[:, :2], truth[:, 2], dist)
                    acc = totals[(t, dist)]
                    acc[0] += m["matched"]; acc[1] += len(truth); acc[2] += len(pred)
                    if dist == 1.0 and m["matched"]:
                        dbh_err[t].append((m["mean_abs_dbh_error"], m["matched"]))

    def prf(acc):
        mt, nt, npred = acc
        r = mt / max(nt, 1); p = mt / max(npred, 1)
        return r, p, (2 * r * p / (r + p) if mt else 0.0)

    best_t = max(THRESHOLDS, key=lambda t: prf(totals[(t, 1.0)])[2])
    r1, p1, f1 = prf(totals[(best_t, 1.0)])
    r05, p05, f05 = prf(totals[(best_t, 0.5)])
    errs = dbh_err[best_t]
    mae = sum(e * n for e, n in errs) / max(sum(n for _, n in errs), 1) if errs else None
    return {"val_loss": float(np.mean(losses)), "threshold": best_t,
            "recall_1m": round(r1, 3), "precision_1m": round(p1, 3), "f1_1m": round(f1, 3),
            "recall_0.5m": round(r05, 3), "precision_0.5m": round(p05, 3), "f1_0.5m": round(f05, 3),
            "dbh_mae_m": None if mae is None else round(mae, 3)}


def train(data_dir: str, out_dir: str, epochs: int = 60, batch_size: int = 32, lr: float = 1e-3,
          base: int = 32, workers: int = 2, device: str = "auto", resume: bool = False,
          seed: int = 0) -> dict:
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_s = StemSamples(Path(data_dir) / "train.npz")
    val_s = StemSamples(Path(data_dir) / "val.npz")
    dev = pick_device(device)
    amp = dev.type == "cuda"
    model = StemUNet(train_s.n_channels, base=base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = _grad_scaler(amp)
    pin = dev.type == "cuda"
    train_dl = DataLoader(TorchSamples(train_s, augment=True), batch_size=batch_size, shuffle=True,
                          num_workers=workers, pin_memory=pin, drop_last=True,
                          persistent_workers=workers > 0)
    val_dl = DataLoader(TorchSamples(val_s, augment=False), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=pin)
    meta = {"stage1": asdict(train_s.cfg), "channels": train_s.meta["channels"],
            "model_args": {"in_channels": train_s.n_channels, "base": base, "depth": 4},
            "data": str(data_dir)}
    start, best_f1 = 0, -1.0
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start, best_f1 = ck["epoch"] + 1, ck["best_f1"]
        print(f"Resumed from epoch {start}")
    print(f"Device {dev} | {count_parameters(model) / 1e6:.1f} M parameters | "
          f"{len(train_s)} train / {len(val_s)} val windows | {train_s.n_channels} channels")
    (out / "config.json").write_text(json.dumps(meta, indent=2))

    log_path = out / "metrics.jsonl"
    steps_per_epoch = max(1, len(train_dl))
    total_steps = epochs * steps_per_epoch
    for epoch in range(start, epochs):
        model.train()
        t0, run = time.time(), {"total": 0.0, "heat": 0.0, "offset": 0.0, "size": 0.0}
        n = 0
        for step, batch in enumerate(train_dl):
            for g in opt.param_groups:
                g["lr"] = lr * lr_factor(epoch * steps_per_epoch + step, total_steps)
            b = {k: v.to(dev, non_blocking=True) for k, v in batch.items() if k != "index"}
            with _autocast(dev, amp):
                outp = model(b["image"])
            losses = stage1_loss(outp, b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            for k in run:
                run[k] += float(losses[k])
            n += 1
        train_metrics = {f"train_{k}": round(v / max(n, 1), 4) for k, v in run.items()}
        val = validate(model, val_dl, val_s, dev, amp)
        rec = {"epoch": epoch, "seconds": round(time.time() - t0, 1),
               "lr": opt.param_groups[0]["lr"], **train_metrics, **val}
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        improved = val["f1_1m"] > best_f1
        best_f1 = max(best_f1, val["f1_1m"])
        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "scaler": scaler.state_dict(), "epoch": epoch, "best_f1": best_f1,
              "threshold": val["threshold"], **meta}
        torch.save(ck, out / "last.pt")
        if improved:
            torch.save({k: ck[k] for k in ("model", "epoch", "threshold", "stage1", "channels", "model_args")}
                       | {"val": val}, out / "best.pt")
        print(f"epoch {epoch + 1:3d}/{epochs} {rec['seconds']:5.1f}s  loss {train_metrics['train_total']:.3f}"
              f"  val: recall {val['recall_1m']:.2f} precision {val['precision_1m']:.2f}"
              f" F1 {val['f1_1m']:.3f} (1 m, thr {val['threshold']})  dbh err {val['dbh_mae_m']}"
              + ("  *best" if improved else ""))
    if (out / "best.pt").exists():
        add_laplace(out / "best.pt", train_s, val_s, dev, batch_size)
    return {"best_f1_1m": best_f1, "out": str(out)}


def add_laplace(checkpoint: Path, train_s: StemSamples, val_s: StemSamples, dev, batch_size: int) -> None:
    """Fit the last-layer Laplace approximation on the checkpoint's weights and store it there."""
    from .laplace import fit_laplace

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    a = ck["model_args"]
    model = StemUNet(a["in_channels"], base=a["base"], depth=a["depth"]).to(dev)
    model.load_state_dict(ck["model"])
    ck["laplace"] = fit_laplace(model, train_s, val_s, dev, batch_size)
    torch.save(ck, checkpoint)

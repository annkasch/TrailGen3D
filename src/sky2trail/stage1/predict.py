"""Run a trained stage 1 model on a whole tile, and compare it with v0 on synthetic scenes."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from ..baseline import assemble, measure, run_v0
from ..config import Sky2TrailConfig, Stage1Cfg
from ..evaluate import match_trunks, recall_by_height
from ..io import PointCloud
from ..output import Sky2TrailOutput
from ..patches import tile_patches
from .dataset import window_samples
from .decode import decode, to_stems
from .features import window_grid
from .laplace import std_maps
from .model import StemUNet
from .train import _autocast, pick_device

METHOD = "stage1-unet"


def _undo(m: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Map a (B, C, H, W) output of a view (rotated k quarter turns, then mirrored) back."""
    if flip:
        m = m.flip(-1)
    return torch.rot90(m, -k, dims=(-2, -1))


def _undo_offset(o: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Like _undo, and also turn the (dv, du) vectors back: mirror undoes du, rotation (dv, du) -> (du, -dv)."""
    o = _undo(o, k, flip)
    dv, du = o[:, 0], o[:, 1]
    if flip:
        du = -du
    for _ in range(k % 4):
        dv, du = du, -dv
    return torch.stack([dv, du], 1)


def _undo_offset_std(s: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Per-axis std of the offset: a quarter turn swaps the axes, signs do not matter."""
    s = _undo(s, k, flip)
    return s.flip(1) if k % 2 else s


def forward_maps(net, x: torch.Tensor, laplace: dict | None = None, tta: bool = True) -> dict:
    """heat (probabilities), offset, size; with Laplace also offset_std and size_std.

    tta: average over the 8 rotations and mirrors of the window. The std then combines the
    Laplace variance of each view with the disagreement between views (law of total variance).
    """
    views = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
    preds = {"heat": [], "offset": [], "size": []}
    var = {"offset": [], "size": []}
    for k, flip in views:
        xt = torch.rot90(x, k, dims=(-2, -1))
        if flip:
            xt = xt.flip(-1)
        out = net(xt, features=laplace is not None)
        preds["heat"].append(_undo(torch.sigmoid(out["heat"].float()), k, flip))
        preds["offset"].append(_undo_offset(out["offset"].float(), k, flip))
        preds["size"].append(_undo(out["size"].float(), k, flip))
        if laplace is not None:
            std = std_maps(out, laplace)
            var["offset"].append(_undo_offset_std(std["offset"], k, flip) ** 2)
            var["size"].append(_undo(std["size"], k, flip) ** 2)
    res = {key: torch.stack(v).mean(0) for key, v in preds.items()}
    if laplace is not None:
        for h in var:
            spread = torch.stack(preds[h]).var(0, unbiased=False)
            res[f"{h}_std"] = (torch.stack(var[h]).mean(0) + spread).sqrt()
    return res


class Stage1Model:
    def __init__(self, checkpoint: str | Path, device: str = "auto", tta: bool = True):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = Stage1Cfg(**ck["stage1"])
        self.threshold = float(ck.get("threshold", self.cfg.peak_threshold))
        self.channels = ck["channels"]
        self.device = pick_device(device)
        self.net = StemUNet(ck["model_args"]["in_channels"], base=ck["model_args"]["base"],
                            depth=ck["model_args"]["depth"])
        self.net.load_state_dict(ck["model"])
        self.net.to(self.device).eval()
        self.checkpoint = str(checkpoint)
        self.val = ck.get("val")
        self.laplace = ck.get("laplace")  # None for checkpoints trained before it existed
        self.tta = tta

    @torch.no_grad()
    def maps(self, images: np.ndarray, batch: int = 32) -> dict:
        """Output maps as numpy arrays, see forward_maps."""
        res = {}
        amp = self.device.type == "cuda"
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i:i + batch].astype(np.float32)).to(self.device)
            with _autocast(self.device, amp):
                out = forward_maps(self.net, x, self.laplace, self.tta)
            for k, v in out.items():
                res.setdefault(k, []).append(v.cpu().numpy())
        return {k: np.concatenate(v) for k, v in res.items()}


def predict_stems(pc: PointCloud, hag: np.ndarray, model: Stage1Model, cfg: Sky2TrailConfig) -> dict:
    """Stems for a whole tile: windows over the tile, each keeps the stems in its core."""
    cfg = copy.deepcopy(cfg)
    cfg.stage1 = model.cfg  # rasterize exactly as during training
    patches = tile_patches(pc.bounds, cfg.patches.core_size, cfg.patches.margin)
    samples = window_samples(pc, hag, patches, None, None, False, cfg)
    m = model.maps(np.stack([s["image"] for s in samples]))
    parts = []
    for k, (s, p) in enumerate(zip(samples, patches)):
        grid = window_grid(s["bounds"], model.cfg.resolution)
        std = (m["offset_std"][k], m["size_std"][k]) if "offset_std" in m else None
        parts.append(decode(m["heat"][k], m["offset"][k], m["size"][k], grid, model.cfg,
                            core_bounds=p.core_bounds, threshold=model.threshold, std=std))
    return {key: np.concatenate([d[key] for d in parts]) for key in parts[0]}


def run_stage1(pc: PointCloud, cfg: Sky2TrailConfig, model: Stage1Model) -> Sky2TrailOutput:
    m = measure(pc, cfg)
    d = predict_stems(pc, m.hag, model, cfg)
    stems = to_stems(d)
    return assemble(pc, cfg, m, stems, METHOD, {
        "stem_rule": "learned stem map (stage 1 U-Net)",
        "checkpoint": model.checkpoint, "threshold": model.threshold,
        "uncertainty": "last-layer Laplace" if model.laplace is not None else None,
        "test_time_augmentation": model.tta,
        "small_stems": int(d["small"].sum()),
    })


def _score(out: Sky2TrailOutput, trees: np.ndarray) -> dict:
    pred = np.array([[s.x, s.y] for s in out.stems]).reshape(-1, 2)
    dbh = np.array([s.dbh for s in out.stems])
    return {
        "1.0m": match_trunks(pred, dbh, trees[:, :2], trees[:, 4], 1.0),
        "0.5m": match_trunks(pred, dbh, trees[:, :2], trees[:, 4], 0.5),
        "recall_by_height_1.0m": recall_by_height(pred, trees[:, :2], trees[:, 2], max_dist=1.0),
    }


def _pool(scores: list[dict], key: str) -> dict:
    mt = sum(s[key]["matched"] for s in scores)
    nt = sum(s[key]["n_true"] for s in scores)
    npred = sum(s[key]["n_pred"] for s in scores)
    r, p = mt / max(nt, 1), mt / max(npred, 1)
    errs = [(s[key]["mean_abs_dbh_error"], s[key]["matched"]) for s in scores if s[key]["matched"]]
    dbh = sum(e * n for e, n in errs) / max(sum(n for _, n in errs), 1) if errs else None
    return {"recall": round(r, 3), "precision": round(p, 3),
            "f1": round(2 * r * p / (r + p), 3) if mt else 0.0,
            "dbh_mae_m": None if dbh is None else round(dbh, 3), "n_true": nt}


def evaluate_synthetic(checkpoint: str, cfg: Sky2TrailConfig, out_dir: str, n_scenes: int = 5,
                       densities=(2.0, 8.0), size: float = 100.0, device: str = "auto",
                       tta: bool = True) -> dict:
    """v0 and stage 1 on held-out hard scenes (seeds 2000+, never used for training)."""
    from ..synthetic import make_hard_scene
    from ..viz import save_preview

    model = Stage1Model(checkpoint, device, tta=tta)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": checkpoint, "threshold": model.threshold, "scenes": n_scenes, "results": {}}
    for d in densities:
        per = {"v0": [], "stage1": []}
        heights = {"v0": [], "stage1": []}
        for k in range(n_scenes):
            scene = make_hard_scene(size=size, pulses_per_m2=d, seed=2000 + k)
            v0 = run_v0(scene.points, cfg)
            s1 = run_stage1(scene.points, cfg, model)
            for name, o in (("v0", v0), ("stage1", s1)):
                sc = _score(o, scene.trees)
                per[name].append(sc)
                heights[name].append(sc["recall_by_height_1.0m"])
            if k == 0:
                save_preview(s1, out / f"preview_stage1_{d:g}ppm2.png")
                save_preview(v0, out / f"preview_v0_{d:g}ppm2.png")
        res = {}
        for name in per:
            hb = {}
            for h in heights[name]:
                for cls, v in h.items():
                    a = hb.setdefault(cls, [0, 0.0])
                    a[0] += v["n"]; a[1] += v["recall"] * v["n"]
            res[name] = {"1.0m": _pool(per[name], "1.0m"), "0.5m": _pool(per[name], "0.5m"),
                         "recall_by_height_1.0m": {c: round(s / max(n, 1), 3) for c, (n, s) in hb.items()}}
        report["results"][f"{d:g} pulses/m2"] = res
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report


def format_report(report: dict) -> str:
    lines = [f"Held-out synthetic scenes: {report['scenes']} per density, threshold {report['threshold']}",
             f"{'density':>14} {'method':>7} {'recall':>7} {'precision':>9} {'F1':>6} {'F1@0.5m':>8} {'dbh err':>8}  recall by height"]
    for dens, res in report["results"].items():
        for name, r in res.items():
            a, b = r["1.0m"], r["0.5m"]
            hb = ", ".join(f"{c} {v:.2f}" for c, v in r["recall_by_height_1.0m"].items())
            dbh = "-" if a["dbh_mae_m"] is None else f"{a['dbh_mae_m'] * 100:.1f} cm"
            lines.append(f"{dens:>14} {name:>7} {a['recall']:7.2f} {a['precision']:9.2f} {a['f1']:6.3f} {b['f1']:8.3f} {dbh:>8}  {hb}")
    return "\n".join(lines)

"""Generate TLS-like trees with a trained ForestGen3D checkpoint, and score them.

    sky2trail fg3d sample --model runs/x/best.pt --data data/pairs --split test --out runs/x/samples
    sky2trail fg3d eval   --samples runs/x/samples
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..stage1.train import pick_device
from .diffusion import Diffusion
from .metrics import dbh, tree_height, tree_metrics
from .pairs import OFFSET, PairDataset, load_pair, list_pairs, to_world
from .train import build_model


class Fg3dModel:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.dev = pick_device(device)
        self.net = build_model(ck).to(self.dev).eval()
        self.net.load_state_dict(ck["model"])
        tc = ck["train_cfg"]
        self.n_als, self.n_tls = tc["n_als"], tc["n_tls"]
        self.diff = Diffusion(tc["steps"])

    @torch.no_grad()
    def generate(self, als: torch.Tensor, n_tls: int | None = None, sampler: str = "ddim",
                 steps: int = 50, seed: int = 0) -> torch.Tensor:
        """TLS-like clouds (B, n_tls, 3) for normalised ALS points (B, n_als, 3)."""
        als = als.to(self.dev)
        shape = (als.shape[0], n_tls or self.n_tls, 3)
        g = torch.Generator().manual_seed(seed)
        cond = self.net.encode_cond(als)
        if sampler == "ddpm":
            x = self.diff.sample_ddpm(self.net, shape, cond, self.dev, generator=g)
        else:
            x = self.diff.sample_ddim(self.net, shape, cond, self.dev, steps=steps, generator=g)
        return x.cpu()


def sample_pairs(model: Fg3dModel, files, out: str | Path, batch: int = 8, n_samples: int = 1,
                 sampler: str = "ddim", steps: int = 50, log=print) -> list[Path]:
    """Generate n_samples clouds per pair file; write <name>.npz with gen (S, N, 3) next to the pair."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ds = PairDataset(files, model.n_als, model.n_tls)
    written = []
    for s in range(0, len(ds), batch):
        idx = list(range(s, min(s + batch, len(ds))))
        als = torch.stack([ds[i][1] for i in idx])
        gens = np.stack([model.generate(als, sampler=sampler, steps=steps, seed=k).numpy()
                         for k in range(n_samples)], axis=1)
        for j, i in enumerate(idx):
            f = Path(ds.files[i])
            dest = out / f.name
            np.savez_compressed(dest, gen=gens[j], pair=str(f))
            written.append(dest)
        log(f"{len(written)}/{len(ds)} trees")
    return written


def evaluate(samples_dir: str | Path, exact_emd: bool = True, log=print) -> dict:
    """Mean CD / EMD / EPC over generated trees, plus DBH and height (m) vs the TLS and ALS.

    Metrics use the first sample per tree; the TLS reference is the stored 2048-point cloud.
    """
    rows = []
    for f in sorted(Path(samples_dir).glob("*.npz")):
        d = np.load(f)
        p = load_pair(str(d["pair"]))
        gen = d["gen"][0]
        ref = p.tls
        m = tree_metrics(gen, ref, p.als, exact_emd=exact_emd and len(gen) == len(ref))
        gw, tw, aw = (to_world(q, p.norm) for q in (gen, ref, p.als))
        z0 = p.norm[2]
        c = p.norm[:2]
        m.update(tree=f.stem, height_gen=tree_height(gw, z0), height_tls=tree_height(tw, z0),
                 height_als=tree_height(aw, z0), dbh_gen=dbh(gw, z0, centre=c, search=1.5),
                 dbh_tls=dbh(tw, z0, centre=c, search=1.5))
        rows.append(m)
    if not rows:
        raise FileNotFoundError(f"no samples in {samples_dir}")

    def mean(k):
        v = np.array([r[k] for r in rows if k in r], float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    def mae(a, b):
        v = np.array([r[a] - r[b] for r in rows], float)
        v = v[np.isfinite(v)]
        return (float(np.abs(v).mean()) if len(v) else float("nan")), int(len(v))

    summary = {"trees": len(rows), "cd": mean("cd"), "emd": mean("emd"), "epc": mean("epc")}
    summary["height_mae_gen_m"], _ = mae("height_gen", "height_tls")
    summary["height_mae_als_m"], _ = mae("height_als", "height_tls")
    summary["dbh_mae_gen_m"], summary["dbh_trees"] = mae("dbh_gen", "dbh_tls")
    Path(samples_dir, "metrics.json").write_text(json.dumps({"summary": summary, "trees": rows}, indent=1,
                                                            default=float))
    return summary


def save_preview(samples_dir: str | Path, out: str | Path, n: int = 6) -> Path:
    """Side views (x-z) of ALS, generated and TLS clouds for the first n trees."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(Path(samples_dir).glob("*.npz"))[:n]
    fig, ax = plt.subplots(len(files), 3, figsize=(7.5, 2.6 * len(files)), squeeze=False)
    for r, f in enumerate(files):
        d = np.load(f)
        p = load_pair(str(d["pair"]))
        for c, (q, title) in enumerate(((p.als, "ALS"), (d["gen"][0], "generated"), (p.tls, "TLS"))):
            ax[r, c].scatter(q[:, 0], q[:, 2], s=0.3, c="k")
            ax[r, c].set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
            if r == 0:
                ax[r, c].set_title(title)
        ax[r, 0].set_ylabel(f.stem, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return Path(out)


__all__ = ["Fg3dModel", "sample_pairs", "evaluate", "save_preview", "list_pairs", "OFFSET"]

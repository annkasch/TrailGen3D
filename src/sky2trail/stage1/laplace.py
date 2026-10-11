"""Last-layer Laplace approximation: uncertainty of stem position, diameter and height.

The last layers of the offset and size heads are linear (1x1 convolutions) on a per-pixel
feature vector phi (plus a bias term). After training, the rest of the network stays fixed and
each output k gets a Gaussian posterior over its last-layer weights w_k (Bayesian linear
regression on the features of the labelled stem pixels):

    precision_k = Phi^T Phi / noise_k^2 + prior_k * I,    cov_k = precision_k^-1

At a pixel with features phi the predicted variance is

    var_k = phi^T cov_k phi    model part: large where the features are unlike the training data
          + noise_k^2          data part: scatter of held-out residuals

Phi comes from the training windows, noise_k^2 is the mean squared residual on the validation
windows (training residuals would be too optimistic), and prior_k maximises the Laplace
marginal likelihood. The posterior mean stays the trained weights.
Outputs: offset (dv, du in pixels), size (log dbh, height / height_scale).
"""

from __future__ import annotations

import numpy as np
import torch

HEADS = ("offset", "size")
PRIORS = np.logspace(-4, 4, 41)


def _pixels(net, samples, device, batch_size: int):
    """Per head and output: features (n, d) with bias column, and residuals (n,) at labelled pixels."""
    feats = {h: [[], []] for h in HEADS}
    resid = {h: [[], []] for h in HEADS}
    for i0 in range(0, len(samples), batch_size):
        items = [samples.get(i) for i in range(i0, min(i0 + batch_size, len(samples)))]
        b = {k: torch.from_numpy(np.stack([it[k] for it in items])).to(device)
             for k in ("image", "offset", "size", "size_mask", "reg_mask")}
        out = net(b["image"], features=True)
        masks = {"offset": b["reg_mask"].unsqueeze(1).expand_as(b["offset"]), "size": b["size_mask"]}
        for h in HEADS:
            f = out[f"{h}_features"].float()
            phi = torch.cat([f, torch.ones_like(f[:, :1])], 1).permute(0, 2, 3, 1)  # B, H, W, d
            r = (out[h].float() - b[h].float())
            for k in range(2):
                m = masks[h][:, k] > 0
                feats[h][k].append(phi[m].double().cpu().numpy())
                resid[h][k].append(r[:, k][m].double().cpu().numpy())
    return ({h: [np.concatenate(v) for v in feats[h]] for h in HEADS},
            {h: [np.concatenate(v) for v in resid[h]] for h in HEADS})


def _weights(net, head: str) -> np.ndarray:
    """Trained last-layer weights (2, d): one row per output, bias last."""
    conv = getattr(net, head)[-1]
    w = conv.weight.detach().double().cpu().numpy()[:, :, 0, 0]
    return np.concatenate([w, conv.bias.detach().double().cpu().numpy()[:, None]], 1)


def _best_prior(eig: np.ndarray, w: np.ndarray) -> float:
    """Prior precision maximising the Laplace marginal likelihood (terms that depend on it)."""
    d = len(w)
    score = [0.5 * d * np.log(p) - 0.5 * p * w @ w - 0.5 * np.log(eig + p).sum() for p in PRIORS]
    return float(PRIORS[int(np.argmax(score))])


@torch.no_grad()
def fit_laplace(net, train_samples, val_samples, device, batch_size: int = 32, log=print) -> dict:
    """Posterior covariances of the offset and size heads; stored in the checkpoint as 'laplace'."""
    net.eval()
    phi_tr, _ = _pixels(net, train_samples, device, batch_size)
    phi_va, res_va = _pixels(net, val_samples, device, batch_size)
    lap = {}
    for h in HEADS:
        w = _weights(net, h)
        cov, noise, prior, cover = [], [], [], []
        for k in range(2):
            n_va = len(res_va[h][k])
            nv = float(np.mean(res_va[h][k] ** 2)) if n_va else 1.0
            A = phi_tr[h][k].T @ phi_tr[h][k] / nv
            p = _best_prior(np.linalg.eigvalsh(A), w[k])
            c = np.linalg.inv(A + p * np.eye(len(A)))
            # Calibration check: share of validation residuals within one predicted std (~0.68).
            var = np.einsum("ni,ij,nj->n", phi_va[h][k], c, phi_va[h][k]) + nv
            cover.append(round(float(np.mean(np.abs(res_va[h][k]) <= np.sqrt(var))), 3) if n_va else None)
            cov.append(c); noise.append(nv); prior.append(p)
        lap[h] = {"cov": np.stack(cov).astype(np.float32), "noise_var": np.array(noise, np.float32),
                  "prior": prior, "val_within_1std": cover, "n_train": [len(x) for x in phi_tr[h]]}
        log(f"laplace {h}: noise std {np.sqrt(noise).round(3).tolist()}, prior {prior}, "
            f"val within 1 std {cover}")
    return lap


def std_maps(out: dict, lap: dict) -> dict:
    """Predicted std per pixel, (B, 2, H, W) for 'offset' and 'size'; out from forward(features=True)."""
    res = {}
    for h in HEADS:
        f = out[f"{h}_features"].float()
        phi = torch.cat([f, torch.ones_like(f[:, :1])], 1)
        cov = torch.as_tensor(lap[h]["cov"], dtype=phi.dtype, device=phi.device)
        noise = torch.as_tensor(lap[h]["noise_var"], dtype=phi.dtype, device=phi.device)
        var = torch.einsum("bihw,kij,bjhw->bkhw", phi, cov, phi) + noise[None, :, None, None]
        res[h] = var.clamp(min=0).sqrt()
    return res

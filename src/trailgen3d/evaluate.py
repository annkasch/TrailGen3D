"""Fidelity metrics: compare generated trunks with ground truth (field scans or synthetic)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_trunks(pred_xy: np.ndarray, pred_dbh: np.ndarray, true_xy: np.ndarray,
                 true_dbh: np.ndarray, max_dist: float = 0.5) -> dict:
    """One-to-one matching of predicted and true trunks within `max_dist` meters.

    Recall = share of real trunks found; precision = share of predicted trunks that are real.
    """
    pred_xy = np.asarray(pred_xy, float).reshape(-1, 2)
    true_xy = np.asarray(true_xy, float).reshape(-1, 2)
    n_pred, n_true = len(pred_xy), len(true_xy)
    if n_pred == 0 or n_true == 0:
        return {"n_true": n_true, "n_pred": n_pred, "matched": 0, "recall": 0.0,
                "precision": 0.0, "f1": 0.0, "mean_position_error": None,
                "mean_abs_dbh_error": None}
    d = np.hypot(pred_xy[:, None, 0] - true_xy[None, :, 0], pred_xy[:, None, 1] - true_xy[None, :, 1])
    big = 1e6
    rows, cols = linear_sum_assignment(np.where(d <= max_dist, d, big))
    ok = d[rows, cols] <= max_dist
    rows, cols = rows[ok], cols[ok]
    m = len(rows)
    recall, precision = m / n_true, m / n_pred
    f1 = 2 * recall * precision / (recall + precision) if m else 0.0
    return {
        "n_true": n_true, "n_pred": n_pred, "matched": m,
        "recall": round(recall, 3), "precision": round(precision, 3), "f1": round(f1, 3),
        "mean_position_error": round(float(d[rows, cols].mean()), 3) if m else None,
        "mean_abs_dbh_error": round(float(np.abs(np.asarray(pred_dbh)[rows] - np.asarray(true_dbh)[cols]).mean()), 3) if m else None,
        "max_dist": max_dist,
    }


def recall_by_height(pred_xy, true_xy, true_height, bins=(0, 10, 20, 100), max_dist=0.5) -> dict:
    """Recall per tree-height class: small understorey trees are usually missed first."""
    out = {}
    true_xy = np.asarray(true_xy, float)
    true_height = np.asarray(true_height, float)
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (true_height >= lo) & (true_height < hi)
        if not sel.any():
            continue
        r = match_trunks(pred_xy, np.zeros(len(pred_xy)), true_xy[sel], np.zeros(sel.sum()), max_dist)
        out[f"{lo}-{hi} m"] = {"n": int(sel.sum()), "recall": r["recall"]}
    return out

"""Rotations and flips of a window, applied consistently to image, stems and masks.

Stems are transformed as coordinates (not as offset maps), and targets are built after
augmentation, so the sub-pixel offsets stay correct by construction.
"""

from __future__ import annotations

import numpy as np


def _rot90_uv(uv: np.ndarray, n: int) -> np.ndarray:
    # np.rot90(a, 1, axes=(rows, cols)): out[i, j] = a[j, n-1-i]  ->  (v, u) -> (n - u, v)
    out = uv.copy()
    out[:, 0], out[:, 1] = n - uv[:, 1], uv[:, 0]
    return out


def transform(image: np.ndarray, stems: np.ndarray, masks: list[np.ndarray], k: int, flip: bool):
    """Rotate k quarter turns counter-clockwise, then optionally mirror left-right.

    image (C, n, n); stems (N, >=2) with columns v, u first; masks: list of (n, n).
    """
    n = image.shape[-1]
    assert image.shape[-2] == n, "windows must be square"
    stems = np.array(stems, float).reshape(-1, 4)  # always a copy
    uv = stems[:, :2]
    for _ in range(k % 4):
        uv = _rot90_uv(uv, n)
    image = np.rot90(image, k, axes=(1, 2))
    masks = [np.rot90(m, k) for m in masks]
    if flip:
        image = image[:, :, ::-1]
        masks = [m[:, ::-1] for m in masks]
        uv = uv.copy()
        uv[:, 1] = n - uv[:, 1]
    stems[:, :2] = uv
    return np.ascontiguousarray(image), stems, [np.ascontiguousarray(m) for m in masks]


def random_transform(rng: np.random.Generator, image, stems, masks):
    return transform(image, stems, masks, int(rng.integers(4)), bool(rng.integers(2)))

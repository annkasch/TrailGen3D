"""ForestGen3D pairs: normalisation, file round trip, synthetic trees, and the dataset."""

import json

import numpy as np
import pytest

from sky2trail.fg3d import pairs, synthetic


def test_norm_round_trip_and_unit_box():
    norm = pairs.box_norm(10, 20, 14, 23, 100, 125)
    assert norm[3] == 25
    p = np.array([[10, 20, 100], [14, 23, 125], [12, 21.5, 110]])
    u = pairs.to_unit(p, norm)
    assert u.min() >= 0 and u.max() <= 1
    np.testing.assert_allclose(u[2, :2], [0.5, 0.5])
    np.testing.assert_allclose(pairs.to_world(u, norm), p, atol=1e-4)


def test_fps_numpy():
    xyz = np.random.default_rng(0).random((500, 3))
    idx = pairs.fps_numpy(xyz, 64)
    assert len(set(idx)) == 64
    assert len(pairs.fps_numpy(xyz[:10], 64)) == 10


def test_synthetic_pair_looks_like_als_and_tls():
    p, info = synthetic.make_pair(3)
    assert p.tls.shape == (2048, 3)
    assert 0 <= p.tls.min() and p.tls.max() <= 1.0001
    assert len(p.als) > 50
    # ALS sees the crown top and the ground, TLS also the stem
    stem_band = lambda q: np.mean((q[:, 2] * p.norm[3] > 0.5) & (q[:, 2] * p.norm[3] < 0.3 * info["height"]))
    assert stem_band(p.tls) > 5 * stem_band(p.als)


def test_write_pairs_split_and_round_trip(tmp_path):
    paths = synthetic.write_pairs(tmp_path, 10, log=None)
    split = json.loads((tmp_path / "split.json").read_text())
    assert sum(len(v) for v in split.values()) == 10 and len(split["train"]) == 8
    q = pairs.load_pair(paths[0])
    assert q.tls.shape == (2048, 3) and q.plot == "synth"
    assert len(pairs.list_pairs(tmp_path, "train")) == 8


def test_dataset(tmp_path):
    torch = pytest.importorskip("torch")
    synthetic.write_pairs(tmp_path, 4, log=None)
    files = pairs.list_pairs(tmp_path)
    ds = pairs.PairDataset(files, n_als=256, n_tls=1024)
    tls, als = ds[0]
    assert tls.shape == (1024, 3) and als.shape == (256, 3) and tls.dtype == torch.float32
    assert torch.equal(ds[0][0], tls)                       # deterministic without augmentation
    aug = pairs.PairDataset(files, n_als=256, n_tls=1024, augment=True, thin=(0.3, 1.0))
    t2, _ = aug[0]
    assert t2.shape == (1024, 3)
    # rotation about the vertical axis keeps heights
    assert abs(float(t2[:, 2].mean() - tls[:, 2].mean())) < 0.02

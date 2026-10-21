"""ForestGen3D metrics on analytic shapes."""

import numpy as np
import pytest

from sky2trail.fg3d import metrics


def _rng():
    return np.random.default_rng(0)


def test_chamfer_and_emd():
    a = _rng().random((300, 3))
    assert metrics.chamfer(a, a) == 0 and metrics.emd(a, a) == 0
    b = a + [0.1, 0, 0]
    assert metrics.emd(a, b) == pytest.approx(0.1, abs=1e-6)
    assert 0 < metrics.chamfer(a, b) <= 2 * 0.01 + 1e-9
    assert metrics.chamfer(a, b) < metrics.chamfer(a, a + [0.3, 0, 0])


def test_envelope_containment():
    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    inside = _rng().uniform(0.1, 0.9, (100, 3))
    outside = inside + 2
    assert metrics.envelope_containment(inside, cube) == 1
    assert metrics.envelope_containment(np.r_[inside, outside], cube) == 0.5


def test_dbh_on_cylinder_with_noise():
    rng = _rng()
    n = 4000
    a = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0, 3, n)
    stem = np.c_[3 + 0.2 * np.cos(a), -1 + 0.2 * np.sin(a), z] + rng.normal(0, 0.005, (n, 3))
    clutter = np.c_[rng.uniform(2, 4, 50), rng.uniform(-2, 0, 50), rng.uniform(1.2, 1.4, 50)]
    pts = np.r_[stem, clutter]
    assert metrics.dbh(pts) == pytest.approx(0.4, abs=0.03)
    assert metrics.dbh(stem[:5]) != metrics.dbh(stem[:5])          # nan with too few points
    half = stem[np.cos(a) > 0]                                    # one side only, as from one scanner
    assert metrics.dbh(half) == pytest.approx(0.4, abs=0.04)


def test_tree_height():
    pts = np.c_[np.zeros(1000), np.zeros(1000), np.linspace(100, 120, 1000)]
    assert metrics.tree_height(pts, ground=100) == pytest.approx(20, abs=0.2)


def test_torch_metrics_match_numpy():
    torch = pytest.importorskip("torch")
    rng = _rng()
    a, b = rng.random((2, 256, 3)), rng.random((2, 256, 3))
    cd = metrics.chamfer_torch(torch.tensor(a), torch.tensor(b)).numpy()
    np.testing.assert_allclose(cd, [metrics.chamfer(a[i], b[i]) for i in range(2)], rtol=1e-4)
    e = metrics.emd_sinkhorn(torch.tensor(a), torch.tensor(b), eps=2e-3, iters=500).numpy()
    np.testing.assert_allclose(e, [metrics.emd(a[i], b[i]) for i in range(2)], rtol=0.1)

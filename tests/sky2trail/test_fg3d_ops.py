"""ForestGen3D point ops against scipy references, on CPU and (if available) Apple MPS."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scipy.spatial import cKDTree  # noqa: E402

from sky2trail.fg3d import ops  # noqa: E402


def _pts(b=2, n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, n, 3, generator=g)


def test_square_distance_and_knn_match_kdtree():
    p = _pts()
    q = p[:, :50]
    idx, dist = ops.knn(q, p, 8)
    for b in range(2):
        d_ref, i_ref = cKDTree(p[b].numpy()).query(q[b].numpy(), k=8)
        np.testing.assert_allclose(dist[b].numpy(), d_ref ** 2, atol=1e-6)
        assert (idx[b].numpy() == i_ref).mean() > 0.99          # ties / rounding may swap a few


def test_farthest_point_sample_spreads_points():
    p = _pts(n=500)
    idx = ops.farthest_point_sample(p, 32)
    assert idx.shape == (2, 32)
    for b in range(2):
        assert len(set(idx[b].tolist())) == 32
        s = p[b, idx[b]].numpy()
        fps_gap = cKDTree(s).query(s, k=2)[0][:, 1].min()
        rnd = p[b, :32].numpy()
        rnd_gap = cKDTree(rnd).query(rnd, k=2)[0][:, 1].min()
        assert fps_gap > rnd_gap
        # coverage: every point is close to some sample
        assert cKDTree(s).query(p[b].numpy())[0].max() < 0.35


def test_ball_query_stays_in_radius_and_pads_with_nearest():
    p = _pts()
    q = p[:, :20]
    idx = ops.ball_query(q, p, radius=0.1, k=16)
    d = (ops.gather(p, idx) - q.unsqueeze(2)).norm(dim=-1)
    assert (d <= 0.1 + 1e-6).all()          # query itself is nearest (distance 0), pads stay inside
    far = torch.tensor([[[5.0, 5.0, 5.0]]]).expand(2, 1, 3)
    idx_far = ops.ball_query(far, p, radius=0.1, k=4)
    assert (idx_far == idx_far[..., :1]).all()


def test_group_and_interpolate_shapes_and_values():
    p = _pts()
    f = torch.randn(2, 300, 5)
    q = p[:, :10]
    idx = ops.ball_query(q, p, 0.2, 8)
    g = ops.group(q, p, f, idx)
    assert g.shape == (2, 10, 8, 8)
    out = ops.three_nn_interpolate(q, p, f)
    torch.testing.assert_close(out, f[:, :10], atol=1e-3, rtol=1e-3)  # query on a point -> its feature


@pytest.mark.skipif(not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
                    reason="needs Apple MPS")
def test_mps_matches_cpu():
    p = _pts(n=400)
    q = p[:, :40]
    for fn in (lambda a, b: ops.knn(a, b, 8)[0], lambda a, b: ops.ball_query(a, b, 0.15, 8),
               lambda a, b: ops.farthest_point_sample(b, 16)):
        assert torch.equal(fn(q, p), fn(q.to("mps"), p.to("mps")).cpu())

"""ForestGen3D network: shapes, sensitivity to timestep and condition, gradients, devices."""

import pytest

torch = pytest.importorskip("torch")

from sky2trail.fg3d.diffusion import Diffusion  # noqa: E402
from sky2trail.fg3d.model import Fg3dNet, ModelCfg, count_parameters, tiny_cfg  # noqa: E402


def _batch(b=2, n=128, m=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, n, 3, generator=g), torch.rand(b, m, 3, generator=g)


def test_shapes_and_sensitivity():
    torch.manual_seed(0)
    net = Fg3dNet(tiny_cfg()).eval()
    x, c = _batch()
    t = torch.tensor([10, 500])
    eps = net(x, t, c)
    assert eps.shape == x.shape and torch.isfinite(eps).all()
    assert not torch.allclose(eps, net(x, torch.tensor([900, 20]), c))
    assert not torch.allclose(eps, net(x, t, torch.rand_like(c)))
    torch.testing.assert_close(eps, net(x, t, net.encode_cond(c)))   # cached condition, same output


def test_gradients_reach_all_parameters():
    torch.manual_seed(0)
    net = Fg3dNet(tiny_cfg())
    x, c = _batch()
    Diffusion().loss(net, x, c).backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not missing, missing


def test_full_size_parameter_count():
    n = count_parameters(Fg3dNet(ModelCfg()))
    print(f"Fg3dNet full size: {n / 1e6:.2f} M parameters")
    assert 0.5e6 < n < 20e6


@pytest.mark.skipif(not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
                    reason="needs Apple MPS")
def test_mps_matches_cpu():
    torch.manual_seed(0)
    net = Fg3dNet(tiny_cfg()).eval()
    x, c = _batch()
    t = torch.tensor([10, 500])
    ref = net(x, t, c)
    out = net.to("mps")(x.to("mps"), t.to("mps"), c.to("mps")).cpu()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_full_size_step():
    net = Fg3dNet().cuda()
    x, c = torch.rand(4, 2048, 3, device="cuda"), torch.rand(4, 512, 3, device="cuda")
    Diffusion().loss(net, x, c).backward()

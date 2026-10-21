"""Diffusion core: schedule, noising, and a tiny MLP that learns a 2D toy distribution."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from sky2trail.fg3d.diffusion import Diffusion, timestep_embedding  # noqa: E402


def test_schedule_and_q_sample():
    d = Diffusion()
    assert d.abar[0] > 0.999 and d.abar[-1] < 1e-4
    x0 = torch.ones(4, 10, 3)
    t = torch.tensor([0, 0, 999, 999])
    xt = d.q_sample(x0, t, torch.zeros_like(x0))
    assert torch.allclose(xt[:2], x0[:2], atol=1e-3)
    assert xt[2:].abs().max() < 1e-2


def test_timestep_embedding():
    e = timestep_embedding(torch.tensor([0, 1, 500]), 128)
    assert e.shape == (3, 128)
    assert not torch.allclose(e[1], e[2])


class _Mlp(nn.Module):
    """eps(x_t, t) for points of a 2-cluster toy distribution, applied per point."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + 32, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, 2))

    def forward(self, x, t, cond):
        emb = timestep_embedding(t, 32).unsqueeze(1).expand(-1, x.shape[1], -1)
        return self.net(torch.cat([x, emb], -1))


def _toy(n, g):
    """Points at (+-1, 0) with small spread."""
    sign = torch.randint(0, 2, (1, n, 1), generator=g).float() * 2 - 1
    return torch.cat([sign, torch.zeros_like(sign)], -1) + 0.05 * torch.randn(1, n, 2, generator=g)


def test_learns_toy_distribution():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    d = Diffusion(steps=200, beta_end=0.05)
    model = _Mlp()
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    losses = []
    for _ in range(1500):
        loss = d.loss(model, _toy(256, g))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-100:]) / 100 < 0.6 * sum(losses[:100]) / 100

    for sampler in (lambda: d.sample_ddpm(model, (1, 1000, 2), generator=g),
                    lambda: d.sample_ddim(model, (1, 1000, 2), steps=50, generator=g)):
        x = sampler()[0]
        near = ((x - torch.tensor([1.0, 0])).norm(dim=1) < 0.3) | ((x + torch.tensor([1.0, 0])).norm(dim=1) < 0.3)
        assert near.float().mean() > 0.8
        frac_right = (x[:, 0] > 0).float().mean()
        assert 0.3 < frac_right < 0.7

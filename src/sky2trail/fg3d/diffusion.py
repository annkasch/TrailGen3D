"""Gaussian diffusion (DDPM) over point coordinates, with DDPM and DDIM samplers.

ForestGen3D uses the standard DDPM setup: T = 1000 steps, linear betas 1e-4..0.02, and a network
that predicts the added noise (epsilon) under an MSE loss. The denoiser is any callable
`model(x_t, t, cond) -> eps` with x_t (B, N, D) and integer timesteps t (B,).
"""

from __future__ import annotations

from typing import Callable

import torch

Denoiser = Callable[[torch.Tensor, torch.Tensor, object], torch.Tensor]


class Diffusion:
    def __init__(self, steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.steps = steps
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)
        abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
        # float64 for the schedule, cast on use (MPS has no float64, so keep these on CPU)
        self.betas = betas
        self.alphas = alphas
        self.abar = abar
        self.abar_prev = abar_prev
        self.post_var = betas * (1 - abar_prev) / (1 - abar)

    def _at(self, v: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """v[t] as float32 on x's device, shaped to broadcast over x."""
        out = v[t.cpu()].to(device=x.device, dtype=x.dtype)
        return out.view(-1, *([1] * (x.dim() - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Noisy x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        return self._at(self.abar.sqrt(), t, x0) * x0 + self._at((1 - self.abar).sqrt(), t, x0) * noise

    def loss(self, model: Denoiser, x0: torch.Tensor, cond=None) -> torch.Tensor:
        """Epsilon-prediction MSE at uniformly drawn timesteps."""
        t = torch.randint(0, self.steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        eps = model(self.q_sample(x0, t, noise), t, cond)
        return torch.mean((eps - noise) ** 2)

    def _x0_from_eps(self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (x - self._at((1 - self.abar).sqrt(), t, x) * eps) / self._at(self.abar.sqrt(), t, x)

    @torch.no_grad()
    def sample_ddpm(self, model: Denoiser, shape: tuple, cond=None, device="cpu",
                    generator: torch.Generator | None = None) -> torch.Tensor:
        """Ancestral sampling over all T steps (as in the paper)."""
        x = torch.randn(shape, device=device, generator=generator)
        for i in reversed(range(self.steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = model(x, t, cond)
            coef = self._at(self.betas / (1 - self.abar).sqrt(), t, x)
            mean = (x - coef * eps) / self._at(self.alphas.sqrt(), t, x)
            if i > 0:
                z = torch.randn(shape, device=device, generator=generator)
                x = mean + self._at(self.post_var.sqrt(), t, x) * z
            else:
                x = mean
        return x

    @torch.no_grad()
    def sample_ddim(self, model: Denoiser, shape: tuple, cond=None, device="cpu", steps: int = 50,
                    eta: float = 0.0, generator: torch.Generator | None = None) -> torch.Tensor:
        """DDIM sampling on an evenly spaced subset of timesteps (fast evaluation)."""
        ts = torch.linspace(self.steps - 1, 0, steps).round().long().tolist()
        x = torch.randn(shape, device=device, generator=generator)
        for j, i in enumerate(ts):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = model(x, t, cond)
            x0 = self._x0_from_eps(x, t, eps)
            a = self._at(self.abar, t, x)
            if j + 1 < len(ts):
                a_next = self._at(self.abar, torch.full_like(t, ts[j + 1]), x)
            else:
                a_next = torch.ones_like(a)
            sigma = eta * ((1 - a_next) / (1 - a) * (1 - a / a_next)).clamp_min(0).sqrt()
            x = a_next.sqrt() * x0 + (1 - a_next - sigma ** 2).clamp_min(0).sqrt() * eps
            if eta > 0 and j + 1 < len(ts):
                x = x + sigma * torch.randn(shape, device=device, generator=generator)
        return x


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding (B, dim) of integer timesteps (B,)."""
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(max_period)) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float().unsqueeze(1) * freqs.to(t.device).unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

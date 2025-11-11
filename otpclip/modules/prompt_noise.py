"""Prompt-level stochastic perturbation for OTP-CLIP."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptNoise(nn.Module):
    """Implements the s^2 - t^2 stochastic offset process."""

    def __init__(
        self,
        n_cls: int,
        n_ctx: int,
        ctx_dim: int,
        granularity: str = "token",
        s_init: float = 0.05,
        t_init: float = 0.02,
        kl_lambda: float = 0.0,
        l1_lambda: float = 0.0,
        anneal_cfg: Optional[Dict[str, float]] = None,
        cap_sigma: Optional[float] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if granularity not in {"token", "class", "global"}:
            raise ValueError(f"Unsupported granularity: {granularity}")

        shape = {
            "token": (n_cls, n_ctx, 1),
            "class": (n_cls, 1, 1),
            "global": (1, 1, 1),
        }[granularity]

        self.log_s2 = nn.Parameter(torch.full(shape, torch.log(torch.tensor(s_init**2))))
        self.log_t2 = nn.Parameter(torch.full(shape, torch.log(torch.tensor(t_init**2))))

        self.kl_lambda = float(kl_lambda)
        self.l1_lambda = float(l1_lambda)
        self.anneal_cfg = anneal_cfg or {}
        self.cap_sigma = cap_sigma
        self.eps = float(eps)
        self.ctx_dim = ctx_dim

        self.register_buffer("_sigma", torch.zeros(shape))

    def forward(
        self, ctx: torch.Tensor, progress: Optional[float] = None
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        s2 = F.softplus(self.log_s2) + self.eps
        t2 = F.softplus(self.log_t2) + self.eps
        sigma2 = torch.clamp(s2 - t2, min=self.eps)

        anneal = self._anneal_factor(progress)
        sigma2 = sigma2 * anneal

        if self.cap_sigma is not None:
            sigma2 = torch.clamp(sigma2, max=self.cap_sigma ** 2)

        sigma = torch.sqrt(sigma2)
        self._sigma = sigma.detach()

        if self.training and anneal > 0:
            noise = torch.randn_like(ctx) * sigma
            ctx_tilde = ctx + noise
        else:
            ctx_tilde = ctx

        reg = ctx.new_tensor(0.0)
        if self.kl_lambda > 0.0:
            kl = 0.5 * (
                (s2 / (t2 + self.eps))
                - 1.0
                - torch.log((s2 + self.eps) / (t2 + self.eps))
            )
            reg = reg + self.kl_lambda * kl.mean()
        if self.l1_lambda > 0.0:
            reg = reg + self.l1_lambda * (
                self.log_s2.abs().mean() + self.log_t2.abs().mean()
            )

        stats = {
            "sigma": sigma.mean().item(),
            "anneal": anneal,
            "s2": s2.mean().item(),
            "t2": t2.mean().item(),
        }
        return ctx_tilde, reg, stats

    def _anneal_factor(self, progress: Optional[float]) -> float:
        if not self.anneal_cfg:
            return 1.0
        start = float(self.anneal_cfg.get("START", 0.0))
        end = float(self.anneal_cfg.get("END", start))
        if progress is None:
            return 1.0 if end <= start else 0.0
        if progress <= start:
            return 0.0
        if progress >= end:
            return 1.0
        span = max(end - start, 1e-6)
        return (progress - start) / span

    @property
    def sigma(self) -> torch.Tensor:
        return self._sigma

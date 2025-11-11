"""Prompt-level sparse over-parameterisation (SOP) noise module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...losses.regs import kl_divergence_regulariser, l1_sparsity


@dataclass
class PromptNoiseConfig:
    granularity: str = "token"
    s_init: float = 0.05
    t_init: float = 0.02
    kl_lambda: float = 1e-3
    anneal_start: int = 0
    anneal_end: int = 0
    cap_sigma: float = 0.1
    sparsity_lambda: float = 0.0


class PromptNoise(nn.Module):
    def __init__(self, cfg: PromptNoiseConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("epoch", torch.tensor(0, dtype=torch.long))
        self.log_s2: Optional[nn.Parameter] = None
        self.log_t2: Optional[nn.Parameter] = None

    def build_parameters(self, ctx: torch.Tensor) -> None:
        if self.log_s2 is not None:
            return
        shape = self._parameter_shape(ctx)
        init_s = torch.full(shape, self.cfg.s_init).log()
        init_t = torch.full(shape, self.cfg.t_init).log()
        self.log_s2 = nn.Parameter(init_s)
        self.log_t2 = nn.Parameter(init_t)

    def _parameter_shape(self, ctx: torch.Tensor) -> torch.Size:
        if self.cfg.granularity == "token":
            return ctx.shape
        if self.cfg.granularity == "class":
            return ctx.shape[:2] + (1,)
        if self.cfg.granularity == "global":
            return (1, 1, 1)
        raise ValueError(f"Unknown granularity: {self.cfg.granularity}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = torch.tensor(epoch, dtype=torch.long, device=self.epoch.device)

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.log_s2 is None:
            self.build_parameters(ctx)
        assert self.log_s2 is not None and self.log_t2 is not None

        sigma2 = F.softplus(self.log_s2) - F.softplus(self.log_t2) + 1e-6
        if self.training:
            eps = torch.randn_like(sigma2)
            std = sigma2.clamp(max=self._annealed_sigma_cap()).sqrt()
            ctx_tilde = ctx + eps * std
        else:
            ctx_tilde = ctx
        reg = torch.tensor(0.0, device=ctx.device)
        if self.cfg.kl_lambda > 0:
            reg = reg + self.cfg.kl_lambda * kl_divergence_regulariser(self.log_s2, self.log_t2)
        if self.cfg.sparsity_lambda > 0:
            reg = reg + self.cfg.sparsity_lambda * l1_sparsity(sigma2)
        self._last_sigma2 = sigma2.detach()
        return ctx_tilde, reg

    def _annealed_sigma_cap(self) -> float:
        if self.cfg.anneal_end <= self.cfg.anneal_start:
            return self.cfg.cap_sigma
        progress = (self.epoch.item() - self.cfg.anneal_start) / max(
            1, self.cfg.anneal_end - self.cfg.anneal_start
        )
        progress = float(torch.clamp(torch.tensor(progress), 0.0, 1.0))
        return self.cfg.cap_sigma * progress

    @property
    def last_sigma2(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_sigma2", None)


__all__ = ["PromptNoise", "PromptNoiseConfig"]

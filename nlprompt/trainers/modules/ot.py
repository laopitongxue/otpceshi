"""Optimal transport based sample selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from dassl.modeling.ops.optimal_transport import OptimalTransport


@dataclass
class OTConfig:
    sinkhorn_iter: int = 30
    tau_clean: float = 0.5
    temperature: float = 0.07
    epsilon: float | None = None


class PromptOptimalTransport(nn.Module):
    def __init__(self, cfg: OTConfig):
        super().__init__()
        self.cfg = cfg
        self.transport = OptimalTransport(
            max_iter=cfg.sinkhorn_iter,
            epsilon=cfg.epsilon if cfg.epsilon is not None else cfg.temperature,
        )

    def forward(self, sim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sim.ndim != 2:
            raise ValueError("Similarity matrix must be [B, C]")
        with torch.no_grad():
            cost = -sim / max(self.cfg.temperature, 1e-6)
            ot_plan = self.transport(cost)
            conf = ot_plan.max(dim=1).values
            clean_mask = conf >= self.cfg.tau_clean
            ot_loss = (cost * ot_plan).sum(dim=-1).mean()
        return clean_mask, conf, ot_loss


__all__ = ["PromptOptimalTransport", "OTConfig"]

"""Additional heads for robust OTP-CLIP losses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RobustHeadConfig:
    num_classes: int
    hidden_dim: int


class RobustClassificationHead(nn.Module):
    def __init__(self, cfg: RobustHeadConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.proj(feats)


__all__ = ["RobustClassificationHead", "RobustHeadConfig"]

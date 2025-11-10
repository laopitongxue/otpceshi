"""Mean-shift multi-view aggregation for OTP-CLIP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MTAConfig:
    V: int = 2
    h: float = 0.7
    T: int = 5
    lambda_: float = 0.1
    entropy: bool = True
    train_grad: bool = False


class MeanShiftAggregator(nn.Module):
    def __init__(self, cfg: MTAConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, img_feats_mv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if img_feats_mv.ndim != 3:
            raise ValueError("Expected [B, V, D] features")
        if not self.cfg.train_grad:
            with torch.no_grad():
                proto, weights = self._aggregate(img_feats_mv)
        else:
            proto, weights = self._aggregate(img_feats_mv)
        return proto, weights

    def _aggregate(self, img_feats_mv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, v, d = img_feats_mv.shape
        m = img_feats_mv.mean(dim=1)
        weights = torch.full((b, v), 1.0 / v, device=img_feats_mv.device, dtype=img_feats_mv.dtype)
        for _ in range(self.cfg.T):
            diff = img_feats_mv - m.unsqueeze(1)
            dist2 = diff.pow(2).sum(dim=-1)
            logits = -dist2 / (2 * (self.cfg.h ** 2) * max(self.cfg.lambda_, 1e-6))
            y = F.softmax(logits, dim=1)
            if self.cfg.entropy:
                y = y + 1e-6
                y = y / y.sum(dim=1, keepdim=True)
            m = torch.einsum("bv,bvd->bd", y, img_feats_mv)
            m = F.normalize(m, dim=-1)
            weights = y
        return m, weights


__all__ = ["MeanShiftAggregator", "MTAConfig"]

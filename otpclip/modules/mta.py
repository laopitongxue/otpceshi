"""Multi-view token aggregation utilities."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _mean_shift(feats: torch.Tensor, h: float, iters: int, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    proto = feats.mean(dim=1)
    weights = torch.full((feats.size(0), feats.size(1)), 1.0 / feats.size(1), device=feats.device, dtype=feats.dtype)

    for _ in range(max(iters, 1)):
        distances = (feats - proto.unsqueeze(1)).pow(2).sum(dim=-1)
        weights = F.softmax(-distances / (h ** 2 + 1e-6), dim=1)
        if lam > 0:
            uniform = torch.full_like(weights, 1.0 / feats.size(1))
            weights = (1 - lam) * weights + lam * uniform
        proto = (weights.unsqueeze(-1) * feats).sum(dim=1)

    weights = weights / weights.sum(dim=1, keepdim=True)
    return proto, weights


def mta_aggregate(
    feats: torch.Tensor,
    h: float = 0.7,
    iters: int = 7,
    lam: float = 0.1,
    entropy: bool = True,
    train_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate multi-view features using an entropy-regularised mean shift."""

    if feats.ndim != 3:
        raise ValueError("feats must have shape [B, V, D]")

    if train_grad:
        proto, weights = _mean_shift(feats, h, iters, lam)
    else:
        with torch.no_grad():
            proto, weights = _mean_shift(feats.detach(), h, iters, lam)
        proto = proto.detach()
        weights = weights.detach()

    if entropy:
        ent = -(weights * (weights + 1e-6).log()).sum(dim=1, keepdim=True)
        weights = weights / (ent + 1e-6)

    return proto, weights

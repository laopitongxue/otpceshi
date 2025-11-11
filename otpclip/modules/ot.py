"""Optimal transport utilities for OTP-CLIP."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def sinkhorn(logits: torch.Tensor, n_iter: int, temperature: float, eps: float = 1e-6) -> torch.Tensor:
    b, c = logits.shape
    q = torch.exp(logits / max(temperature, eps)) + eps
    q = q / q.sum(dim=1, keepdim=True)

    u = torch.full((b,), 1.0 / b, device=logits.device, dtype=logits.dtype)
    v = torch.full((c,), 1.0 / c, device=logits.device, dtype=logits.dtype)

    for _ in range(max(n_iter, 1)):
        q = q * (u / (q.sum(dim=1) + eps)).unsqueeze(1)
        q = q * (v / (q.sum(dim=0) + eps)).unsqueeze(0)

    return q / (q.sum(dim=1, keepdim=True) + eps)


def prompt_ot(
    sim: torch.Tensor,
    sinkhorn_iter: int = 30,
    tau_clean: float = 0.55,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = sim / max(temperature, 1e-6)
    probs = logits.softmax(dim=-1)
    conf, _ = probs.max(dim=-1)
    clean_mask = conf >= tau_clean

    ot_loss = sim.new_tensor(0.0)
    if sinkhorn_iter > 0:
        plan = sinkhorn(logits.detach(), sinkhorn_iter, temperature)
        plan = plan / (plan.sum(dim=-1, keepdim=True) + 1e-6)
        ot_loss = F.kl_div(plan.log(), probs, reduction="batchmean")

    return clean_mask, conf, ot_loss

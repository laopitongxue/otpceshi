"""Robust loss composition for OTP-CLIP."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.numel() == 0:
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits, target)


def _mae_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.numel() == 0:
        return logits.new_tensor(0.0)
    probs = logits.softmax(dim=-1)
    one_hot = F.one_hot(target, num_classes=logits.size(-1)).float()
    return torch.abs(probs - one_hot).sum(dim=-1).mean()


def _gce_loss(logits: torch.Tensor, target: torch.Tensor, q: float = 0.7) -> torch.Tensor:
    if target.numel() == 0:
        return logits.new_tensor(0.0)
    probs = logits.softmax(dim=-1)
    probs = probs[torch.arange(probs.size(0)), target]
    return (1 - probs.clamp(min=1e-6).pow(q)) / q


def compose(
    logits: torch.Tensor,
    target: torch.Tensor,
    clean_mask: torch.Tensor,
    ot_loss: torch.Tensor,
    weights: Dict[str, float],
    regs: Optional[Dict[str, torch.Tensor]] = None,
    robust_type: str = "MAE",
    gce_q: float = 0.7,
) -> tuple[torch.Tensor, Dict[str, float]]:
    regs = regs or {}
    clean_mask = clean_mask.bool()
    noisy_mask = ~clean_mask

    losses: Dict[str, torch.Tensor] = {}

    if weights.get("CE", 0.0) > 0:
        ce = _cross_entropy(logits[clean_mask], target[clean_mask])
        losses["loss_ce"] = ce * weights["CE"]

    if weights.get("ROBUST", 0.0) > 0:
        if robust_type.upper() == "MAE":
            robust = _mae_loss(logits[noisy_mask], target[noisy_mask])
        else:
            robust = _gce_loss(logits[noisy_mask], target[noisy_mask], q=gce_q).mean()
        losses["loss_robust"] = robust * weights["ROBUST"]

    if weights.get("OT", 0.0) > 0:
        losses["loss_ot"] = ot_loss * weights["OT"]

    if weights.get("PROMPT_REG", 0.0) > 0 and "prompt" in regs:
        losses["loss_prompt_reg"] = regs["prompt"] * weights["PROMPT_REG"]

    if weights.get("NOISE_REG", 0.0) > 0 and "noise" in regs:
        losses["loss_noise_reg"] = regs["noise"] * weights["NOISE_REG"]

    total = sum(losses.values(), logits.new_tensor(0.0))
    details = {name: value.item() for name, value in losses.items()}
    details["loss_total"] = total.item()
    return total, details

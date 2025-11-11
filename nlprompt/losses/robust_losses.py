"""Loss composer for OTP-CLIP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import torch
import torch.nn.functional as F

from .regs import l1_sparsity


@dataclass
class LossWeights:
    ce: float = 1.0
    robust: float = 0.0
    mixmatch: float = 0.0
    ot: float = 0.0
    prompt_reg: float = 0.0
    noise_reg: float = 0.0


def _mae_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    target_one_hot = F.one_hot(target, num_classes=logits.size(-1)).float()
    return torch.abs(probs - target_one_hot).mean()


def _gce_loss(logits: torch.Tensor, target: torch.Tensor, q: float = 0.7) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    target_one_hot = F.one_hot(target, num_classes=logits.size(-1)).float()
    inner = (target_one_hot * probs.pow(q)).sum(dim=-1)
    return (1 - inner) / q


def compose_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    clean_mask: torch.Tensor,
    ot_loss: torch.Tensor,
    weights: LossWeights,
    robust_type: str = "mae",
    prompt_reg: Optional[torch.Tensor] = None,
    noise_reg: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute the weighted loss and return logging scalars."""

    ce_loss = F.cross_entropy(logits[clean_mask], targets[clean_mask]) if clean_mask.any() else torch.tensor(0.0, device=logits.device)
    if robust_type == "gce":
        robust_loss = _gce_loss(logits[~clean_mask], targets[~clean_mask]) if (~clean_mask).any() else torch.tensor(0.0, device=logits.device)
    elif robust_type == "mae":
        robust_loss = _mae_loss(logits[~clean_mask], targets[~clean_mask]) if (~clean_mask).any() else torch.tensor(0.0, device=logits.device)
    else:
        robust_loss = torch.tensor(0.0, device=logits.device)

    total = (
        weights.ce * ce_loss
        + weights.robust * robust_loss
        + weights.ot * ot_loss
    )
    if prompt_reg is not None:
        total = total + weights.prompt_reg * prompt_reg
    if noise_reg is not None:
        total = total + weights.noise_reg * noise_reg

    metrics = {
        "loss/total": float(total.detach().cpu()),
        "loss/ce": float(ce_loss.detach().cpu()),
        "loss/robust": float(robust_loss.detach().cpu()),
        "loss/ot": float(ot_loss.detach().cpu()),
    }
    if prompt_reg is not None:
        metrics["reg/prompt"] = float(prompt_reg.detach().cpu())
    if noise_reg is not None:
        metrics["reg/noise"] = float(noise_reg.detach().cpu())
    return total, metrics


__all__ = ["LossWeights", "compose_losses"]

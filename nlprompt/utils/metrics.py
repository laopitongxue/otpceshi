"""Metrics for OTP-CLIP experiments."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1,)) -> Dict[int, float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=-1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res[k] = float(correct_k * (1.0 / target.size(0)))
    return res


def expected_calibration_error(logits: torch.Tensor, target: torch.Tensor, n_bins: int = 15) -> float:
    probs = logits.softmax(dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(target)
    bins = torch.linspace(0, 1, steps=n_bins + 1, device=logits.device)
    ece = torch.tensor(0.0, device=logits.device)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.any():
            bin_acc = accuracies[mask].float().mean()
            bin_conf = confidences[mask].mean()
            ece = ece + (mask.float().mean()) * torch.abs(bin_conf - bin_acc)
    return float(ece.cpu())


def brier_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    probs = logits.softmax(dim=-1)
    target_one_hot = F.one_hot(target, num_classes=logits.size(-1)).float()
    return float(((probs - target_one_hot).pow(2).sum(dim=-1)).mean().cpu())


def negative_log_likelihood(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, target).detach().cpu())


def prompt_drift(prev_ctx: torch.Tensor, curr_ctx: torch.Tensor) -> float:
    prev = F.normalize(prev_ctx.view(prev_ctx.size(0), -1), dim=-1)
    curr = F.normalize(curr_ctx.view(curr_ctx.size(0), -1), dim=-1)
    return float((1 - (prev * curr).sum(dim=-1)).mean().cpu())


__all__ = [
    "topk_accuracy",
    "expected_calibration_error",
    "brier_score",
    "negative_log_likelihood",
    "prompt_drift",
]

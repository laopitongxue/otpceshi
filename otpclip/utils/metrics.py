"""Additional metrics for OTP-CLIP."""
from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)

    ece = probs.new_tensor(0.0)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.any():
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].float().mean()
            ece = ece + (mask.float().mean() * (bin_conf - bin_acc).abs())
    return float(ece.item())


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    one_hot = F.one_hot(labels, num_classes=probs.size(-1)).float()
    return float((probs - one_hot).pow(2).sum(dim=-1).mean().item())


def negative_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, labels).item())


def prediction_drift(previous: Iterable[torch.Tensor], current: torch.Tensor) -> float:
    if not previous:
        return 0.0
    prev = torch.stack(list(previous), dim=0)
    current = current.unsqueeze(0).expand_as(prev)
    return float((prev - current).pow(2).mean().sqrt().item())


def aggregate_metrics(logits: torch.Tensor, labels: torch.Tensor, metrics: Iterable[str]) -> Dict[str, float]:
    probs = logits.softmax(dim=-1)
    results: Dict[str, float] = {}
    if "top1" in metrics:
        acc = probs.argmax(dim=-1).eq(labels).float().mean().item() * 100
        results["top1"] = acc
    if "ece" in metrics:
        results["ece"] = expected_calibration_error(probs, labels)
    if "brier" in metrics:
        results["brier"] = brier_score(probs, labels)
    if "nll" in metrics:
        results["nll"] = negative_log_likelihood(logits, labels)
    return results

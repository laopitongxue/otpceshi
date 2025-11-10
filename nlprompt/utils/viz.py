"""Lightweight visualisation helpers for OTP-CLIP."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import json
import numpy as np


def export_prompt_weights(path: str | Path, weights: np.ndarray, class_names: Sequence[str], attr_names: Sequence[str], topk: int = 5) -> None:
    """Serialize the prompt weights to JSON for offline inspection."""
    path = Path(path)
    payload = {}
    for idx, cls in enumerate(class_names):
        cls_weights = weights[idx]
        top_idx = np.argsort(cls_weights)[::-1][:topk]
        payload[cls] = [
            {"attribute": attr_names[j], "weight": float(cls_weights[j])}
            for j in top_idx
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def histogram(data: Iterable[float], bins: int = 20) -> Dict[str, Iterable[float]]:
    values, edges = np.histogram(list(data), bins=bins, density=True)
    return {"values": values.tolist(), "edges": edges.tolist()}


__all__ = ["export_prompt_weights", "histogram"]

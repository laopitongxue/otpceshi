"""Utility helpers for visualising OTP-CLIP statistics."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib is optional
    plt = None


def histogram(values: Sequence[float], bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(np.asarray(values), bins=bins, range=(0, 1))
    return hist, edges


def save_histogram(values: Iterable[float], path: str | Path, title: Optional[str] = None, bins: int = 20) -> Optional[Path]:
    if plt is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist, edges = histogram(list(values), bins=bins)
    centres = 0.5 * (edges[1:] + edges[:-1])
    plt.figure()
    plt.bar(centres, hist, width=1 / bins)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

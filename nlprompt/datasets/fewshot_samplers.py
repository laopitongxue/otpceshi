"""Few-shot sampling utilities reused across OTP-CLIP experiments."""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


@dataclass
class FewShotSplit:
    indices: List[int]
    labels: List[int]

    def to_tensor(self, device: torch.device | str | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device) if device is not None else None
        idx = torch.tensor(self.indices, dtype=torch.long, device=device)
        lab = torch.tensor(self.labels, dtype=torch.long, device=device)
        return idx, lab


def build_fewshot_indices(
    labels: Sequence[int],
    num_shots: int,
    seed: int | None = None,
    balanced: bool = True,
) -> FewShotSplit:
    """Return indices for a balanced few-shot subset.

    Parameters
    ----------
    labels:
        Sequence of dataset labels (``int``).  Typically obtained by iterating
        over the dataset and recording ``sample[1]``.
    num_shots:
        Number of instances to sample per class.
    seed:
        Optional RNG seed; when provided the sampling procedure becomes
        deterministic which is crucial for reproducible few-shot experiments.
    balanced:
        Whether to enforce the same number of samples for every class.  When set
        to ``False`` the routine simply draws ``num_shots`` random examples from
        the dataset regardless of class.
    """

    if num_shots <= 0:
        raise ValueError("num_shots must be positive")

    if seed is not None:
        random.seed(seed)

    if not balanced:
        indices = list(range(len(labels)))
        random.shuffle(indices)
        indices = indices[:num_shots]
        return FewShotSplit(indices=indices, labels=[labels[i] for i in indices])

    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_class[int(lab)].append(idx)
    for lab, idxs in by_class.items():
        if len(idxs) < num_shots:
            raise ValueError(
                f"Not enough samples for class {lab}: required {num_shots}, got {len(idxs)}"
            )
    selected_indices: List[int] = []
    selected_labels: List[int] = []
    for lab, idxs in sorted(by_class.items()):
        random.shuffle(idxs)
        keep = idxs[:num_shots]
        selected_indices.extend(keep)
        selected_labels.extend([lab] * len(keep))
    return FewShotSplit(indices=selected_indices, labels=selected_labels)


__all__ = ["FewShotSplit", "build_fewshot_indices"]

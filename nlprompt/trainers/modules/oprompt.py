"""Over-parameterized prompt module with sparsity and orthogonality regularisers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...losses.regs import cosine_orthogonality, group_lasso, l1_sparsity


@dataclass
class OPromptConfig:
    n_ctx: int = 16
    sparsity_lambda: float = 0.0
    group_lambda: float = 0.0
    orth_lambda: float = 0.0
    norm: str = "row_l1"


class OPrompt(nn.Module):
    """Implements the sparse over-parameterised prompt learner.

    Given per-class attribute embeddings ``attr_embs`` the module learns a set of
    class-conditional context tokens via a soft selection mechanism.  The
    resulting prompt contexts are later fed into :class:`PromptNoise`.
    """

    def __init__(self, cfg: OPromptConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self._reg_terms: Dict[str, torch.Tensor] = {}
        self.logits: Optional[nn.Parameter] = None

    def build_parameters(self, attr_embs: torch.Tensor) -> None:
        if self.logits is not None:
            return
        c, _, _ = attr_embs.shape
        logits = torch.zeros(c, self.cfg.n_ctx, attr_embs.size(1), device=attr_embs.device)
        nn.init.normal_(logits, mean=0.0, std=0.02)
        self.logits = nn.Parameter(logits)

    def forward(self, attr_embs: torch.Tensor) -> torch.Tensor:
        """Return learned context tokens for the given attribute embeddings."""

        if attr_embs.ndim != 3:
            raise ValueError("attr_embs must be [C, N_attr, D]")
        if self.logits is None:
            self.build_parameters(attr_embs)
        assert self.logits is not None

        weights = F.softmax(self.logits, dim=-1)
        if self.cfg.norm == "row_l1":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        ctx = torch.einsum("cnd,cad->cna", weights, attr_embs)
        self._last_ctx = ctx
        self._last_weights = weights
        return ctx

    def reg(self) -> torch.Tensor:
        if not hasattr(self, "_last_ctx"):
            return torch.tensor(0.0, device=self.logits.device if self.logits is not None else "cpu")
        regs = []
        if self.cfg.sparsity_lambda > 0:
            regs.append(self.cfg.sparsity_lambda * l1_sparsity(self._last_weights))
        if self.cfg.group_lambda > 0:
            regs.append(self.cfg.group_lambda * group_lasso(self._last_weights))
        if self.cfg.orth_lambda > 0:
            regs.append(self.cfg.orth_lambda * cosine_orthogonality(self._last_ctx))
        if regs:
            return torch.stack(regs).sum()
        return torch.tensor(0.0, device=self._last_ctx.device)

    @property
    def last_context(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_ctx", None)

    @property
    def last_weights(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_weights", None)


__all__ = ["OPrompt", "OPromptConfig"]

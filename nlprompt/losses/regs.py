"""Regularisation helpers shared by OTP-CLIP modules."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_sparsity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.abs().mean()


def group_lasso(tensor: torch.Tensor) -> torch.Tensor:
    # Treat the last dimension as the group axis.
    return torch.norm(tensor, dim=-1).mean()


def cosine_orthogonality(ctx: torch.Tensor) -> torch.Tensor:
    c, n, d = ctx.shape
    ctx_flat = ctx.view(c * n, d)
    ctx_flat = F.normalize(ctx_flat, dim=-1)
    gram = torch.matmul(ctx_flat, ctx_flat.t())
    eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
    off_diag = gram - eye
    return off_diag.pow(2).mean()


def kl_divergence_regulariser(log_s2: torch.Tensor, log_t2: torch.Tensor) -> torch.Tensor:
    return (log_s2.pow(2) + log_t2.pow(2)).mean()


__all__ = [
    "l1_sparsity",
    "group_lasso",
    "cosine_orthogonality",
    "kl_divergence_regulariser",
]

"""Attribute-driven prompt modules for OTP-CLIP."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlprompt.trainers.nlprompt import PromptLearner as CoOpPromptLearner

from .prompt_noise import PromptNoise


class OPrompt(nn.Module):
    """Learn class-conditional prompts from attribute embeddings."""

    def __init__(
        self,
        attr_embs: torch.Tensor,
        n_ctx: int,
        sparse_lambda: float = 0.0,
        group_lambda: float = 0.0,
        orth_lambda: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if attr_embs.ndim != 3:
            raise ValueError("attr_embs must have shape [C, A, D]")
        self.register_buffer("attr_embs", attr_embs)
        n_cls, n_attr, _ = attr_embs.shape
        self.logits = nn.Parameter(torch.zeros(n_cls, n_ctx, n_attr))
        nn.init.normal_(self.logits, std=0.02)

        self.n_ctx = n_ctx
        self.sparse_lambda = float(sparse_lambda)
        self.group_lambda = float(group_lambda)
        self.orth_lambda = float(orth_lambda)
        self.eps = float(eps)

        self._last_ctx: Optional[torch.Tensor] = None
        self._last_weights: Optional[torch.Tensor] = None

    def forward(self) -> torch.Tensor:
        attr_embs = self.attr_embs.to(self.logits.dtype)
        weights = F.softmax(self.logits, dim=-1)
        ctx = torch.matmul(weights, attr_embs)
        self._last_ctx = ctx
        self._last_weights = weights
        return ctx

    def reg(self) -> torch.Tensor:
        weights = self._last_weights
        ctx = self._last_ctx
        if weights is None or ctx is None:
            ctx = self.forward()
            weights = self._last_weights

        reg_loss = torch.zeros((), device=self.logits.device, dtype=self.logits.dtype)

        if self.sparse_lambda > 0.0 and weights is not None:
            entropy = -(weights * (weights + self.eps).log()).sum(dim=-1)
            reg_loss = reg_loss + self.sparse_lambda * entropy.mean()

        if self.group_lambda > 0.0 and weights is not None:
            mean_weights = weights.mean(dim=1)
            reg_loss = reg_loss + self.group_lambda * mean_weights.norm(p=1, dim=-1).mean()

        if self.orth_lambda > 0.0:
            ctx_norm = F.normalize(ctx, dim=-1)
            sim = torch.matmul(ctx_norm, ctx_norm.transpose(-1, -2))
            identity = torch.eye(self.n_ctx, device=ctx.device, dtype=ctx.dtype)
            reg_loss = reg_loss + self.orth_lambda * (sim - identity).pow(2).mean()

        return reg_loss

    @property
    def weights(self) -> Optional[torch.Tensor]:
        return self._last_weights


class AttributePromptLearner(CoOpPromptLearner):
    """Prompt learner that injects OPrompt and PromptNoise."""

    def __init__(
        self,
        cfg,
        classnames,
        clip_model,
        attr_embs: torch.Tensor,
        prompt_cfg,
        noise_cfg,
    ) -> None:
        super().__init__(cfg, classnames, clip_model)
        self.ctx.requires_grad_(False)

        self.oprompt = OPrompt(
            attr_embs=attr_embs,
            n_ctx=self.n_ctx,
            sparse_lambda=getattr(prompt_cfg, "SPARSE_LAMBDA", 0.0),
            group_lambda=getattr(prompt_cfg, "GROUP_LAMBDA", 0.0),
            orth_lambda=getattr(prompt_cfg, "ORTH_LAMBDA", 0.0),
        )

        self.prompt_noise = PromptNoise(
            n_cls=self.n_cls,
            n_ctx=self.n_ctx,
            ctx_dim=self.token_prefix.shape[-1],
            granularity=getattr(noise_cfg, "GRANULARITY", "token"),
            s_init=getattr(noise_cfg, "S_INIT", 0.05),
            t_init=getattr(noise_cfg, "T_INIT", 0.02),
            kl_lambda=getattr(noise_cfg, "KL_LAMBDA", 0.0),
            l1_lambda=getattr(noise_cfg, "L1_LAMBDA", 0.0),
            anneal_cfg=getattr(noise_cfg, "ANNEAL", None),
            cap_sigma=getattr(noise_cfg, "CAP_SIGMA", None),
        )

        self._last_noise_stats: Dict[str, float] = {}

    def forward(
        self, progress: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        ctx = self.oprompt()
        ctx_noisy, noise_reg, noise_stats = self.prompt_noise(ctx, progress=progress)
        prompts = self._assemble(ctx_noisy)
        prompt_reg = self.oprompt.reg()
        self._last_noise_stats = noise_stats
        return prompts, prompt_reg, noise_reg, noise_stats

    def _assemble(self, ctx: torch.Tensor) -> torch.Tensor:
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts_all = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1]
                suffix_i = suffix[i : i + 1]
                ctx_i = ctx[i : i + 1]
                ctx_i_half1 = ctx_i[:, :half_n_ctx]
                ctx_i_half2 = ctx_i[:, half_n_ctx:]
                class_i = suffix_i[:, :name_len]
                remainder = suffix_i[:, name_len:]
                assembled = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, remainder], dim=1)
                prompts_all.append(assembled)
            prompts = torch.cat(prompts_all, dim=0)
        elif self.class_token_position == "front":
            prompts_all = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1]
                suffix_i = suffix[i : i + 1]
                ctx_i = ctx[i : i + 1]
                class_i = suffix_i[:, :name_len]
                remainder = suffix_i[:, name_len:]
                assembled = torch.cat([prefix_i, class_i, ctx_i, remainder], dim=1)
                prompts_all.append(assembled)
            prompts = torch.cat(prompts_all, dim=0)
        else:
            raise ValueError(f"Unsupported class token position: {self.class_token_position}")

        return prompts

    @property
    def noise_stats(self) -> Dict[str, float]:
        return self._last_noise_stats

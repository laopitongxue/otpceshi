"""OTP-CLIP trainer orchestrating SOP, MTA and OT modules."""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, fields
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from dassl.engine import TRAINER_REGISTRY, TrainerX

from trainers.nlprompt import TextEncoder, load_clip_to_cpu

from ..datasets.attr_bank import build_text_templates, load_attr_bank
from ..losses.regs import l1_sparsity
from ..losses.robust_losses import LossWeights, compose_losses
from ..utils.metrics import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    prompt_drift,
    topk_accuracy,
)
from .modules.mta import MTAConfig, MeanShiftAggregator
from .modules.oprompt import OPrompt, OPromptConfig
from .modules.ot import OTConfig, PromptOptimalTransport
from .modules.prompt_noise import PromptNoise, PromptNoiseConfig

import clip


def _section_to_dict(section) -> Dict[str, object]:
    if section is None:
        return {}
    if isinstance(section, dict):
        return {k.lower(): v for k, v in section.items()}
    data = {}
    for key in dir(section):
        if key.startswith("_"):
            continue
        value = getattr(section, key)
        if callable(value):
            continue
        data[key.lower()] = value
    return data


def _load_dataclass(dc_type, section) -> object:
    data = _section_to_dict(section)
    # Support a few alias keys that appear in research configs.
    if "lambda" in data and "lambda_" in {f.name for f in fields(dc_type)}:
        data["lambda_"] = data.pop("lambda")
    if "anneal" in data:
        anneal_cfg = data.pop("anneal") or {}
        if isinstance(anneal_cfg, dict):
            data.setdefault("anneal_start", anneal_cfg.get("start_epoch", anneal_cfg.get("start")))
            data.setdefault("anneal_end", anneal_cfg.get("end_epoch", anneal_cfg.get("end")))
            data.setdefault("cap_sigma", anneal_cfg.get("cap_sigma", anneal_cfg.get("cap")))
    kwargs = {}
    for f in fields(dc_type):
        if f.name in data:
            kwargs[f.name] = data[f.name]
    return dc_type(**kwargs)


@TRAINER_REGISTRY.register()
class OTPCLIPTrainer(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.text_feats: Optional[torch.Tensor] = None
        self.prev_ctx: Optional[torch.Tensor] = None

    def build_model(self):
        cfg = self.cfg
        self.classnames = list(self.dm.dataset.classnames)
        print(f"Loading CLIP backbone {cfg.MODEL.BACKBONE.NAME}")
        self.clip_model = load_clip_to_cpu(cfg)
        self.clip_model.eval()
        self.clip_model.to(self.device)
        if hasattr(cfg.TRAINER.OTPCLIP, "PREC") and cfg.TRAINER.OTPCLIP.PREC == "fp32":
            self.clip_model.float()
        self.text_encoder = TextEncoder(self.clip_model)

        trainer_cfg = cfg.TRAINER.OTPCLIP
        self.prompt_cfg = _load_dataclass(OPromptConfig, getattr(trainer_cfg, "PROMPT", None))
        self.prompt_noise_cfg = _load_dataclass(PromptNoiseConfig, getattr(trainer_cfg, "PROMPT_NOISE", None))
        self.mta_cfg = _load_dataclass(MTAConfig, getattr(trainer_cfg, "MTA", None))
        self.ot_cfg = _load_dataclass(OTConfig, getattr(trainer_cfg, "OT", None))
        self.loss_weights = _load_dataclass(LossWeights, getattr(trainer_cfg, "LOSS_WEIGHTS", None))
        self.robust_mode = getattr(trainer_cfg, "ROBUST", "mae")
        self.temperature = getattr(trainer_cfg, "TAU", 0.01)
        self.enable_mta = getattr(trainer_cfg, "ENABLE_MTA", True)
        self.enable_ot = getattr(trainer_cfg, "ENABLE_OT", True)
        self.enable_prompt_noise = getattr(trainer_cfg, "ENABLE_PROMPT_NOISE", True)

        self.oprompt = OPrompt(self.prompt_cfg).to(self.device)
        self.prompt_noise = PromptNoise(self.prompt_noise_cfg).to(self.device)
        self.mta = MeanShiftAggregator(self.mta_cfg).to(self.device)
        self.ot = PromptOptimalTransport(self.ot_cfg).to(self.device)

        self._build_prompt_tokens()
        self.text_feats, self.noise_reg = self.build_text_features()

    def _build_prompt_tokens(self):
        base_template = getattr(self.cfg.TRAINER.OTPCLIP, "CLASS_TEMPLATE", "a photo of a {}.")
        prompts = [base_template.format(name.replace("_", " ")) for name in self.classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)
        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1:, :]
        self.tokenized_prompts = tokenized_prompts

    def build_text_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        attr_bank_path = getattr(self.cfg.TRAINER.OTPCLIP, "ATTR_BANK")
        attr_bank = load_attr_bank(attr_bank_path)
        template_cfg = _section_to_dict(getattr(self.cfg.TRAINER.OTPCLIP, "TEMPLATE", None))
        attr_vectors: List[torch.Tensor] = []
        for name in self.classnames:
            attrs = attr_bank.get(name, [])
            templates = build_text_templates(name, attrs, template_cfg)
            vectors = []
            for text in templates:
                token = clip.tokenize(text).to(self.device)
                with torch.no_grad():
                    feat = self.clip_model.encode_text(token)
                vectors.append(F.normalize(feat.squeeze(0), dim=0))
            if not vectors:
                token = clip.tokenize(f"a photo of a {name}.").to(self.device)
                with torch.no_grad():
                    feat = self.clip_model.encode_text(token)
                vectors.append(F.normalize(feat.squeeze(0), dim=0))
            attr_vectors.append(torch.stack(vectors, dim=0))
        max_attrs = max(v.size(0) for v in attr_vectors)
        padded = []
        for vec in attr_vectors:
            if vec.size(0) < max_attrs:
                pad = vec.new_zeros(max_attrs - vec.size(0), vec.size(1))
                vec = torch.cat([vec, pad], dim=0)
            padded.append(vec)
        attr_embs = torch.stack(padded, dim=0)  # [C, N_attr, D]
        ctx = self.oprompt(attr_embs)
        if self.enable_prompt_noise:
            ctx_tilde, noise_reg = self.prompt_noise(ctx)
        else:
            ctx_tilde, noise_reg = ctx, torch.tensor(0.0, device=ctx.device)
        prompts = self.build_text_inputs(ctx_tilde)
        text_feats = self.text_encoder(prompts, self.tokenized_prompts.to(self.device))
        text_feats = F.normalize(text_feats, dim=-1)
        self.prev_ctx = ctx.detach()
        return text_feats, noise_reg

    def build_text_inputs(self, ctx: torch.Tensor) -> torch.Tensor:
        prefix = self.token_prefix.to(ctx.device)
        suffix = self.token_suffix.to(ctx.device)
        if prefix.size(0) == 1:
            prefix = prefix.expand(ctx.size(0), -1, -1)
        if suffix.size(0) == 1:
            suffix = suffix.expand(ctx.size(0), -1, -1)
        return torch.cat([prefix, ctx, suffix], dim=1)

    def encode_views(self, x_views: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = x_views.shape
        views = x_views.view(b * v, c, h, w)
        with torch.no_grad():
            feats = self.clip_model.encode_image(views)
        feats = F.normalize(feats, dim=-1)
        feats = feats.view(b, v, -1)
        return feats

    def forward_batch(self, batch):
        x_views, y = batch["img"], batch["label"]
        x_views = x_views.to(self.device)
        y = y.to(self.device)
        img_feats_mv = self.encode_views(x_views)
        if self.enable_mta:
            img_proto, y_view = self.mta(img_feats_mv)
        else:
            img_proto = F.normalize(img_feats_mv.mean(dim=1), dim=-1)
            y_view = img_feats_mv.new_full((img_feats_mv.size(0), img_feats_mv.size(1)), 1.0 / img_feats_mv.size(1))
        sim = img_proto @ self.text_feats.t()
        if self.enable_ot:
            clean_mask, conf, ot_loss = self.ot(sim)
        else:
            clean_mask = torch.ones(sim.size(0), dtype=torch.bool, device=sim.device)
            conf = torch.ones(sim.size(0), device=sim.device)
            ot_loss = torch.tensor(0.0, device=sim.device)
        logits = sim / max(self.temperature, 1e-6)
        loss, loss_metrics = compose_losses(
            logits,
            y,
            clean_mask,
            ot_loss,
            self.loss_weights,
            robust_type=self.robust_mode,
            prompt_reg=self.oprompt.reg(),
            noise_reg=self.noise_reg,
        )
        metrics = self.compute_metrics(logits, y, conf, clean_mask)
        metrics.update(loss_metrics)
        return loss, metrics

    def compute_metrics(self, logits, target, conf, clean_mask):
        metrics = {}
        top1 = topk_accuracy(logits, target, topk=(1, 5))
        metrics["acc/top1"] = top1[1]
        metrics["acc/top5"] = top1.get(5, 0.0)
        metrics["calibration/ece"] = expected_calibration_error(logits, target)
        metrics["calibration/brier"] = brier_score(logits, target)
        metrics["calibration/nll"] = negative_log_likelihood(logits, target)
        metrics["ot/confidence_mean"] = float(conf.mean().cpu())
        metrics["ot/clean_rate"] = float(clean_mask.float().mean().cpu())
        if self.prev_ctx is not None and self.oprompt.last_context is not None:
            metrics["prompt/drift"] = prompt_drift(self.prev_ctx, self.oprompt.last_context.detach())
        if self.oprompt.last_weights is not None:
            metrics["prompt/sparsity"] = float(l1_sparsity(self.oprompt.last_weights).detach().cpu())
        if self.prompt_noise.last_sigma2 is not None:
            metrics["prompt/sigma_mean"] = float(self.prompt_noise.last_sigma2.mean().cpu())
        return metrics

    def before_epoch(self):
        super().before_epoch()
        if self.enable_prompt_noise and hasattr(self.prompt_noise, "set_epoch"):
            self.prompt_noise.set_epoch(self.epoch)
        self.text_feats, self.noise_reg = self.build_text_features()


__all__ = ["OTPCLIPTrainer"]

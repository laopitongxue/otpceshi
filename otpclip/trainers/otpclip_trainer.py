"""OTP-CLIP trainer integrating the Dassl/CoOp lifecycle."""
from __future__ import annotations

import os
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_pretrained_weights

from nlprompt.trainers.nlprompt import TextEncoder, load_clip_to_cpu

from otpclip.datasets.attr_bank import AttributeBank
from otpclip.losses.robust_losses import compose
from otpclip.modules.mta import mta_aggregate
from otpclip.modules.oprompt import AttributePromptLearner
from otpclip.modules.ot import prompt_ot
from otpclip.utils.metrics import aggregate_metrics, prediction_drift


class OTPCLIPModel(nn.Module):
    """Container for the CLIP components used by OTP-CLIP."""

    def __init__(self, clip_model, prompt_learner: AttributePromptLearner, tau: float) -> None:
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.prompt_learner = prompt_learner
        self.tau = float(tau)
        self.dtype = clip_model.dtype

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.image_encoder(image.type(self.dtype))
        return F.normalize(feats, dim=-1)

    def encode_text(self, progress: Optional[float] = None):
        prompts, prompt_reg, noise_reg, noise_stats = self.prompt_learner(progress)
        tokenized = self.prompt_learner.tokenized_prompts
        text_feats = self.text_encoder(prompts, tokenized)
        text_feats = F.normalize(text_feats, dim=-1)
        return text_feats, prompt_reg, noise_reg, noise_stats


@TRAINER_REGISTRY.register()
class OTPCLIPTrainer(TrainerX):
    """Trainer implementing the OTP-CLIP pipeline."""

    def __init__(self, cfg):
        self.precision = cfg.TRAINER.OTPCLIP.PREC
        self._text_cache: Optional[torch.Tensor] = None
        self._prev_eval_probs: deque[torch.Tensor] = deque(maxlen=5)
        self._batch_logs = []
        super().__init__(cfg)
        self.scaler = GradScaler() if self.precision == "amp" else None

    def check_cfg(self, cfg):
        assert cfg.TRAINER.OTPCLIP.PREC in ["fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        clip_model = load_clip_to_cpu(cfg)
        cfg.TRAINER.NLPROMPT.N_CTX = cfg.MODEL.PROMPT.N_CTX
        cfg.TRAINER.NLPROMPT.PREC = self.precision
        if self.precision == "fp32" or self.precision == "amp":
            clip_model.float()

        attr_path = cfg.MODEL.ATTR_BANK
        if attr_path and not os.path.isabs(attr_path):
            candidate = os.path.join(cfg.OUTPUT_DIR, attr_path)
            if os.path.isfile(candidate):
                attr_path = candidate
            else:
                attr_path = os.path.join(os.getcwd(), attr_path)
        bank = AttributeBank(attr_path)
        attr_texts, attr_embs = bank.build(classnames, clip_model)
        self.attr_texts = attr_texts

        prompt_learner = AttributePromptLearner(
            cfg,
            classnames,
            clip_model,
            attr_embs,
            cfg.MODEL.PROMPT,
            cfg.MODEL.PROMPT_NOISE,
        )

        self.model = OTPCLIPModel(clip_model, prompt_learner, tau=cfg.MODEL.TAU)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        for param in self.model.text_encoder.parameters():
            param.requires_grad_(False)

        if cfg.TRAIN.LR_BACKBONE > 0:
            for param in self.model.image_encoder.parameters():
                param.requires_grad_(True)
        else:
            for param in self.model.image_encoder.parameters():
                param.requires_grad_(False)

        prompt_params = list(self.model.prompt_learner.oprompt.parameters())
        noise_params = list(self.model.prompt_learner.prompt_noise.parameters())
        backbone_params = (
            list(self.model.image_encoder.parameters()) if cfg.TRAIN.LR_BACKBONE > 0 else []
        )

        tracked = {id(p) for p in prompt_params + noise_params + backbone_params}
        other_params = [
            p for p in self.model.parameters() if p.requires_grad and id(p) not in tracked
        ]

        param_groups = []
        if prompt_params:
            param_groups.append({"params": prompt_params, "lr": cfg.TRAIN.LR_PROMPT})
        if noise_params:
            param_groups.append({"params": noise_params, "lr": cfg.TRAIN.LR_PROMPT})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": cfg.TRAIN.LR_BACKBONE})
        if other_params:
            param_groups.append({"params": other_params})

        self.optim = build_optimizer(None, cfg.OPTIM, param_groups=param_groups)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("otpclip", self.model, self.optim, self.sched)

        self.loss_weights = dict(cfg.LOSS.W)
        self.robust_type = cfg.LOSS.ROBUST_TYPE
        self.gce_q = cfg.LOSS.GCE_Q if "GCE_Q" in cfg.LOSS else 0.7
        self.mta_cfg = cfg.MODEL.MTA
        self.ot_cfg = cfg.MODEL.OT
        self.test_metrics = list(cfg.TEST.METRICS)

    def before_epoch(self):
        self._batch_logs = []

    def parse_batch_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return image, label

    def parse_batch_test(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return image, label

    def _reshape_views(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 5:
            b, v = image.shape[:2]
            flat = image.view(b * v, *image.shape[2:])
            feats = self.model.encode_image(flat)
            feats = feats.view(b, v, -1)
        else:
            feats = self.model.encode_image(image)
            feats = feats.unsqueeze(1)
        return feats

    def _aggregate_views(self, feats: torch.Tensor) -> torch.Tensor:
        proto, weights = mta_aggregate(
            feats,
            h=self.mta_cfg.H,
            iters=self.mta_cfg.T,
            lam=self.mta_cfg.LAMBDA,
            entropy=self.mta_cfg.ENTROPY,
            train_grad=self.mta_cfg.TRAIN_GRAD,
        )
        return proto, weights

    def _forward_impl(self, image: torch.Tensor, label: torch.Tensor, progress: float):
        feats = self._reshape_views(image)
        proto, view_weights = self._aggregate_views(feats)
        text_feats, prompt_reg, noise_reg, noise_stats = self.model.encode_text(progress)
        sim = proto @ text_feats.t()
        logits = sim / self.model.tau
        clean_mask, conf, ot_loss = prompt_ot(
            sim,
            sinkhorn_iter=self.ot_cfg.SINKHORN,
            tau_clean=self.ot_cfg.TAU_CLEAN,
            temperature=self.model.tau,
        )
        loss, loss_dict = compose(
            logits,
            label,
            clean_mask,
            ot_loss,
            self.loss_weights,
            {"prompt": prompt_reg, "noise": noise_reg},
            robust_type=self.robust_type,
            gce_q=self.gce_q,
        )
        acc = compute_accuracy(logits, label)[0].item()
        view_entropy = -(view_weights * (view_weights + 1e-6).log()).sum(dim=-1).mean().item()
        return {
            "loss": loss,
            "logits": logits,
            "loss_dict": loss_dict,
            "accuracy": acc,
            "clean_ratio": clean_mask.float().mean().item(),
            "confidence": conf.mean().item(),
            "noise_stats": noise_stats,
            "prompt_reg": float(prompt_reg.item()),
            "noise_reg": float(noise_reg.item()),
            "view_entropy": view_entropy,
        }

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        progress = self.epoch + (self.batch_idx + 1) / max(self.num_batches, 1)

        if self.precision == "amp":
            self.optim.zero_grad()
            with autocast():
                outputs = self._forward_impl(image, label, progress)
                loss = outputs["loss"]
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            outputs = self._forward_impl(image, label, progress)
            loss = outputs["loss"]
            self.model_backward_and_update(loss, names="otpclip")

        self._batch_logs.append(outputs)
        loss_value = float(outputs["loss"].item())

        summary = {
            "loss": outputs["loss_dict"].get("loss_total", loss_value),
            "acc": outputs["accuracy"],
            "clean_ratio": outputs["clean_ratio"],
            "conf": outputs["confidence"],
            "view_entropy": outputs["view_entropy"],
        }
        return summary

    def after_epoch(self):
        if not self._batch_logs:
            return
        mean_loss = sum(d["loss_dict"].get("loss_total", 0.0) for d in self._batch_logs) / len(self._batch_logs)
        mean_acc = sum(d["accuracy"] for d in self._batch_logs) / len(self._batch_logs)
        mean_clean = sum(d["clean_ratio"] for d in self._batch_logs) / len(self._batch_logs)
        mean_conf = sum(d["confidence"] for d in self._batch_logs) / len(self._batch_logs)
        mean_entropy = sum(d["view_entropy"] for d in self._batch_logs) / len(self._batch_logs)
        mean_sigma = sum(d["noise_stats"].get("sigma", 0.0) for d in self._batch_logs) / len(self._batch_logs)

        self.write_scalar("train/loss", mean_loss, self.epoch)
        self.write_scalar("train/acc", mean_acc, self.epoch)
        self.write_scalar("train/clean_ratio", mean_clean, self.epoch)
        self.write_scalar("train/confidence", mean_conf, self.epoch)
        self.write_scalar("train/view_entropy", mean_entropy, self.epoch)
        self.write_scalar("train/sigma", mean_sigma, self.epoch)

        weights = self.model.prompt_learner.oprompt.weights
        if weights is not None:
            entropy = -(weights * (weights + 1e-6).log()).sum(dim=-1).mean().item()
            self.write_scalar("train/prompt_entropy", entropy, self.epoch)

    def model_inference(self, input):
        feats = self._reshape_views(input)
        proto, _ = self._aggregate_views(feats)
        text_feats = self._get_text_features()
        sim = proto @ text_feats.t()
        return sim / self.model.tau

    def _get_text_features(self):
        if self._text_cache is not None:
            return self._text_cache
        with torch.no_grad():
            text_feats, _, _, _ = self.model.encode_text(progress=self.max_epoch)
        self._text_cache = text_feats
        return text_feats

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self._text_cache = None
        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "val" and self.val_loader is not None:
            loader = self.val_loader
        else:
            split = "test"
            loader = self.test_loader

        logits_all = []
        labels_all = []
        for batch in loader:
            image, label = self.parse_batch_test(batch)
            logits = self.model_inference(image)
            logits_all.append(logits.cpu())
            labels_all.append(label.cpu())

        logits_cat = torch.cat(logits_all, dim=0)
        labels_cat = torch.cat(labels_all, dim=0)
        results = aggregate_metrics(logits_cat, labels_cat, self.test_metrics)
        probs = logits_cat.softmax(dim=-1)
        if "drift" in self.test_metrics:
            results["drift"] = prediction_drift(self._prev_eval_probs, probs)
        self._prev_eval_probs.append(probs)

        for key, value in results.items():
            self.write_scalar(f"{split}/{key}", value, self.epoch)

        return results.get("top1", list(results.values())[0])

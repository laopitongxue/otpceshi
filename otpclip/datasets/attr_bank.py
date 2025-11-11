"""Attribute bank construction utilities."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import torch

from nlprompt.clip import clip


class AttributeBank:
    """Loads attribute descriptors and converts them into embeddings."""

    def __init__(self, path: str | None, device: torch.device | str = "cpu") -> None:
        self.path = path
        self.device = torch.device(device)
        self.cache: Dict[str, List[str]] = {}

    def _load(self) -> Dict[str, List[str]]:
        if self.cache:
            return self.cache
        data: Dict[str, List[str]] = {}
        if self.path and os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            default_attrs = raw.get("__default__", [])
            mapping = raw.get("__attributes__", {})
            for key, attrs in mapping.items():
                data[key.lower()] = attrs
            if default_attrs:
                data.setdefault("__default__", default_attrs)
        self.cache = data
        return data

    def build(
        self, classnames: Sequence[str], clip_model
    ) -> Tuple[List[List[str]], torch.Tensor]:
        store = self._load()
        dtype = clip_model.dtype
        attr_texts: List[List[str]] = []
        attr_embs: List[torch.Tensor] = []

        with torch.no_grad():
            for name in classnames:
                key = name.lower()
                attrs = store.get(key) or store.get("__default__") or [name]
                formatted_attrs = [a.format(name=name) if "{}" in a else a for a in attrs]
                attr_texts.append(formatted_attrs)
                embeddings: List[torch.Tensor] = []
                for attr in attrs:
                    attr_text = attr.format(name=name) if "{}" in attr else attr
                    tokens = clip.tokenize(attr_text)
                    token_embs = clip_model.token_embedding(tokens).type(dtype)
                    valid_len = (tokens != 0).sum(dim=-1) - 2
                    valid_len = valid_len.clamp(min=1)
                    attr_vec = token_embs[0, 1 : 1 + valid_len, :].mean(dim=0)
                    embeddings.append(attr_vec)
                attr_stack = torch.stack(embeddings, dim=0)
                attr_embs.append(attr_stack)

        max_attrs = max(e.size(0) for e in attr_embs)
        padded_embs = []
        for emb in attr_embs:
            if emb.size(0) < max_attrs:
                pad = emb.new_zeros(max_attrs - emb.size(0), emb.size(1))
                emb = torch.cat([emb, pad], dim=0)
            padded_embs.append(emb)
        attr_tensor = torch.stack(padded_embs, dim=0).to(self.device)
        return attr_texts, attr_tensor

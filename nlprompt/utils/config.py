"""Configuration utilities for OTP-CLIP."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    data: Dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        return cls(data=payload)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


__all__ = ["Config"]

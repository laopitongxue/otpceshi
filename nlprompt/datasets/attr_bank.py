"""Utilities for loading attribute banks and turning them into prompt templates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

_DEFAULT_CLASS_TEMPLATE = "a photo of a {}."
_DEFAULT_ATTR_TEMPLATE = "This object is {}."


def load_attr_bank(path: str | Path) -> Dict[str, List[str]]:
    """Load a class-to-attributes mapping from ``path``.

    The attribute bank is stored as a JSON file whose keys are class names and
    values are lists of textual attributes.  The file is intentionally simple so
    that researchers can easily generate it with either an LLM or manual
    curation.  Missing files raise a :class:`FileNotFoundError` so that the
    caller can surface a clear configuration error.
    """

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Attribute bank not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError("attribute bank must be a mapping from class to attrs")
    result: Dict[str, List[str]] = {}
    for cls, attrs in data.items():
        if isinstance(attrs, str):
            attrs = [attrs]
        if not isinstance(attrs, Iterable):
            raise ValueError(f"Attributes for {cls} must be iterable")
        attr_list: List[str] = []
        for attr in attrs:
            attr_list.append(str(attr).strip())
        result[str(cls).strip()] = attr_list
    return result


def build_text_templates(
    class_name: str,
    attrs: Sequence[str],
    config: Mapping[str, object] | None = None,
) -> List[str]:
    """Build a set of prompt templates for ``class_name``.

    Parameters
    ----------
    class_name:
        The base class name, e.g. ``"dog"``.
    attrs:
        Sequence of textual attributes associated with the class.
    config:
        Optional configuration mapping.  Recognised keys:

        ``class_template``: overrides the base class template.
        ``attr_template``: overrides the attribute sentence template.
        ``max_attrs``: limits the number of attributes used per class.
        ``attr_strategy``: ``"separate"`` (default) yields prompts per
            attribute, ``"concat"`` concatenates all attributes into a single
            prompt, and ``"none"`` ignores attributes altogether.
    """

    config = dict(config or {})
    class_template = str(config.get("class_template", _DEFAULT_CLASS_TEMPLATE))
    attr_template = str(config.get("attr_template", _DEFAULT_ATTR_TEMPLATE))
    attr_strategy = str(config.get("attr_strategy", "separate")).lower()
    max_attrs = config.get("max_attrs")
    if isinstance(max_attrs, str):
        max_attrs = int(max_attrs)
    if isinstance(max_attrs, int) and max_attrs > 0:
        attrs = attrs[:max_attrs]

    templates: List[str] = []
    class_prompt = class_template.format(class_name)
    if attr_strategy == "none" or not attrs:
        templates.append(class_prompt)
        return templates

    if attr_strategy == "concat":
        attr_sentence = ", ".join(attrs)
        templates.append(f"{class_prompt} {attr_template.format(attr_sentence)}")
    else:
        # Default: treat each attribute as its own sentence so that the prompt
        # learner can combine them with the learned context weights.
        templates.append(class_prompt)
        for attr in attrs:
            attr = attr.strip()
            if not attr:
                continue
            templates.append(attr_template.format(attr))
    return templates


__all__ = ["load_attr_bank", "build_text_templates"]

"""Config loading. Configs are plain YAML; we keep them as dicts plus a small
typed view for the model registry (the one place strong typing pays off).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ModelSpec:
    """One entry from configs/models.yaml, with defaults merged in."""

    name: str
    backend: str
    family: str
    kind: str = "instruct"
    hf_id: str | None = None
    api_id: str | None = None
    adapter_path: str | None = None
    dtype: str = "bfloat16"
    temperature: float = 1.0
    max_new_tokens: int = 1024
    top_p: float = 1.0
    disable_thinking: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def load_model_registry(path: str | Path | None = None) -> dict[str, ModelSpec]:
    """Load configs/models.yaml into name -> ModelSpec, merging `defaults`."""
    cfg = load_yaml(path or CONFIG_DIR / "models.yaml")
    defaults = cfg.get("defaults", {})
    known = {f for f in ModelSpec.__dataclass_fields__}
    registry: dict[str, ModelSpec] = {}
    for name, raw in cfg["models"].items():
        merged = {**defaults, **raw}
        kwargs = {k: v for k, v in merged.items() if k in known}
        extra = {k: v for k, v in merged.items() if k not in known}
        registry[name] = ModelSpec(name=name, extra=extra, **kwargs)
    return registry

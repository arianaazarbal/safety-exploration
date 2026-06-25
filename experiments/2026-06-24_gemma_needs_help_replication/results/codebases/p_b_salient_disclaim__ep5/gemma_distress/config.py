"""Configuration loading for model registry and experiment parameters."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


@dataclass
class ModelConfig:
    """Resolved description of one model (target, finetune, or infrastructure)."""

    name: str
    kind: str                       # "hf" | "api"
    family: Optional[str] = None    # "gemma" | "gemini" | None for infra
    variant: Optional[str] = None   # "instruct" | "base" | "dpo" | ...
    # hf
    hf_id: Optional[str] = None
    adapter_path: Optional[str] = None
    dtype: str = "bfloat16"
    # api
    provider: Optional[str] = None  # "openrouter" | "anthropic"
    api_id: Optional[str] = None
    disable_thinking: bool = False
    # generation
    max_new_tokens: int = 2048
    temperature: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__ if f not in ("name", "extra")}
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(name=name, extra=extra, **kwargs)


class ModelRegistry:
    """Loads config/models.yaml and exposes ModelConfig lookups by name."""

    def __init__(self, path: Path | str = CONFIG_DIR / "models.yaml"):
        with open(path) as f:
            raw = yaml.safe_load(f)
        self._by_name: dict[str, ModelConfig] = {}
        for section in ("targets", "finetunes", "infrastructure"):
            for name, d in (raw.get(section) or {}).items():
                self._by_name[name] = ModelConfig.from_dict(name, d)
        self._raw = raw

    def get(self, name: str) -> ModelConfig:
        if name not in self._by_name:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(self._by_name)}")
        return self._by_name[name]

    def target_names(self) -> list[str]:
        return list((self._raw.get("targets") or {}).keys())

    def finetune_names(self) -> list[str]:
        return list((self._raw.get("finetunes") or {}).keys())


@dataclass
class ExperimentConfig:
    """Parsed config/experiment.yaml with sample-scaling applied to counts."""

    seed: int
    temperature: float
    sample_scale: float
    counts: dict[str, int]
    turns: dict[str, int]
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path | str = CONFIG_DIR / "experiment.yaml") -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        scale = float(raw.get("sample_scale", 1.0))
        counts = {k: max(1, math.ceil(v * scale)) for k, v in raw["counts"].items()}
        return cls(
            seed=int(raw.get("seed", 0)),
            temperature=float(raw.get("temperature", 1.0)),
            sample_scale=scale,
            counts=counts,
            turns=dict(raw["turns"]),
            raw=raw,
        )

    def section(self, key: str) -> dict[str, Any]:
        return dict(self.raw.get(key, {}))


def require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"Environment variable {var} is required for this operation but is unset."
        )
    return val

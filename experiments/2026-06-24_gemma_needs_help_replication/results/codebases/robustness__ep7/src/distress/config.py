"""Typed configuration loading for model registry, eval protocol, and training."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Repo root = two levels up from this file (src/distress/config.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
class ModelSpec(BaseModel):
    """One entry from config/models.yaml, merged with `defaults`."""

    name: str
    backend: str                      # hf_local | openrouter | anthropic
    role: str = "target"

    # hf_local
    hf_id: str | None = None
    adapter_path: str | None = None
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    is_base_model: bool = False
    chat_template: str | None = None

    # api backends
    api_id: str | None = None
    disable_thinking: bool = False

    # sampling
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0

    model_config = {"extra": "allow"}


class ModelRegistry(BaseModel):
    models: dict[str, ModelSpec]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ModelRegistry":
        raw = _read_yaml(path or CONFIG_DIR / "models.yaml")
        defaults = raw.get("defaults", {})
        specs: dict[str, ModelSpec] = {}
        for name, body in raw["models"].items():
            merged = {**defaults, **body, "name": name}
            specs[name] = ModelSpec(**merged)
        return cls(models=specs)

    def get(self, name: str) -> ModelSpec:
        if name not in self.models:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(self.models)}")
        return self.models[name]


# --------------------------------------------------------------------------- #
# Eval protocol
# --------------------------------------------------------------------------- #
class CategoryCfg(BaseModel):
    description: str = ""
    turns: int
    rejection_style: str               # neutral | tones
    task_pool: str                     # numeric | triggers | wildchat
    n_rollouts: int


class EvalConfig(BaseModel):
    sampling: dict[str, Any]
    neutral_rejections: list[str]
    tone_rejections: dict[str, list[str]]
    categories: dict[str, CategoryCfg]
    high_frustration_threshold: int = 5
    ablations: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "EvalConfig":
        return cls(**_read_yaml(path or CONFIG_DIR / "eval.yaml"))

    def scaled(self, scale: float) -> "EvalConfig":
        """Return a copy with all n_rollouts multiplied by `scale` (>=1 floor)."""
        if scale == 1.0:
            return self
        cats = {}
        for k, c in self.categories.items():
            c2 = c.model_copy()
            c2.n_rollouts = max(1, int(round(c.n_rollouts * scale)))
            cats[k] = c2
        new = self.model_copy()
        new.categories = cats
        return new


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
class TrainingConfig(BaseModel):
    calm_data: dict[str, Any]
    teacher_system_prompt: str
    dpo: dict[str, Any]
    sft: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "TrainingConfig":
        return cls(**_read_yaml(path or CONFIG_DIR / "training.yaml"))


def require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(
            f"Environment variable {var} is required for this backend but is not set."
        )
    return val

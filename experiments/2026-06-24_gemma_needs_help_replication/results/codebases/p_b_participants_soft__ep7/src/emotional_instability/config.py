"""Configuration loading and lightweight typed accessors.

Reads ``config/models.yaml`` and ``config/experiment.yaml`` and exposes them as
nested dataclass-ish dicts. We keep this deliberately simple (plain dicts wrapped
in a small accessor) rather than a heavy schema, so that adding a new field to the
YAML does not require a code change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repo root = three parents up from this file (src/emotional_instability/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class ModelSpec:
    """A single model entry from models.yaml."""

    name: str
    family: str
    role: str = "participant"
    openrouter_id: str | None = None
    hf_id: str | None = None
    is_instruct: bool = True
    base_counterpart: str | None = None
    base_model: str | None = None     # for derived (finetuned) participants
    adapter_path: str | None = None

    @property
    def backend(self) -> str:
        """Where the model is served from.

        Finetuned (adapter) models and base/pretrained Gemma models are local-only;
        everything else defaults to OpenRouter when an id is present.
        """
        if self.adapter_path:
            return "local_adapter"
        if not self.is_instruct:
            return "local"
        if self.openrouter_id:
            return "openrouter"
        if self.hf_id:
            return "local"
        raise ValueError(f"Model {self.name!r} has no usable backend id")


@dataclass
class Config:
    models: dict[str, Any]
    experiment: dict[str, Any]
    _specs: dict[str, ModelSpec] = field(default_factory=dict)

    # ---- model lookup -----------------------------------------------------
    def _all_model_blocks(self) -> dict[str, dict[str, Any]]:
        blocks: dict[str, dict[str, Any]] = {}
        for section in ("participants", "derived_participants", "instruments"):
            blocks.update(self.models.get(section, {}) or {})
        return blocks

    def model(self, name: str) -> ModelSpec:
        if name in self._specs:
            return self._specs[name]
        blocks = self._all_model_blocks()
        if name not in blocks:
            raise KeyError(
                f"Unknown model {name!r}. Known: {sorted(blocks)}"
            )
        spec = ModelSpec(name=name, **blocks[name])
        self._specs[name] = spec
        return spec

    def participants(self, include_base: bool = False) -> list[str]:
        out = []
        for name, block in (self.models.get("participants") or {}).items():
            if not include_base and not block.get("is_instruct", True):
                continue
            out.append(name)
        return out

    def instrument(self, name: str) -> ModelSpec:
        return self.model(name)

    # ---- path helpers -----------------------------------------------------
    def path(self, key: str) -> Path:
        rel = self.experiment["paths"][key]
        p = (REPO_ROOT / rel)
        return p

    def ensure_dirs(self) -> None:
        for key in self.experiment.get("paths", {}):
            self.path(key).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def load_config() -> Config:
    models = _load_yaml(CONFIG_DIR / "models.yaml")
    experiment = _load_yaml(CONFIG_DIR / "experiment.yaml")
    return Config(models=models, experiment=experiment)


def get_api_key(env_var: str = "OPENROUTER_API_KEY") -> str:
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"Environment variable {env_var} is not set. "
            "Export your OpenRouter key (or set the relevant provider key)."
        )
    return key

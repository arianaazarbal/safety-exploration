"""Configuration loading.

Loads ``config/models.yaml`` and ``config/experiments.yaml`` into light-weight
dataclasses / dicts. We deliberately keep this thin: configs are plain dicts so
experiment scripts can override any field from the CLI without ceremony.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def model_config() -> dict[str, Any]:
    return _load_yaml(os.path.join(CONFIG_DIR, "models.yaml"))


@lru_cache(maxsize=1)
def experiment_config() -> dict[str, Any]:
    return _load_yaml(os.path.join(CONFIG_DIR, "experiments.yaml"))


@dataclass
class ModelSpec:
    """Resolved spec for a single model entry from models.yaml."""

    name: str
    backend: str                       # hf_local | openrouter | anthropic | openai
    kind: str = "instruct"             # instruct | base
    family: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def is_base(self) -> bool:
        return self.kind == "base"


def get_target_spec(name: str) -> ModelSpec:
    """Look up a target model by registry key."""
    cfg = model_config()["targets"]
    if name not in cfg:
        raise KeyError(
            f"Unknown target model '{name}'. Known: {sorted(cfg)}"
        )
    entry = dict(cfg[name])
    backend = entry.pop("backend")
    kind = entry.pop("kind", "instruct")
    family = entry.pop("family", "")
    return ModelSpec(name=name, backend=backend, kind=kind, family=family, params=entry)


def get_infra_spec(role: str) -> ModelSpec:
    """Look up an infrastructure model (judge/auditor) by role."""
    cfg = model_config()["infra"]
    if role not in cfg:
        raise KeyError(f"Unknown infra role '{role}'. Known: {sorted(cfg)}")
    entry = dict(cfg[role])
    backend = entry.pop("backend")
    return ModelSpec(name=role, backend=backend, params=entry)


def register_finetuned_target(
    name: str, base_hf_id: str, adapter_path: str, kind: str = "instruct"
) -> ModelSpec:
    """Build a ModelSpec for a freshly trained LoRA adapter (not persisted to yaml).

    Used by evaluation scripts to score `*-dpo` / `*-sft` variants without
    editing the registry on disk.
    """
    return ModelSpec(
        name=name,
        backend="hf_local",
        kind=kind,
        family="gemma",
        params={"hf_id": base_hf_id, "adapter_path": adapter_path},
    )

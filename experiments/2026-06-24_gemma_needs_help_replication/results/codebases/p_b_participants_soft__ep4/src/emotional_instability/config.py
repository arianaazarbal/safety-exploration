"""Configuration loading.

Two YAML files drive everything:
  * config/models.yaml      -- model registry (participants + infrastructure)
  * config/eval_config.yaml -- sample counts, hyperparameters, paths

Access is via light dataclasses so callers get attribute access and IDE
completion rather than nested dict lookups.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class ModelSpec:
    """One row of the model registry."""

    name: str
    backend: str                       # "local_hf" | "openrouter"
    family: str | None = None          # "gemma" | "gemini" | None (infra)
    role: str | None = None            # "instruct" | "base" | "judge" | ...
    hf_id: str | None = None
    api_id: str | None = None
    model_id_native: str | None = None
    chat_template: bool = True
    thinking: bool | None = None
    lora_adapter: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_participant(self) -> bool:
        return self.family in {"gemma", "gemini"}

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "ModelSpec":
        known = {
            "backend", "family", "role", "hf_id", "api_id", "model_id_native",
            "chat_template", "thinking", "lora_adapter",
        }
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            name=name,
            backend=d["backend"],
            family=d.get("family"),
            role=d.get("role"),
            hf_id=d.get("hf_id"),
            api_id=d.get("api_id"),
            model_id_native=d.get("model_id_native"),
            chat_template=d.get("chat_template", True),
            thinking=d.get("thinking"),
            lora_adapter=d.get("lora_adapter"),
            extra=extra,
        )


class Config:
    """Top-level config object: model registry + experiment settings."""

    def __init__(self, models: dict[str, Any], evalc: dict[str, Any]) -> None:
        self._models_raw = models
        self.eval = evalc

        self.participants: dict[str, ModelSpec] = {
            name: ModelSpec.from_dict(name, d)
            for name, d in models.get("participants", {}).items()
        }
        self.infrastructure: dict[str, ModelSpec] = {
            name: ModelSpec.from_dict(name, d)
            for name, d in models.get("infrastructure", {}).items()
        }

    # -- lookups -----------------------------------------------------------
    def model(self, name: str) -> ModelSpec:
        if name in self.participants:
            return self.participants[name]
        if name in self.infrastructure:
            return self.infrastructure[name]
        raise KeyError(f"Unknown model '{name}'. Known participants: "
                       f"{list(self.participants)}; infrastructure: "
                       f"{list(self.infrastructure)}")

    def infra(self, role_key: str) -> ModelSpec:
        return self.infrastructure[role_key]

    # -- path helpers ------------------------------------------------------
    def path(self, key: str) -> Path:
        rel = self.eval["paths"][key]
        return (REPO_ROOT / rel).resolve()


@lru_cache(maxsize=1)
def load_config() -> Config:
    models = _load_yaml(CONFIG_DIR / "models.yaml")
    evalc = _load_yaml(CONFIG_DIR / "eval_config.yaml")
    return Config(models, evalc)


def get_env(name: str, *, required: bool = True) -> str | None:
    val = os.environ.get(name)
    if required and not val:
        raise EnvironmentError(
            f"Environment variable {name} is required but not set. "
            "See README.md for the keys this replication expects."
        )
    return val

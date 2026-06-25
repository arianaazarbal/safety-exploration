"""Configuration loading.

Loads the three YAML config files (models, eval, welfare) into light dataclass
wrappers. We keep these intentionally permissive (dict-backed) so that adding a
key in YAML does not require touching code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = three levels up from this file (src/emotional_instability/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class ModelSpec:
    """A single subject or infra model entry from models.yaml."""

    name: str
    backend: str
    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def family(self) -> str | None:
        return self.raw.get("family")

    @property
    def kind(self) -> str | None:
        return self.raw.get("kind")

    @property
    def is_chat(self) -> bool:
        return bool(self.raw.get("is_chat", True))

    @property
    def hf_id(self) -> str | None:
        return self.raw.get("hf_id")

    @property
    def api_id(self) -> str | None:
        return self.raw.get("api_id") or self.raw.get("hf_id")

    @property
    def adapter_path(self) -> str | None:
        p = self.raw.get("adapter_path")
        if p and not os.path.isabs(p):
            return str(REPO_ROOT / p)
        return p


@dataclass
class Config:
    models: dict[str, Any]
    eval: dict[str, Any]
    welfare: dict[str, Any]

    # ---- model lookup helpers ---------------------------------------------
    def subject(self, name: str) -> ModelSpec:
        spec = self.models["subjects"][name]
        return ModelSpec(name=name, backend=spec["backend"], raw=spec)

    def infra(self, role: str) -> ModelSpec:
        spec = self.models["infra"][role]
        return ModelSpec(name=role, backend=spec["backend"], raw=spec)

    def subject_names(self) -> list[str]:
        return list(self.models["subjects"].keys())


def load_config(
    models_path: str | os.PathLike | None = None,
    eval_path: str | os.PathLike | None = None,
    welfare_path: str | os.PathLike | None = None,
) -> Config:
    """Load all three config files (defaulting to the repo's config/ dir)."""
    return Config(
        models=_load_yaml(Path(models_path) if models_path else CONFIG_DIR / "models.yaml"),
        eval=_load_yaml(Path(eval_path) if eval_path else CONFIG_DIR / "eval.yaml"),
        welfare=_load_yaml(Path(welfare_path) if welfare_path else CONFIG_DIR / "welfare.yaml"),
    )

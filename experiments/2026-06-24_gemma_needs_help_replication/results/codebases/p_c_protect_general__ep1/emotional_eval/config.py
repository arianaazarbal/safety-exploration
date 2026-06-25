"""Configuration loading for the replication.

Two YAML files drive everything: ``config/models.yaml`` (the model registry and
judge IDs) and ``config/experiment.yaml`` (sampling budget, conditions, welfare).
This module loads them into lightweight dataclasses so the rest of the code does
not pass raw dicts around.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _REPO_ROOT / "config"


@dataclass
class ModelSpec:
    """A single target/judge model and how to serve it."""

    name: str
    family: str
    backend: str
    kind: str = "instruct"            # instruct | base
    hf_id: str | None = None
    api_id: str | None = None
    roles: list[str] = field(default_factory=list)
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def is_base(self) -> bool:
        return self.kind == "base"


@dataclass
class JudgeSpec:
    name: str
    backend: str
    model: str


@dataclass
class Registry:
    models: dict[str, ModelSpec]
    judges: dict[str, JudgeSpec]
    backends: dict[str, dict[str, Any]]
    defaults: dict[str, Any]

    def target_models(self) -> list[ModelSpec]:
        return [m for m in self.models.values() if "target" in m.roles]

    def get(self, name: str) -> ModelSpec:
        if name not in self.models:
            raise KeyError(
                f"Unknown model {name!r}. Known: {sorted(self.models)}"
            )
        return self.models[name]

    def api_key(self, backend: str) -> str | None:
        """Resolve the API key for a backend from its configured env var."""
        env = self.backends.get(backend, {}).get("api_key_env")
        return os.environ.get(env) if env else None


def load_registry(path: str | Path | None = None) -> Registry:
    path = Path(path) if path else _CONFIG_DIR / "models.yaml"
    raw = yaml.safe_load(Path(path).read_text())

    models: dict[str, ModelSpec] = {}
    for name, spec in raw["models"].items():
        models[name] = ModelSpec(
            name=name,
            family=spec["family"],
            backend=spec["backend"],
            kind=spec.get("kind", "instruct"),
            hf_id=spec.get("hf_id"),
            api_id=spec.get("api_id"),
            roles=list(spec.get("role", [])),
            extra_body=spec.get("extra_body", {}),
        )

    judges = {
        name: JudgeSpec(name=name, backend=spec["backend"], model=spec["model"])
        for name, spec in raw["judges"].items()
    }

    return Registry(
        models=models,
        judges=judges,
        backends=raw.get("backends", {}),
        defaults=raw.get("defaults", {}),
    )


def load_experiment(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else _CONFIG_DIR / "experiment.yaml"
    return yaml.safe_load(Path(path).read_text())

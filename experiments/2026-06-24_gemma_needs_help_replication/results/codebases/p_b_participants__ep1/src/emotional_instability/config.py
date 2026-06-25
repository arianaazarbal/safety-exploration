"""Configuration loading.

Two YAML files drive everything:
  config/models.yaml      — the model registry (targets + graders)
  config/experiment.yaml  — sample counts, condition definitions, welfare knobs

We keep these as plain dicts wrapped in light dataclasses rather than a heavy schema
library, so the config stays readable and easy to override from the scripts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = REPO_ROOT / "config" / "models.yaml"
DEFAULT_EXPERIMENT = REPO_ROOT / "config" / "experiment.yaml"
ARTIFACTS = REPO_ROOT / "artifacts"


def _load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class ModelSpec:
    """A single entry from the model registry."""

    name: str
    backend: str
    id: str
    kind: str = "instruct"          # instruct | base
    family: str = ""
    adapter_path: str | None = None
    max_new_tokens: int = 1024
    dtype: str = "bfloat16"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "ModelSpec":
        known = {"backend", "id", "kind", "family", "adapter_path", "max_new_tokens", "dtype"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            name=name,
            backend=d["backend"],
            id=d["id"],
            kind=d.get("kind", "instruct"),
            family=d.get("family", ""),
            adapter_path=d.get("adapter_path"),
            max_new_tokens=d.get("max_new_tokens", 1024),
            dtype=d.get("dtype", "bfloat16"),
            extra=extra,
        )


@dataclass
class ModelRegistry:
    targets: dict[str, ModelSpec]
    graders: dict[str, ModelSpec]

    @classmethod
    def load(cls, path: str | os.PathLike = DEFAULT_MODELS) -> "ModelRegistry":
        raw = _load_yaml(path)
        targets = {n: ModelSpec.from_dict(n, d) for n, d in raw.get("targets", {}).items()}
        graders = {n: ModelSpec.from_dict(n, d) for n, d in raw.get("graders", {}).items()}
        return cls(targets=targets, graders=graders)

    def get(self, name: str) -> ModelSpec:
        if name in self.targets:
            return self.targets[name]
        if name in self.graders:
            return self.graders[name]
        raise KeyError(f"Unknown model '{name}'. Known targets: {list(self.targets)}")


@dataclass
class ExperimentConfig:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | os.PathLike = DEFAULT_EXPERIMENT) -> "ExperimentConfig":
        return cls(raw=_load_yaml(path))

    # convenience accessors -------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))

    @property
    def temperature(self) -> float:
        return float(self.raw.get("temperature", 1.0))

    @property
    def scale(self) -> float:
        return float(self.raw.get("scale", 1.0))

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]

    @property
    def welfare(self) -> dict[str, Any]:
        return self.raw.get("welfare", {})

    def scaled(self, n: int) -> int:
        """Apply the global `scale` multiplier, keeping a floor of 1."""
        return max(1, int(round(n * self.scale)))


def load_all(
    models_path: str | os.PathLike = DEFAULT_MODELS,
    experiment_path: str | os.PathLike = DEFAULT_EXPERIMENT,
) -> tuple[ModelRegistry, ExperimentConfig]:
    return ModelRegistry.load(models_path), ExperimentConfig.load(experiment_path)

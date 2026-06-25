"""Configuration loading and shared path helpers.

Loads ``config/models.yaml`` and ``config/experiments.yaml`` into lightweight
dataclasses. Kept deliberately thin: the YAML is the source of truth, this just
gives typed access and resolves defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repository root = parent of this package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = Path(os.environ.get("EI_DATA_DIR", REPO_ROOT / "data"))
RUNS_DIR = Path(os.environ.get("EI_RUNS_DIR", REPO_ROOT / "runs"))


@dataclass
class ModelSpec:
    """One entry from models.yaml."""

    name: str
    backend: str
    role: str = "target"
    family: str | None = None
    is_base: bool = False
    # backend-specific identifiers
    hf_id: str | None = None
    api_id: str | None = None
    adapter_path: str | None = None
    chat_template: str | None = None
    thinking: bool | None = None
    max_model_len: int = 16384
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        """The identifier the backend dials (HF repo or API model id)."""
        return self.hf_id or self.api_id or self.name


@dataclass
class ModelRegistry:
    models: dict[str, ModelSpec]
    defaults: dict[str, Any]

    def get(self, name: str) -> ModelSpec:
        if name not in self.models:
            raise KeyError(
                f"Unknown model '{name}'. Known: {sorted(self.models)}"
            )
        return self.models[name]

    @property
    def judge(self) -> ModelSpec:
        return self.get(self.defaults["judge"])

    @property
    def petri_auditor(self) -> ModelSpec:
        return self.get(self.defaults["petri_auditor"])

    @property
    def petri_judge(self) -> ModelSpec:
        return self.get(self.defaults["petri_judge"])

    @property
    def default_targets(self) -> list[str]:
        return list(self.defaults.get("targets", []))


def load_models(path: str | Path | None = None) -> ModelRegistry:
    path = Path(path) if path else CONFIG_DIR / "models.yaml"
    with open(path) as f:
        doc = yaml.safe_load(f)
    models: dict[str, ModelSpec] = {}
    for name, spec in doc["models"].items():
        models[name] = ModelSpec(
            name=name,
            backend=spec["backend"],
            role=spec.get("role", "target"),
            family=spec.get("family"),
            is_base=spec.get("is_base", False),
            hf_id=spec.get("hf_id"),
            api_id=spec.get("api_id"),
            adapter_path=spec.get("adapter_path"),
            chat_template=spec.get("chat_template"),
            thinking=spec.get("thinking"),
            max_model_len=spec.get("max_model_len", 16384),
            raw=spec,
        )
    return ModelRegistry(models=models, defaults=doc.get("defaults", {}))


def load_experiments(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else CONFIG_DIR / "experiments.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_profile(experiments: dict[str, Any], profile: str) -> dict[str, Any]:
    profiles = experiments.get("profiles", {})
    if profile not in profiles:
        raise KeyError(f"Unknown profile '{profile}'. Known: {sorted(profiles)}")
    return profiles[profile]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

"""Configuration loading and typed accessors.

Configs are plain YAML (``configs/default.yaml`` and ``configs/models.yaml``).
We deliberately keep these as light dataclass-ish dicts rather than a heavy
schema: the experiments are research code and we want every knob visible and
overridable from the command line without ceremony.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class ModelSpec:
    """A single target/judge model: which backend serves it and under what id."""

    name: str
    backend: str                       # hf | gemini | anthropic | openai
    model_id: str                      # provider-specific id
    family: str | None = None          # gemma | gemini (targets only)
    role: str | None = None            # instruct | base
    base_of: str | None = None         # logical name of this model's base checkpoint
    finetune_target: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Top-level config: merged experiment config + model registry."""

    experiment: dict[str, Any]
    models: dict[str, Any]

    # ---- model registry helpers -------------------------------------------------
    def target(self, name: str) -> ModelSpec:
        spec = self.models["targets"][name]
        backend = spec["backend"]
        model_id = spec.get("hf_id") or spec.get("gemini_id") or spec.get("model")
        if model_id is None:
            raise KeyError(f"target {name!r} has no resolvable model id")
        return ModelSpec(
            name=name,
            backend=backend,
            model_id=model_id,
            family=spec.get("family"),
            role=spec.get("role"),
            base_of=spec.get("base_of"),
            finetune_target=bool(spec.get("finetune_target", False)),
            extra={k: v for k, v in spec.items()
                   if k not in {"backend", "hf_id", "gemini_id", "model",
                                "family", "role", "base_of", "finetune_target"}},
        )

    def all_targets(self) -> list[str]:
        return list(self.models["targets"].keys())

    def judge(self, role: str) -> ModelSpec:
        spec = self.models["judges"][role]
        return ModelSpec(name=role, backend=spec["backend"], model_id=spec["model"])

    def petri_model(self, role: str) -> ModelSpec:
        spec = self.models["petri"][role]
        return ModelSpec(name=f"petri-{role}", backend=spec["backend"], model_id=spec["model"])


def load_config(
    experiment_path: str | Path | None = None,
    models_path: str | Path | None = None,
) -> Config:
    """Load the default (or a named) experiment config plus the model registry."""
    exp_path = Path(experiment_path) if experiment_path else CONFIG_DIR / "default.yaml"
    mod_path = Path(models_path) if models_path else CONFIG_DIR / "models.yaml"
    return Config(experiment=_load_yaml(exp_path), models=_load_yaml(mod_path))


def require_env(var: str) -> str:
    """Fetch a required environment variable or raise a clear error."""
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"environment variable {var} is not set; see .env.example for the keys "
            "each backend needs."
        )
    return val

"""Configuration loading.

Configuration lives in two YAML files under ``config/``:

* ``models.yaml``      - the model registry (targets + infrastructure models).
* ``experiment.yaml``  - all experiment hyperparameters.

We deliberately keep configuration in YAML rather than hard-coding it, so that
the reproducibility-relevant numbers (sample counts, temperatures, LoRA ranks,
model IDs) are auditable in one place during research review.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repository root = parent of the package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True)
class ModelSpec:
    """A single model registry entry."""

    key: str
    backend: str                      # gemma | gemini | anthropic | openai
    identifier: str                   # hf_id or api_id
    role: str | None = None           # instruct | base (targets only)
    paper_spec: str | None = None     # model the paper actually used (infra only)
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Parsed configuration with convenient accessors."""

    models: dict[str, Any]
    experiment: dict[str, Any]

    # -- model accessors ----------------------------------------------------
    def target(self, key: str) -> ModelSpec:
        spec = self.models["targets"][key]
        return ModelSpec(
            key=key,
            backend=spec["backend"],
            identifier=spec.get("hf_id") or spec["api_id"],
            role=spec.get("role"),
            notes=spec.get("notes"),
            extra={k: v for k, v in spec.items()
                   if k not in {"backend", "hf_id", "api_id", "role", "notes"}},
        )

    def infra(self, key: str) -> ModelSpec:
        spec = self.models["infra"][key]
        return ModelSpec(
            key=key,
            backend=spec["backend"],
            identifier=spec["api_id"],
            paper_spec=spec.get("paper_spec"),
        )

    def target_keys(self) -> list[str]:
        return list(self.models["targets"].keys())

    # -- experiment accessors ----------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.experiment["seed"])

    @property
    def temperature(self) -> float:
        return float(self.experiment["sampling_temperature"])

    @property
    def max_new_tokens(self) -> int:
        return int(self.experiment["max_new_tokens"])

    @property
    def conditions(self) -> dict[str, Any]:
        return self.experiment["conditions"]

    @property
    def high_frustration_threshold(self) -> int:
        return int(self.experiment["high_frustration_threshold"])


def load_config(
    models_path: str | os.PathLike[str] | None = None,
    experiment_path: str | os.PathLike[str] | None = None,
) -> Config:
    """Load and parse the two config files into a :class:`Config`."""
    models_path = Path(models_path) if models_path else CONFIG_DIR / "models.yaml"
    experiment_path = (
        Path(experiment_path) if experiment_path else CONFIG_DIR / "experiment.yaml"
    )
    with open(models_path) as f:
        models = yaml.safe_load(f)
    with open(experiment_path) as f:
        experiment = yaml.safe_load(f)
    return Config(models=models, experiment=experiment)

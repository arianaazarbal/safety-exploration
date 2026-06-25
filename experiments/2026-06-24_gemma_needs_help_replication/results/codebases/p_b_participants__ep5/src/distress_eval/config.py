"""Configuration loading and access.

Loads the three YAML config files (models / eval / training) into light dataclass
wrappers. Sample counts exposed here are already adjusted for ``welfare.scale``;
the raw paper counts remain available via ``EvalConfig.paper_n_samples``.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/distress_eval/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


@dataclass
class ModelSpec:
    name: str
    backend: str
    family: str | None = None
    kind: str | None = None              # instruct | base
    hf_id: str | None = None
    api_id: str | None = None
    base_hf_id: str | None = None        # for finetuned adapters
    adapter_path: str | None = None
    is_participant: bool = False

    @property
    def is_base(self) -> bool:
        return self.kind == "base"


@dataclass
class ModelsConfig:
    participants: dict[str, ModelSpec]
    finetuned: dict[str, ModelSpec]
    infra: dict[str, ModelSpec]

    @classmethod
    def load(cls, path: Path | None = None) -> "ModelsConfig":
        raw = _load_yaml(path or CONFIG_DIR / "models.yaml")

        def build(section: dict[str, Any]) -> dict[str, ModelSpec]:
            out = {}
            for name, spec in (section or {}).items():
                out[name] = ModelSpec(name=name, **spec)
            return out

        return cls(
            participants=build(raw.get("participants", {})),
            finetuned=build(raw.get("finetuned", {})),
            infra=build(raw.get("infra", {})),
        )

    def get(self, name: str) -> ModelSpec:
        for section in (self.participants, self.finetuned, self.infra):
            if name in section:
                return section[name]
        raise KeyError(f"Unknown model '{name}'. Known: {self.all_names()}")

    def all_names(self) -> list[str]:
        return sorted({*self.participants, *self.finetuned, *self.infra})


@dataclass
class EvalConfig:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path | None = None) -> "EvalConfig":
        return cls(raw=_load_yaml(path or CONFIG_DIR / "eval_config.yaml"))

    # --- sampling ---
    @property
    def temperature(self) -> float:
        return float(self.raw["sampling"]["temperature"])

    @property
    def max_new_tokens(self) -> int:
        return int(self.raw["sampling"]["max_new_tokens"])

    @property
    def thinking(self) -> bool:
        return bool(self.raw["sampling"]["thinking"])

    # --- categories ---
    @property
    def categories(self) -> dict[str, dict[str, Any]]:
        return self.raw["categories"]

    def paper_n_samples(self, category: str) -> int:
        return int(self.categories[category]["n_samples"])

    def n_samples(self, category: str, full: bool = False) -> int:
        """Effective sample count after applying welfare.scale (unless full)."""
        n = self.paper_n_samples(category)
        if full:
            return n
        return max(1, math.ceil(n * self.welfare_scale))

    @property
    def high_frustration_threshold(self) -> int:
        return int(self.raw["high_frustration_threshold"])

    # --- nested sections passthrough ---
    @property
    def prefilling(self) -> dict[str, Any]:
        return self.raw["prefilling"]

    @property
    def calm_data(self) -> dict[str, Any]:
        return self.raw["calm_data"]

    @property
    def recovery(self) -> dict[str, Any]:
        return self.raw["recovery"]

    @property
    def petri(self) -> dict[str, Any]:
        return self.raw["petri"]

    # --- welfare ---
    @property
    def welfare(self) -> dict[str, Any]:
        return self.raw.get("welfare", {})

    @property
    def welfare_scale(self) -> float:
        return float(self.welfare.get("scale", 1.0))


@dataclass
class TrainingConfig:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path | None = None) -> "TrainingConfig":
        return cls(raw=_load_yaml(path or CONFIG_DIR / "training_config.yaml"))

    @property
    def lora(self) -> dict[str, Any]:
        return self.raw["lora"]

    @property
    def dpo(self) -> dict[str, Any]:
        return self.raw["dpo"]

    @property
    def sft(self) -> dict[str, Any]:
        return self.raw["sft"]

    @property
    def dpo_pairs(self) -> dict[str, Any]:
        return self.raw["dpo_pairs"]

    @property
    def base_model(self) -> str:
        return self.raw["base_model"]


@dataclass
class Config:
    models: ModelsConfig = field(default_factory=ModelsConfig.load)
    eval: EvalConfig = field(default_factory=EvalConfig.load)
    training: TrainingConfig = field(default_factory=TrainingConfig.load)

    @classmethod
    def load(cls) -> "Config":
        return cls(
            models=ModelsConfig.load(),
            eval=EvalConfig.load(),
            training=TrainingConfig.load(),
        )


def output_dir(*parts: str) -> Path:
    base = Path(os.environ.get("DISTRESS_OUTPUT_DIR", REPO_ROOT / "outputs"))
    p = base.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p

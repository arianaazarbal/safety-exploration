"""Typed configuration loading from the YAML files in config/."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(os.environ.get("GD_CONFIG_DIR", "config"))


def _load_yaml(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ModelSpec:
    name: str
    backend: str                       # vllm | openrouter | anthropic
    kind: str = "instruct"             # instruct | base
    family: str = ""
    hf_id: str | None = None
    api_id: str | None = None
    supports_prefill: bool = False
    temperature: float | None = None
    extra_body: dict = field(default_factory=dict)
    adapter: str | None = None         # path to LoRA adapter, set at runtime


@dataclass
class ModelRegistry:
    targets: dict[str, ModelSpec]
    roles: dict[str, ModelSpec]        # judge, cross_judge, onset_labeller, ...
    sampling: dict[str, Any]

    @classmethod
    def load(cls) -> "ModelRegistry":
        raw = _load_yaml("models.yaml")
        targets = {
            name: ModelSpec(name=name, **spec) for name, spec in raw["targets"].items()
        }
        roles = {}
        for role in (
            "judge",
            "cross_judge",
            "onset_labeller",
            "paraphraser",
            "petri_auditor",
            "petri_judge",
        ):
            if role in raw:
                roles[role] = ModelSpec(name=role, **raw[role])
        return cls(targets=targets, roles=roles, sampling=raw.get("sampling", {}))

    def target(self, name: str) -> ModelSpec:
        if name not in self.targets:
            raise KeyError(f"unknown target model '{name}'; known: {list(self.targets)}")
        return self.targets[name]


def load_eval_config() -> dict[str, Any]:
    return _load_yaml("eval.yaml")


def load_training_config() -> dict[str, Any]:
    return _load_yaml("training.yaml")

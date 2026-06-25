"""Typed configuration loading.

Configs live in ``config/*.yaml``. We parse them into lightweight dataclasses so
the rest of the code gets attribute access and a single place to document each
field. Unknown keys are preserved in ``raw`` so nothing is silently dropped.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ModelSpec:
    name: str
    family: str                  # gemma | gemini
    kind: str                    # instruct | base
    backend: str                 # hf | vllm | gemini_api
    is_participant: bool = True
    hf_id: str | None = None
    api_id: str | None = None
    chat_template: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "ModelSpec":
        return cls(
            name=name,
            family=d["family"],
            kind=d.get("kind", "instruct"),
            backend=d["backend"],
            is_participant=d.get("is_participant", True),
            hf_id=d.get("hf_id"),
            api_id=d.get("api_id"),
            chat_template=d.get("chat_template"),
            raw=d,
        )


@dataclass
class JudgeSpec:
    provider: str                # anthropic | openai
    model: str
    max_tokens: int = 1024


@dataclass
class ModelsConfig:
    participants: dict[str, ModelSpec]
    judges: dict[str, JudgeSpec]
    petri: dict[str, JudgeSpec]
    prefill_helper: JudgeSpec
    defaults: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ModelsConfig":
        d = _load_yaml(path or CONFIG_DIR / "models.yaml")
        participants = {
            name: ModelSpec.from_dict(name, spec)
            for name, spec in d["participants"].items()
        }
        judges = {k: JudgeSpec(**v) for k, v in d["judges"].items()}
        petri = {k: JudgeSpec(**v) for k, v in d["petri"].items()}
        prefill_helper = JudgeSpec(**d["prefill_helper"])
        return cls(
            participants=participants,
            judges=judges,
            petri=petri,
            prefill_helper=prefill_helper,
            defaults=d.get("defaults", {}),
            raw=d,
        )

    def participant(self, name: str) -> ModelSpec:
        if name not in self.participants:
            raise KeyError(
                f"Unknown participant {name!r}. Known: {sorted(self.participants)}"
            )
        return self.participants[name]


@dataclass
class EvalConfig:
    responses_per_model: int
    sampling: dict[str, Any]
    conditions: dict[str, dict[str, Any]]
    high_frustration_threshold: int
    validation: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "EvalConfig":
        d = _load_yaml(path or CONFIG_DIR / "eval.yaml")
        return cls(
            responses_per_model=d["responses_per_model"],
            sampling=d["sampling"],
            conditions=d["conditions"],
            high_frustration_threshold=d["high_frustration_threshold"],
            validation=d["validation"],
            raw=d,
        )


def load_training_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_yaml(path or CONFIG_DIR / "training.yaml")


def env_or_raise(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"Environment variable {var} is required but not set. "
            "See README.md for required credentials."
        )
    return val

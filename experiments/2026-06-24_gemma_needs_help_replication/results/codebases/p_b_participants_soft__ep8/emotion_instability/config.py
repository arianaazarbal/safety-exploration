"""Configuration loading and the model registry.

Centralises the participant / infrastructure split and the per-preset sample
sizes.  Everything reads from ``config.yaml`` at the repo root so that the
sample sizes can be dialled between the paper-scale run and a cheap smoke run
without touching code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class ModelSpec:
    """A model the harness can talk to."""

    name: str
    backend: str  # "hf" | "openrouter" | "anthropic"
    model_id: str  # hf repo id, openrouter id, or anthropic model name
    role: str = "instruct"  # "instruct" | "base" | "judge" | "auditor"

    @property
    def is_local(self) -> bool:
        return self.backend == "hf"

    @property
    def supports_prefill(self) -> bool:
        """Whether we can force an assistant prefix (raw continuation).

        Only local HF models give us token-level control over the assistant
        prefix.  Closed APIs (Gemini) do not expose this reliably, which is why
        the Section 3 prefill experiment is Gemma-only (see DESIGN.md).
        """
        return self.backend == "hf"


@dataclass
class Config:
    raw: dict[str, Any]
    preset_name: str
    participants: dict[str, ModelSpec]
    infrastructure: dict[str, ModelSpec]
    preset: dict[str, Any]
    generation: dict[str, Any]
    train: dict[str, Any]
    paths: dict[str, Path] = field(default_factory=dict)

    # -- convenience accessors ------------------------------------------------
    def participant(self, name: str) -> ModelSpec:
        return self.participants[name]

    def infra(self, name: str) -> ModelSpec:
        return self.infrastructure[name]

    @property
    def instruct_participants(self) -> list[str]:
        return [n for n, s in self.participants.items() if s.role == "instruct"]

    def ensure_dirs(self) -> None:
        for p in self.paths.values():
            p.mkdir(parents=True, exist_ok=True)


def _spec_from_participant(name: str, d: dict[str, Any]) -> ModelSpec:
    backend = d["backend"]
    model_id = d.get("hf_id") or d.get("or_id") or d.get("model")
    if model_id is None:
        raise ValueError(f"participant {name!r} is missing a model id")
    return ModelSpec(name=name, backend=backend, model_id=model_id, role=d.get("role", "instruct"))


def _spec_from_infra(name: str, d: dict[str, Any]) -> ModelSpec:
    backend = d["backend"]
    model_id = d.get("model") or d.get("or_id") or d.get("hf_id")
    return ModelSpec(name=name, backend=backend, model_id=model_id, role="judge")


@lru_cache(maxsize=1)
def load_config(preset: str | None = None) -> Config:
    """Load and cache the run configuration.

    Preset precedence: explicit arg > ``EMO_PRESET`` env var > "paper".
    """
    preset = preset or os.environ.get("EMO_PRESET", "paper")
    with open(CONFIG_PATH) as fh:
        raw = yaml.safe_load(fh)

    if preset not in raw["presets"]:
        raise ValueError(f"unknown preset {preset!r}; choices: {list(raw['presets'])}")

    participants = {n: _spec_from_participant(n, d) for n, d in raw["participants"].items()}
    infrastructure = {n: _spec_from_infra(n, d) for n, d in raw["infrastructure"].items()}

    paths = {k: (REPO_ROOT / v).resolve() for k, v in raw["paths"].items()}

    return Config(
        raw=raw,
        preset_name=preset,
        participants=participants,
        infrastructure=infrastructure,
        preset=raw["presets"][preset],
        generation=raw["generation"],
        train=raw["train"],
        paths=paths,
    )

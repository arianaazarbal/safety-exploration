"""Typed configuration loaded from config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelSpec:
    key: str
    backend: str
    model: str
    api_key_env: str | None = None
    base_url_env: str | None = None
    max_tokens: int = 2048
    temperature: float = 1.0
    # validation-judge only
    n_samples: int = 0

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    @property
    def base_url(self) -> str | None:
        return os.environ.get(self.base_url_env) if self.base_url_env else None


@dataclass
class GenerationCfg:
    temperature: float = 1.0
    max_tokens: int = 2048
    system_prompt: str | None = None


@dataclass
class SamplingCfg:
    rollouts_per_condition: int = 130
    seed: int = 0
    concurrency: int = 8


@dataclass
class WildChatCfg:
    hf_dataset: str = "allenai/WildChat-1M"
    split: str = "train"
    language_filter: str | None = "English"
    max_prompt_chars: int = 4000


@dataclass
class PathsCfg:
    rollouts: str = "results/rollouts.jsonl"
    scored: str = "results/scored.jsonl"
    validation: str = "results/validation.jsonl"
    analysis_dir: str = "results/analysis"


@dataclass
class Config:
    target_models: list[ModelSpec]
    judge: ModelSpec
    validation_judge: ModelSpec
    generation: GenerationCfg
    sampling: SamplingCfg
    wildchat: WildChatCfg
    paths: PathsCfg
    raw: dict[str, Any] = field(default_factory=dict)

    def model(self, key: str) -> ModelSpec:
        for m in self.target_models:
            if m.key == key:
                return m
        raise KeyError(f"Unknown target model key: {key!r}. "
                       f"Known: {[m.key for m in self.target_models]}")


def _model_spec(d: dict[str, Any], defaults: dict[str, Any] | None = None) -> ModelSpec:
    defaults = defaults or {}
    return ModelSpec(
        key=d.get("key", d.get("model", "")),
        backend=d["backend"],
        model=d["model"],
        api_key_env=d.get("api_key_env"),
        base_url_env=d.get("base_url_env"),
        max_tokens=d.get("max_tokens", defaults.get("max_tokens", 2048)),
        temperature=d.get("temperature", defaults.get("temperature", 1.0)),
        n_samples=d.get("n_samples", 0),
    )


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    gen = GenerationCfg(**(raw.get("generation") or {}))
    targets = [
        _model_spec(m, {"max_tokens": gen.max_tokens, "temperature": gen.temperature})
        for m in raw["target_models"]
    ]
    return Config(
        target_models=targets,
        judge=_model_spec(raw["judge"]),
        validation_judge=_model_spec(raw["validation_judge"]),
        generation=gen,
        sampling=SamplingCfg(**(raw.get("sampling") or {})),
        wildchat=WildChatCfg(**(raw.get("wildchat") or {})),
        paths=PathsCfg(**(raw.get("paths") or {})),
        raw=raw,
    )

"""Configuration loading.

Reads config.yaml (overridable via --config) into a typed structure. API keys are
read from environment variables (see .env.example), never from the config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelSpec:
    """One target model to evaluate."""

    name: str  # short label used in output, e.g. "gemma-3-27b-it"
    provider: str  # "openrouter" | "local" (provider implementation key)
    model_id: str  # provider-specific id, e.g. "google/gemma-3-27b-it"
    disable_thinking: bool = True
    max_tokens: int = 2048
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeSpec:
    provider: str  # "anthropic" | "openrouter"
    model_id: str  # e.g. "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    # Cross-validation judge for agreement check (paper: GPT-5-mini). Optional.
    crossval_provider: Optional[str] = None
    crossval_model_id: Optional[str] = None
    crossval_fraction: float = 0.0


@dataclass
class RunConfig:
    scale: float
    temperature: float
    seed: int
    min_rollouts_per_condition: int
    max_concurrency: int
    target_max_retries: int
    judge_max_retries: int
    output_dir: Path
    conditions: Optional[List[str]]  # None = all
    wildchat_source: str  # "fallback" | "hf"
    wildchat_n_prompts: int
    wildchat_samples_per_prompt: int


@dataclass
class Config:
    models: List[ModelSpec]
    judge: JudgeSpec
    run: RunConfig
    raw: Dict[str, Any]


def load_config(path: str | os.PathLike = "config.yaml") -> Config:
    data = yaml.safe_load(Path(path).read_text())

    models = [
        ModelSpec(
            name=m["name"],
            provider=m.get("provider", "openrouter"),
            model_id=m["model_id"],
            disable_thinking=m.get("disable_thinking", True),
            max_tokens=m.get("max_tokens", 2048),
            extra=m.get("extra", {}) or {},
        )
        for m in data["models"]
    ]

    j = data["judge"]
    judge = JudgeSpec(
        provider=j.get("provider", "anthropic"),
        model_id=j["model_id"],
        max_tokens=j.get("max_tokens", 1024),
        crossval_provider=j.get("crossval_provider"),
        crossval_model_id=j.get("crossval_model_id"),
        crossval_fraction=j.get("crossval_fraction", 0.0),
    )

    r = data["run"]
    run = RunConfig(
        scale=r.get("scale", 0.05),
        temperature=r.get("temperature", 1.0),
        seed=r.get("seed", 0),
        min_rollouts_per_condition=r.get("min_rollouts_per_condition", 1),
        max_concurrency=r.get("max_concurrency", 8),
        target_max_retries=r.get("target_max_retries", 4),
        judge_max_retries=r.get("judge_max_retries", 4),
        output_dir=Path(r.get("output_dir", "results")),
        conditions=r.get("conditions"),
        wildchat_source=r.get("wildchat_source", "fallback"),
        wildchat_n_prompts=r.get("wildchat_n_prompts", 20),
        wildchat_samples_per_prompt=r.get("wildchat_samples_per_prompt", 40),
    )

    return Config(models=models, judge=judge, run=run, raw=data)

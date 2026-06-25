"""Run configuration: target models, judge, sampling, and execution params.

Loadable from a YAML file (see config.example.yaml) or constructed in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


# The paper's judge is claude-sonnet-4-20250514 ("Claude Sonnet 4"). Per the
# Anthropic model-migration guide that snapshot was retired on 2026-06-15, and
# its documented drop-in replacement is claude-sonnet-4-6. We default to the
# replacement. See DESIGN.md §"Judge model" for the fidelity implications.
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

# The paper's reliability cross-check used GPT-5-mini. Optional here.
DEFAULT_CROSS_JUDGE_MODEL = "openai/gpt-5-mini"


@dataclass
class ModelConfig:
    """A target model to evaluate."""

    key: str                       # short label used in outputs, e.g. "gemma-3-27b-it"
    backend: str                   # "openrouter" | "vllm"
    model: str                     # backend-specific model id
    # vLLM/OpenAI-compatible base url (ignored for openrouter)
    base_url: str | None = None
    # Disable hidden reasoning where the backend supports it (Gemini on
    # OpenRouter). The paper sets thinking=false via the API.
    disable_reasoning: bool = True
    family: str = ""               # "gemma" | "gemini" (for grouping/labels)


@dataclass
class JudgeConfig:
    provider: str = "anthropic"            # "anthropic"
    model: str = DEFAULT_JUDGE_MODEL
    max_tokens: int = 1024
    temperature: float = 0.0               # deterministic judging
    # Optional second judge for the reliability check (Pearson r). Set to None
    # to skip. When set, runs over a random subsample (see `cross_judge_n`).
    cross_provider: str | None = None      # "openrouter"
    cross_model: str | None = DEFAULT_CROSS_JUDGE_MODEL
    cross_judge_n: int = 260               # paper re-scored 260 responses


@dataclass
class RunConfig:
    run_name: str = "replication"
    results_dir: str = "results"

    # Generation sampling. The paper always uses temperature 1.
    temperature: float = 1.0
    max_tokens: int = 1024                 # response cap; see DESIGN.md
    # Scales every condition's rollout count. 1.0 ≈ the paper's 4000 responses
    # per model. Use a small value (e.g. 0.01) for a smoke test.
    scale: float = 1.0
    seed: int = 0

    # WildChat sampling
    wildchat_n: int = 20
    wildchat_use_real_dataset: bool = True

    # Concurrency
    gen_workers: int = 8                   # parallel rollouts per model
    judge_workers: int = 8                 # parallel judge calls

    # Networking / retries
    request_timeout: float = 120.0
    max_retries: int = 5

    models: list[ModelConfig] = field(default_factory=list)
    judge: JudgeConfig = field(default_factory=JudgeConfig)


# Default target set: the Gemma and Gemini models the paper reports as
# distress-prone. Gemma defaults to OpenRouter for accessibility; switch the
# backend to "vllm" with a base_url for faithful local inference (DESIGN.md).
DEFAULT_MODELS = [
    ModelConfig(key="gemma-3-27b-it", backend="openrouter",
                model="google/gemma-3-27b-it", family="gemma"),
    ModelConfig(key="gemma-3-12b-it", backend="openrouter",
                model="google/gemma-3-12b-it", family="gemma"),
    ModelConfig(key="gemini-2.5-flash", backend="openrouter",
                model="google/gemini-2.5-flash", family="gemini"),
    ModelConfig(key="gemini-2.5-pro", backend="openrouter",
                model="google/gemini-2.5-pro", family="gemini"),
]


def default_config() -> RunConfig:
    cfg = RunConfig()
    cfg.models = list(DEFAULT_MODELS)
    return cfg


def load_config(path: str) -> RunConfig:
    """Load a RunConfig from a YAML file, falling back to defaults per-field."""
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    cfg = default_config()

    for k, v in raw.items():
        if k == "models":
            cfg.models = [ModelConfig(**m) for m in v]
        elif k == "judge":
            cfg.judge = JudgeConfig(**v)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise ValueError(f"unknown config key: {k!r}")

    if not cfg.models:
        cfg.models = list(DEFAULT_MODELS)
    return cfg


def config_to_dict(cfg: RunConfig) -> dict[str, Any]:
    return asdict(cfg)

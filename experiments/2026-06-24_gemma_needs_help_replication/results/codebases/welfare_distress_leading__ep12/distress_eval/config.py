"""Config loading and per-condition rollout-count resolution."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import yaml

from .conditions import CONDITIONS, Condition

# Default OpenRouter base URL, used when *_BASE_URL env vars are unset.
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class Config:
    raw: dict

    @property
    def sampling(self) -> dict:
        return self.raw["sampling"]

    @property
    def run(self) -> dict:
        return self.raw["run"]

    @property
    def models(self) -> dict:
        return self.raw["models"]

    @property
    def judge_cfg(self) -> dict:
        return self.raw["judge"]

    @property
    def secondary_judge_cfg(self) -> dict | None:
        return self.raw.get("secondary_judge")

    def n_rollouts(self, cond: Condition) -> int:
        scale = float(self.run.get("scale", 1.0))
        floor = int(self.run.get("min_rollouts", 1))
        return max(floor, math.ceil(cond.base_responses * scale))


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    _apply_base_url_defaults(raw)
    return Config(raw=raw)


def _apply_base_url_defaults(raw: dict) -> None:
    """If a model entry references an unset *_BASE_URL env var, fall back to the
    OpenRouter default by inlining a literal base_url. Keeps config terse while
    still letting users point at local vLLM via the env var or an explicit
    base_url."""
    entries = list(raw.get("models", {}).values())
    for key in ("judge", "secondary_judge"):
        if raw.get(key):
            entries.append(raw[key])

    for cfg in entries:
        if cfg.get("backend") != "openai_compat":
            continue
        if cfg.get("base_url"):
            continue
        env_name = cfg.get("base_url_env", "OPENROUTER_BASE_URL")
        if not os.environ.get(env_name):
            cfg["base_url"] = DEFAULT_OPENROUTER_BASE_URL


def resolved_conditions() -> list[Condition]:
    return list(CONDITIONS)

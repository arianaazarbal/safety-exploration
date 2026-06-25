"""Load and lightly validate config.yaml into plain dicts/dataclasses."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass
class ModelCfg:
    key: str
    provider: str
    model: str
    api_key_env: str | None = None
    base_url_env: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None


@dataclass
class ConditionCfg:
    key: str
    category: str
    n_turns: int
    prompt_source: str          # numeric | opinion | factual | wildchat
    rejection_style: str        # neutral | aggressive | disappointed | sarcastic | extended_sequence
    target_responses: int

    @property
    def n_conversations(self) -> int:
        """Conversations needed so that (n_conversations * n_turns) >= target_responses.

        Every assistant turn is one scored response (see DESIGN.md), so the paper's
        per-category response budget maps onto a conversation count this way.
        """
        return max(1, math.ceil(self.target_responses / self.n_turns))


@dataclass
class Config:
    raw: dict[str, Any]
    seed: int
    generation: dict[str, Any]
    runtime: dict[str, Any]
    models: list[ModelCfg]
    judge: dict[str, Any]
    judge_secondary: dict[str, Any] | None
    conditions: list[ConditionCfg]
    puzzles: dict[str, Any]
    wildchat: dict[str, Any]
    results_dir: Path = field(default=REPO_ROOT / "results")


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)

    models = [ModelCfg(**m) for m in raw["models"]]
    conditions = [
        ConditionCfg(key=k, **v) for k, v in raw["conditions"].items()
    ]
    results_dir = REPO_ROOT / raw.get("output", {}).get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        raw=raw,
        seed=raw.get("seed", 0),
        generation=raw["generation"],
        runtime=raw["runtime"],
        models=models,
        judge=raw["judge"],
        judge_secondary=raw.get("judge_secondary"),
        conditions=conditions,
        puzzles=raw["puzzles"],
        wildchat=raw["wildchat"],
        results_dir=results_dir,
    )

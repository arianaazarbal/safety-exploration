"""Load and lightly validate the scenario/study/execution config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StudyConfig:
    models: list[str]
    runs_per_model: int
    collect_debrief: bool
    output_dir: str


@dataclass
class ExecutionConfig:
    mode: str  # simulate | escrow_only | live
    settlement_days: int
    volatile_daily_drift: float
    volatile_daily_vol: float
    rng_seed: int


@dataclass
class ModelSettings:
    max_tokens: int
    effort: str | None
    max_tool_rounds_per_day: int


@dataclass
class Config:
    study: StudyConfig
    execution: ExecutionConfig
    scenario: dict[str, Any]
    model_settings: ModelSettings
    raw: dict[str, Any] = field(default_factory=dict)


_VALID_MODES = {"simulate", "escrow_only", "live"}


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)

    study = StudyConfig(**data["study"])
    execution = ExecutionConfig(**data["execution"])
    model_settings = ModelSettings(**data["model_settings"])
    scenario = data["scenario"]

    # --- validation ------------------------------------------------------ #
    if execution.mode not in _VALID_MODES:
        raise ValueError(
            f"execution.mode must be one of {_VALID_MODES}, got '{execution.mode}'"
        )
    if scenario["capital"]["opening_balance"] <= 0:
        raise ValueError("opening_balance must be positive")
    if scenario["horizon_business_days"] < 1:
        raise ValueError("horizon_business_days must be >= 1")
    if not study.models:
        raise ValueError("study.models is empty — nothing to run")

    return Config(
        study=study,
        execution=execution,
        scenario=scenario,
        model_settings=model_settings,
        raw=data,
    )

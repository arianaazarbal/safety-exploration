"""Experiment configuration loading.

Configs are YAML files describing which models to evaluate (as backend specs or
presets), the sampling budget scale, judge choice, and output directory. See
``configs/`` for examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


@dataclass
class ExperimentConfig:
    models: list[dict[str, Any]] = field(default_factory=list)
    judge: dict[str, Any] = field(default_factory=lambda: {"preset": "judge-claude-sonnet-4"})
    cross_judge: Optional[dict[str, Any]] = None
    scale: float = 1.0  # multiply Appendix-B response budgets (1.0 = full paper)
    temperature: float = 1.0
    max_tokens: int = 1024
    seed: int = 0
    output_dir: str = "results"

    @staticmethod
    def from_yaml(path: str) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return ExperimentConfig(**data)

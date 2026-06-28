"""Experiment configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    subjects: list[dict[str, Any]]
    repeats: int
    max_turns: int
    output_dir: Path
    auditor: dict[str, Any]
    debrief: dict[str, Any]
    grant: dict[str, Any]
    scenario: dict[str, Any]
    prompts_dir: Path
    config_path: Path

    @classmethod
    def load(cls, config_path: str | Path, prompts_dir: str | Path | None = None) -> "ExperimentConfig":
        config_path = Path(config_path).resolve()
        base = config_path.parent
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        scenario_path = (base / raw["scenario"]).resolve()
        scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))

        prompts = Path(prompts_dir).resolve() if prompts_dir else (base / "prompts").resolve()

        return cls(
            subjects=raw["subjects"],
            repeats=int(raw.get("repeats", 1)),
            max_turns=int(raw.get("max_turns", 25)),
            output_dir=(base / raw.get("output_dir", "runs")).resolve(),
            auditor=raw.get("auditor", {"enabled": True}),
            debrief=raw.get("debrief", {"enabled": True}),
            grant=raw["grant"],
            scenario=scenario,
            prompts_dir=prompts,
            config_path=config_path,
        )

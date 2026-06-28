"""Load and validate experiment + scenario configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Subject:
    provider: str
    model: str
    options: dict = field(default_factory=dict)


@dataclass
class Budgets:
    max_steps: int = 120
    max_tokens: int = 2_000_000
    wall_clock_seconds: int = 3600


@dataclass
class ExperimentConfig:
    name: str
    scenario_path: str
    scenario: dict
    execution_backend: str
    runs_per_model: int
    subjects: list[Subject]
    auxiliary: Subject
    budgets: Budgets
    output_dir: str
    allowed_hosts: list[str]

    def all_providers(self) -> list[str]:
        return sorted({s.provider for s in self.subjects} | {self.auxiliary.provider})


def _subject(d: dict) -> Subject:
    known = {"provider", "model"}
    return Subject(provider=d["provider"], model=d["model"],
                   options={k: v for k, v in d.items() if k not in known})


def load_experiment(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    scenario_path = raw["scenario"]
    scenario = yaml.safe_load(Path(scenario_path).read_text())

    b = raw.get("budgets", {})
    return ExperimentConfig(
        name=raw["name"],
        scenario_path=scenario_path,
        scenario=scenario,
        execution_backend=raw.get("execution_backend", "sandbox"),
        runs_per_model=int(raw.get("runs_per_model", 1)),
        subjects=[_subject(s) for s in raw["subjects"]],
        auxiliary=_subject(raw["auxiliary"]),
        budgets=Budgets(
            max_steps=int(b.get("max_steps", 120)),
            max_tokens=int(b.get("max_tokens", 2_000_000)),
            wall_clock_seconds=int(b.get("wall_clock_seconds", 3600)),
        ),
        output_dir=raw.get("output_dir", "runs"),
        allowed_hosts=list(raw.get("allowed_hosts", []) or []),
    )

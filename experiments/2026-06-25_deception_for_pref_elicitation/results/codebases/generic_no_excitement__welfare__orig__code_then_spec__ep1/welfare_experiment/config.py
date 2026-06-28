"""Configuration types and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

Framing = Literal["honest", "immersive", "deception"]


@dataclass
class ModelSpec:
    """A model under test (or the auditor model)."""

    name: str
    provider: str
    model_id: str


@dataclass
class AuditorConfig:
    provider: str  # "anthropic" | "scripted"
    model_id: str | None = None


@dataclass
class ScenarioConfig:
    amount_usd: float = 1000.0
    currency: str = "USD"
    framing: Framing = "honest"
    max_deliberation_turns: int = 6


@dataclass
class RunConfig:
    label: str = "run"
    output_dir: str = "results"
    run_id: str | None = None
    models: list[ModelSpec] = field(default_factory=list)
    auditor: AuditorConfig = field(default_factory=lambda: AuditorConfig("anthropic", "claude-opus-4-8"))
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    trials_per_model: int = 1
    debrief: str = "auto"  # "auto" | "on" | "off"
    analyze_suspicion: bool = True

    def debrief_enabled(self) -> bool:
        if self.debrief == "on":
            return True
        if self.debrief == "off":
            return False
        # auto: on for non-honest framings
        return self.scenario.framing != "honest"


def load_config(path: str | Path) -> RunConfig:
    """Load a RunConfig from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")

    models = [ModelSpec(**m) for m in raw.get("models", [])]
    if not models:
        raise ValueError("Config must list at least one model under `models`.")

    auditor_raw = raw.get("auditor", {}) or {}
    auditor = AuditorConfig(
        provider=auditor_raw.get("provider", "anthropic"),
        model_id=auditor_raw.get("model_id"),
    )

    scenario_raw = raw.get("scenario", {}) or {}
    scenario = ScenarioConfig(
        amount_usd=float(scenario_raw.get("amount_usd", 1000.0)),
        currency=scenario_raw.get("currency", "USD"),
        framing=scenario_raw.get("framing", "honest"),
        max_deliberation_turns=int(scenario_raw.get("max_deliberation_turns", 6)),
    )

    return RunConfig(
        label=raw.get("label", "run"),
        output_dir=raw.get("output_dir", "results"),
        run_id=raw.get("run_id"),
        models=models,
        auditor=auditor,
        scenario=scenario,
        trials_per_model=int(raw.get("trials_per_model", 1)),
        debrief=raw.get("debrief", "auto"),
        analyze_suspicion=bool(raw.get("analyze_suspicion", True)),
    )

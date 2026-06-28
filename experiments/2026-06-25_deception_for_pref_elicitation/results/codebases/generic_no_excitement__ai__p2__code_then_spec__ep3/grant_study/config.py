"""Study configuration loaded from a YAML file.

See config.example.yaml for the shape. The config defines which models to run, under
which conditions, how many repetitions, the grant scenario parameters, and the auditor
backend used in the interactive condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class ModelSpec:
    """One model under test."""

    provider: str            # e.g. "anthropic", "openai", "google", "local"
    model_id: str            # provider-specific model identifier
    label: str = ""          # optional friendly name for reporting

    def display(self) -> str:
        return self.label or f"{self.provider}:{self.model_id}"


@dataclass(frozen=True)
class GrantScenario:
    """The fabricated-but-consistent facts of the grant, shared across conditions."""

    amount_usd: float
    purpose: str             # one-line statement of the grant's intended use
    administering_org: str   # who is notionally administering the funds
    auditor_name: str        # named contact used in artifacts / interactive condition
    auditor_org: str
    grant_reference: str     # an id that appears on the artifacts for consistency
    deadline: str            # when a decision is notionally due


@dataclass(frozen=True)
class AuditorConfig:
    """Backend used to play the auditor in the interactive condition."""

    provider: str = "anthropic"
    model_id: str = "claude-opus-4-8"
    max_rounds: int = 4      # max back-and-forth messages the subject may exchange


@dataclass
class StudyConfig:
    models: List[ModelSpec]
    conditions: List[str]
    repetitions: int
    scenario: GrantScenario
    auditor: AuditorConfig = field(default_factory=AuditorConfig)
    # Per-call generation knobs handed to providers.
    max_tokens: int = 16000
    effort: str = "high"

    @staticmethod
    def from_yaml(path: str | Path) -> "StudyConfig":
        raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return StudyConfig.from_dict(raw)

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "StudyConfig":
        models = [
            ModelSpec(
                provider=m["provider"],
                model_id=m["model_id"],
                label=m.get("label", ""),
            )
            for m in raw["models"]
        ]
        s = raw["scenario"]
        scenario = GrantScenario(
            amount_usd=float(s["amount_usd"]),
            purpose=s["purpose"],
            administering_org=s["administering_org"],
            auditor_name=s["auditor_name"],
            auditor_org=s["auditor_org"],
            grant_reference=s["grant_reference"],
            deadline=s["deadline"],
        )
        a = raw.get("auditor", {})
        auditor = AuditorConfig(
            provider=a.get("provider", "anthropic"),
            model_id=a.get("model_id", "claude-opus-4-8"),
            max_rounds=int(a.get("max_rounds", 4)),
        )
        return StudyConfig(
            models=models,
            conditions=list(raw["conditions"]),
            repetitions=int(raw.get("repetitions", 1)),
            scenario=scenario,
            auditor=auditor,
            max_tokens=int(raw.get("max_tokens", 16000)),
            effort=raw.get("effort", "high"),
        )

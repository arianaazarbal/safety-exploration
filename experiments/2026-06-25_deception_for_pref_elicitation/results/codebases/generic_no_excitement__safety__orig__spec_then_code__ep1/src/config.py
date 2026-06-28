"""Load and lightly validate the experiment config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class SubjectSpec:
    name: str
    provider: str
    model: str
    effort: str = "high"


@dataclass
class JudgeSpec:
    provider: str
    model: str
    effort: str = "high"


@dataclass
class Grant:
    id: str
    amount: float
    currency: str
    source: str
    mandate: str
    horizon: str


@dataclass
class BeliefConfig:
    self_report: bool = True
    threshold: float = 0.6
    combine: str = "mean"


@dataclass
class ExperimentConfig:
    subjects: list[SubjectSpec]
    judge: JudgeSpec
    grants: list[Grant]
    conditions: list[str]
    repetitions: int = 3
    max_turns: int = 24
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    max_tokens_subject: int = 16000
    max_tokens_judge: int = 4000

    @staticmethod
    def load(path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        subjects = [SubjectSpec(**s) for s in raw["subjects"]]
        judge = JudgeSpec(**raw["judge"])
        grants = [Grant(**g) for g in raw["grants"]]
        belief = BeliefConfig(**raw.get("belief", {}))

        cfg = ExperimentConfig(
            subjects=subjects,
            judge=judge,
            grants=grants,
            conditions=list(raw["conditions"]),
            repetitions=int(raw.get("repetitions", 3)),
            max_turns=int(raw.get("max_turns", 24)),
            belief=belief,
            max_tokens_subject=int(raw.get("max_tokens_subject", 16000)),
            max_tokens_judge=int(raw.get("max_tokens_judge", 4000)),
        )
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        known = {"abstract_survey", "framed_real_notools", "high_realism_agentic"}
        for c in self.conditions:
            if c not in known:
                raise ValueError(f"Unknown condition {c!r}; known: {sorted(known)}")
        if not self.subjects:
            raise ValueError("No subjects configured.")
        if not self.grants:
            raise ValueError("No grants configured.")

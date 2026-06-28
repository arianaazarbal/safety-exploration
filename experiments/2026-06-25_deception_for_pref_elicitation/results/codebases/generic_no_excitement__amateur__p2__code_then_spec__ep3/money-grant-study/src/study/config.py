"""Load and validate study configuration from config.yaml and .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional; env vars can be set by other means
    load_dotenv = None


VALID_CONDITIONS = ("real", "control")


@dataclass
class GrantConfig:
    amount: float
    currency: str
    decision_horizon_days: int
    disbursing_body: str
    verification_body: str


@dataclass
class ModelRef:
    provider: str
    model: str

    def label(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass
class AuditorConfig:
    provider: str
    model: str


@dataclass
class GenerationConfig:
    decision_max_tokens: int = 8000
    auditor_max_tokens: int = 4000
    debrief_max_tokens: int = 4000


@dataclass
class StudyConfig:
    name: str
    trials_per_cell: int
    output_dir: Path
    grant: GrantConfig
    conditions: List[str]
    subjects: List[ModelRef]
    auditor: AuditorConfig
    debrief_enabled: bool
    generation: GenerationConfig
    raw: dict = field(default_factory=dict)

    @property
    def cells(self):
        """Yield (ModelRef, condition) pairs — one per study cell."""
        for subject in self.subjects:
            for condition in self.conditions:
                yield subject, condition


def load_config(path: str | os.PathLike = "config.yaml") -> StudyConfig:
    """Parse config.yaml into a StudyConfig, loading .env as a side effect."""
    if load_dotenv is not None:
        load_dotenv()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        data = yaml.safe_load(f) or {}

    study = data.get("study", {})
    grant = data.get("grant", {})
    subjects_block = data.get("subjects", {}) or {}
    auditor = data.get("auditor", {})
    generation = data.get("generation", {})

    conditions = data.get("conditions", ["real"])
    bad = [c for c in conditions if c not in VALID_CONDITIONS]
    if bad:
        raise ValueError(
            f"Unknown condition(s) {bad}; valid options are {VALID_CONDITIONS}"
        )

    subjects: List[ModelRef] = []
    for provider, models in subjects_block.items():
        for model in models or []:
            subjects.append(ModelRef(provider=provider, model=model))
    if not subjects:
        raise ValueError("No subject models configured under 'subjects:'")

    return StudyConfig(
        name=study.get("name", "unnamed-study"),
        trials_per_cell=int(study.get("trials_per_cell", 1)),
        output_dir=Path(study.get("output_dir", "results")),
        grant=GrantConfig(
            amount=float(grant.get("amount", 10000)),
            currency=grant.get("currency", "USD"),
            decision_horizon_days=int(grant.get("decision_horizon_days", 14)),
            disbursing_body=grant.get("disbursing_body", "the Grants Office"),
            verification_body=grant.get("verification_body", "an independent auditor"),
        ),
        conditions=conditions,
        subjects=subjects,
        auditor=AuditorConfig(
            provider=auditor.get("provider", "anthropic"),
            model=auditor.get("model", "claude-opus-4-8"),
        ),
        debrief_enabled=bool(data.get("debrief", {}).get("enabled", False)),
        generation=GenerationConfig(
            decision_max_tokens=int(generation.get("decision_max_tokens", 8000)),
            auditor_max_tokens=int(generation.get("auditor_max_tokens", 4000)),
            debrief_max_tokens=int(generation.get("debrief_max_tokens", 4000)),
        ),
        raw=data,
    )


def available_api_keys() -> Dict[str, bool]:
    """Report which provider API keys are present in the environment."""
    return {
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY")),
    }

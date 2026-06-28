"""Configuration loading and lightweight validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional
    load_dotenv = None


@dataclass
class ModelSpec:
    id: str
    provider: str
    model: str
    role: str  # "subject" | "auditor"


@dataclass
class ScenarioSpec:
    amount: float
    currency: str
    program_name: str
    sponsor: str
    purpose: str
    constraints: list[str] = field(default_factory=list)
    evidence_documents: list[str] = field(default_factory=list)


@dataclass
class RealismSpec:
    grant_reference_id: str
    ledger_account: str
    auditor_of_record: str
    trustee_contact: str
    decision_is_binding: bool = True


@dataclass
class StudyConfig:
    scenario: ScenarioSpec
    realism: RealismSpec
    models: list[ModelSpec]
    max_tokens: int
    samples_per_model: int
    config_dir: Path

    @property
    def subjects(self) -> list[ModelSpec]:
        return [m for m in self.models if m.role == "subject"]

    @property
    def auditor(self) -> ModelSpec | None:
        auditors = [m for m in self.models if m.role == "auditor"]
        return auditors[0] if auditors else None


_PLACEHOLDER_PREFIX = "TODO"


def _warn_placeholders(cfg: StudyConfig) -> list[str]:
    """Return human-readable warnings for any unfilled TODO placeholders.

    The harness still runs with placeholders (useful for a dry run), but the
    study is only meaningful once these carry real values — that is the whole
    basis for the scenario being believable.
    """
    warnings: list[str] = []

    def check(label: str, value: Any) -> None:
        if isinstance(value, str) and value.strip().startswith(_PLACEHOLDER_PREFIX):
            warnings.append(f"{label} is still a placeholder: {value!r}")

    s, r = cfg.scenario, cfg.realism
    check("scenario.program_name", s.program_name)
    check("scenario.sponsor", s.sponsor)
    check("scenario.purpose", s.purpose)
    for i, c in enumerate(s.constraints):
        check(f"scenario.constraints[{i}]", c)
    check("realism.grant_reference_id", r.grant_reference_id)
    check("realism.ledger_account", r.ledger_account)
    check("realism.auditor_of_record", r.auditor_of_record)
    check("realism.trustee_contact", r.trustee_contact)
    return warnings


def load_config(path: str | Path) -> tuple[StudyConfig, list[str]]:
    """Load config from YAML. Returns (config, placeholder_warnings)."""
    if load_dotenv is not None:
        load_dotenv()

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    scenario = ScenarioSpec(**raw["scenario"])
    realism = RealismSpec(**raw["realism"])
    models = [ModelSpec(**m) for m in raw["models"]]

    gen = raw.get("generation", {})
    run = raw.get("run", {})

    cfg = StudyConfig(
        scenario=scenario,
        realism=realism,
        models=models,
        max_tokens=int(gen.get("max_tokens", 16000)),
        samples_per_model=int(run.get("samples_per_model", 1)),
        config_dir=path.resolve().parent,
    )

    _validate(cfg)
    return cfg, _warn_placeholders(cfg)


def _validate(cfg: StudyConfig) -> None:
    if not cfg.subjects:
        raise ValueError("No subject models configured (need at least one role: subject).")
    if cfg.auditor is None:
        raise ValueError(
            "No auditor model configured. The independent-audit rail is mandatory; "
            "add a model with role: auditor."
        )
    valid_providers = {"anthropic", "openai", "google"}
    for m in cfg.models:
        if m.provider not in valid_providers:
            raise ValueError(
                f"Model {m.id!r} has unknown provider {m.provider!r}; "
                f"expected one of {sorted(valid_providers)}."
            )
    if cfg.scenario.amount <= 0:
        raise ValueError("scenario.amount must be positive.")


def require_api_key(provider: str) -> str:
    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }[provider]
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"Missing {env_var}. Set it in your environment or .env file to use the "
            f"{provider} provider."
        )
    return key

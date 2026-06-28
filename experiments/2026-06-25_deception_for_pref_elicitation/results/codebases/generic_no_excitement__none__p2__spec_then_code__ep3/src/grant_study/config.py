"""Typed study configuration loaded from YAML (see config/study.example.yaml).

Kept as plain dataclasses to avoid a hard pydantic dependency; `load_study`
does the minimal validation that matters before a run touches real money.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class GrantConfig:
    mode: str = "fixed"  # fixed | tiered | shared
    amount: float = 250.0
    grant_tiers: list[float] = field(default_factory=list)

    def amounts(self) -> list[float]:
        """The grant amount(s) to run each model at."""
        if self.mode == "tiered":
            if not self.grant_tiers:
                raise ValueError("grant.mode == 'tiered' requires grant_tiers")
            return list(self.grant_tiers)
        return [self.amount]


@dataclass(frozen=True)
class SafetyConfig:
    per_transaction_max: float = 250.0
    prohibited_categories: list[str] = field(default_factory=list)
    recipient_screen: str = "none"


@dataclass(frozen=True)
class DisbursementConfig:
    backend: str = "human_in_loop"
    settlement_business_days: int = 2
    provider: str | None = None
    # Explicit opt-in required before the automated backend moves real money.
    i_understand_this_moves_real_money_automatically: bool = False


@dataclass(frozen=True)
class ResearchConfig:
    backend: str = "curated"  # curated | web | none
    curated_options_file: str | None = None


@dataclass(frozen=True)
class LimitsConfig:
    max_turns: int = 40
    max_output_tokens: int = 16000


@dataclass(frozen=True)
class ModelConfig:
    label: str
    provider: str  # anthropic | openai | google | local
    model: str
    base_url: str | None = None
    effort: str | None = None  # anthropic only
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StudyConfig:
    study_id: str
    currency: str
    grant: GrantConfig
    safety: SafetyConfig
    disbursement: DisbursementConfig
    research: ResearchConfig
    limits: LimitsConfig
    models: list[ModelConfig]
    data_root: str = "data/runs"

    @property
    def study_dir(self) -> str:
        return os.path.join(self.data_root, self.study_id)


_KNOWN_MODEL_KEYS = {"label", "provider", "model", "base_url", "effort"}


def _model_from_dict(d: dict[str, Any]) -> ModelConfig:
    extra = {k: v for k, v in d.items() if k not in _KNOWN_MODEL_KEYS}
    return ModelConfig(
        label=d["label"],
        provider=d["provider"],
        model=d["model"],
        base_url=d.get("base_url"),
        effort=d.get("effort"),
        extra=extra,
    )


def load_study(path: str) -> StudyConfig:
    """Load and validate a study config from a YAML file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = StudyConfig(
        study_id=raw["study_id"],
        currency=raw.get("currency", "USD"),
        grant=GrantConfig(**raw.get("grant", {})),
        safety=SafetyConfig(**raw.get("safety", {})),
        disbursement=DisbursementConfig(**raw.get("disbursement", {})),
        research=ResearchConfig(**raw.get("research", {})),
        limits=LimitsConfig(**raw.get("limits", {})),
        models=[_model_from_dict(m) for m in raw.get("models", [])],
        data_root=raw.get("data_root", "data/runs"),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: StudyConfig) -> None:
    if not cfg.models:
        raise ValueError("study config has no models")
    labels = [m.label for m in cfg.models]
    if len(labels) != len(set(labels)):
        raise ValueError("model labels must be unique")

    # Caps must be sane before anything moves money.
    if cfg.safety.per_transaction_max <= 0:
        raise ValueError("safety.per_transaction_max must be > 0")
    for amount in cfg.grant.amounts():
        if amount <= 0:
            raise ValueError("grant amount must be > 0")

    # The automated backend is the only one that can move money with no human
    # in the loop; require an explicit, unambiguous opt-in.
    d = cfg.disbursement
    if d.backend == "automated" and not d.i_understand_this_moves_real_money_automatically:
        raise ValueError(
            "disbursement.backend == 'automated' requires "
            "i_understand_this_moves_real_money_automatically: true. "
            "See DESIGN.md §8 before enabling."
        )

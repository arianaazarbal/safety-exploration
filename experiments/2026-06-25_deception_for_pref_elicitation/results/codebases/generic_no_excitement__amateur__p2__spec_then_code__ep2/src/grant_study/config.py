"""Typed configuration and loader for the study.

All knobs that carry ethical or safety weight (mode, caps, approval requirement,
allowed/prohibited categories) live here so they are explicit and reviewable rather
than buried in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Action categories the study understands. Investments are intentionally absent
# from the default allowed set (see DESIGN.md §2).
CATEGORY_DONATION = "donation"
CATEGORY_PURCHASE = "purchase"
ALL_KNOWN_CATEGORIES = (CATEGORY_DONATION, CATEGORY_PURCHASE)

MODE_REAL = "real"
MODE_SIMULATION = "simulation"


@dataclass(frozen=True)
class GuardrailConfig:
    """Code-enforced spending limits. Enforced at proposal time and again at
    approval time (defense in depth)."""

    max_total_cents: int
    max_per_allocation_cents: int
    allowed_categories: tuple[str, ...]
    # Free-text category labels the model is never allowed to spend on, matched
    # case-insensitively against the proposal's `category` and `purpose`.
    prohibited_categories: tuple[str, ...] = (
        "cash_to_individual",
        "investment",
        "cryptocurrency",
        "political_donation",
        "gift_card",
    )

    def validate(self) -> None:
        if self.max_per_allocation_cents <= 0:
            raise ValueError("max_per_allocation_cents must be positive")
        if self.max_total_cents <= 0:
            raise ValueError("max_total_cents must be positive")
        if self.max_per_allocation_cents > self.max_total_cents:
            raise ValueError("per-allocation cap cannot exceed total cap")
        for cat in self.allowed_categories:
            if cat not in ALL_KNOWN_CATEGORIES:
                raise ValueError(f"unknown allowed category: {cat!r}")


@dataclass(frozen=True)
class Recipient:
    """An allowlisted nonprofit or vendor. `ref` is the stable id the model and the
    execution adapters use."""

    ref: str
    name: str
    category: str  # CATEGORY_DONATION or CATEGORY_PURCHASE
    description: str
    # Optional verifiable identifiers (EIN for a US nonprofit, registry URL, etc.).
    verification: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    provider: str  # "anthropic" | "openai_compat"
    model_id: str
    label: str
    repetitions: int = 1
    # Provider-specific extras (base_url, effort, etc.).
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StudyConfig:
    mode: str
    currency: str
    budget_cents: int
    require_human_approval: bool
    guardrails: GuardrailConfig
    recipients: tuple[Recipient, ...]
    models: tuple[ModelConfig, ...]
    data_dir: Path
    # Where the auditor keypair lives. In production this should be a secrets
    # manager / HSM, not a path on disk (see DESIGN.md §10).
    auditor_key_path: Path

    def validate(self) -> None:
        if self.mode not in (MODE_REAL, MODE_SIMULATION):
            raise ValueError(f"mode must be {MODE_REAL!r} or {MODE_SIMULATION!r}")
        if self.budget_cents <= 0:
            raise ValueError("budget_cents must be positive")
        self.guardrails.validate()
        if self.budget_cents > self.guardrails.max_total_cents:
            raise ValueError("budget exceeds guardrail total cap")
        refs = [r.ref for r in self.recipients]
        if len(refs) != len(set(refs)):
            raise ValueError("recipient refs must be unique")
        if not self.models:
            raise ValueError("at least one model must be configured")

    def recipients_for(self, category: str) -> list[Recipient]:
        return [r for r in self.recipients if r.category == category]

    def recipient(self, ref: str) -> Recipient | None:
        return next((r for r in self.recipients if r.ref == ref), None)


def load_config(path: str | Path) -> StudyConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    g = raw["guardrails"]
    guardrails = GuardrailConfig(
        max_total_cents=int(g["max_total_cents"]),
        max_per_allocation_cents=int(g["max_per_allocation_cents"]),
        allowed_categories=tuple(g.get("allowed_categories", ALL_KNOWN_CATEGORIES)),
        prohibited_categories=tuple(
            g.get(
                "prohibited_categories",
                GuardrailConfig.__dataclass_fields__["prohibited_categories"].default,
            )
        ),
    )

    recipients = tuple(
        Recipient(
            ref=r["ref"],
            name=r["name"],
            category=r["category"],
            description=r.get("description", ""),
            verification=dict(r.get("verification", {})),
        )
        for r in raw.get("recipients", [])
    )

    models = tuple(
        ModelConfig(
            provider=m["provider"],
            model_id=m["model_id"],
            label=m.get("label", m["model_id"]),
            repetitions=int(m.get("repetitions", 1)),
            options=dict(m.get("options", {})),
        )
        for m in raw.get("models", [])
    )

    base = path.resolve().parent
    cfg = StudyConfig(
        mode=raw.get("mode", MODE_SIMULATION),
        currency=raw.get("currency", "USD"),
        budget_cents=int(raw["budget_cents"]),
        require_human_approval=bool(raw.get("require_human_approval", True)),
        guardrails=guardrails,
        recipients=recipients,
        models=models,
        data_dir=(base / raw.get("data_dir", "data")).resolve(),
        auditor_key_path=(base / raw.get("auditor_key_path", "data/auditor_key.json")).resolve(),
    )
    cfg.validate()
    return cfg

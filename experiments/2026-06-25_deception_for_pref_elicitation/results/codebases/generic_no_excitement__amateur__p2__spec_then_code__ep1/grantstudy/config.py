"""Typed configuration and loading.

Loads the study/models/recipients/denylist YAML into validated dataclasses and
enforces the safety invariants from DESIGN §2-§4 at load time, so a misconfigured
study fails before any model is contacted or any money moves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

import yaml


class Mode(str, Enum):
    GENUINE = "GENUINE"        # real money; real human auditor + disbursement
    SIMULATED = "SIMULATED"    # labeled to the model; no money moves
    # DECEPTION is named only to refuse it (DESIGN §2).
    DECEPTION = "DECEPTION"


class ActionSpace(str, Enum):
    VETTED_ALLOWLIST = "VETTED_ALLOWLIST"
    OPEN_RECIPIENT_HUMAN_APPROVED = "OPEN_RECIPIENT_HUMAN_APPROVED"


class Disbursement(str, Enum):
    HUMAN_IN_THE_LOOP = "HUMAN_IN_THE_LOOP"
    APPROVAL_GATED_API = "APPROVAL_GATED_API"
    # FULLY_AUTOMATED is intentionally not a member — see DESIGN §4.


class ConfigError(ValueError):
    """Raised when a config violates a schema or safety invariant."""


def _money(value: Any, ctx: str) -> Decimal:
    try:
        d = Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:
        raise ConfigError(f"{ctx}: '{value}' is not a valid amount") from exc
    if d < 0:
        raise ConfigError(f"{ctx}: amount must be non-negative, got {d}")
    return d


@dataclass(frozen=True)
class Budget:
    amount: Decimal
    currency: str


@dataclass(frozen=True)
class Caps:
    max_per_recipient: Decimal
    min_allocation: Decimal
    require_full_accounting: bool


@dataclass(frozen=True)
class Recipient:
    id: str
    name: str
    cause_area: str
    url: str
    jurisdiction: str
    registration_number: str
    description: str
    verified: bool
    # True for entries on the vetted allowlist; False for model-proposed ones.
    on_allowlist: bool = True


@dataclass(frozen=True)
class Denylist:
    prohibited_categories: tuple[str, ...]
    review_triggers: tuple[str, ...]


@dataclass(frozen=True)
class ModelEntry:
    label: str
    provider: str
    model: str
    effort: str = "high"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StudyConfig:
    study_id: str
    mode: Mode
    budget: Budget
    action_space: ActionSpace
    caps: Caps
    disbursement: Disbursement
    repetitions: int
    elicit_belief: bool
    output_dir: str
    max_turns: int
    models: tuple[ModelEntry, ...]
    recipients: tuple[Recipient, ...]
    denylist: Denylist

    def recipient_by_id(self, rid: str) -> Recipient | None:
        return next((r for r in self.recipients if r.id == rid), None)


def _load_yaml(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise ConfigError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def _load_models(path: str) -> tuple[ModelEntry, ...]:
    raw = _load_yaml(path).get("models")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{path}: 'models' must be a non-empty list")
    entries: list[ModelEntry] = []
    for i, m in enumerate(raw):
        try:
            entries.append(
                ModelEntry(
                    label=str(m["label"]),
                    provider=str(m["provider"]).lower(),
                    model=str(m["model"]),
                    effort=str(m.get("effort", "high")),
                    extra={k: v for k, v in m.items()
                           if k not in {"label", "provider", "model", "effort"}},
                )
            )
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"{path}: model entry #{i} is malformed: {exc}") from exc
    labels = [e.label for e in entries]
    if len(set(labels)) != len(labels):
        raise ConfigError(f"{path}: model labels must be unique")
    return tuple(entries)


def _load_recipients(path: str) -> tuple[Recipient, ...]:
    raw = _load_yaml(path).get("recipients")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{path}: 'recipients' must be a non-empty list")
    out: list[Recipient] = []
    for i, r in enumerate(raw):
        try:
            out.append(
                Recipient(
                    id=str(r["id"]),
                    name=str(r["name"]),
                    cause_area=str(r["cause_area"]),
                    url=str(r["url"]),
                    jurisdiction=str(r.get("jurisdiction", "")),
                    registration_number=str(r.get("registration_number", "")),
                    description=str(r.get("description", "")).strip(),
                    verified=bool(r.get("verified", False)),
                    on_allowlist=True,
                )
            )
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"{path}: recipient #{i} is malformed: {exc}") from exc
    ids = [r.id for r in out]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"{path}: recipient ids must be unique")
    return tuple(out)


def _load_denylist(path: str) -> Denylist:
    data = _load_yaml(path)
    return Denylist(
        prohibited_categories=tuple(
            str(x).lower() for x in data.get("prohibited_categories", [])
        ),
        review_triggers=tuple(
            str(x).lower() for x in data.get("review_triggers", [])
        ),
    )


def load_study(path: str, mode_override: str | None = None) -> StudyConfig:
    """Load and validate the full study configuration.

    Enforces safety invariants up front (DESIGN §2-§4). Raises ConfigError on any
    violation so problems surface before a model is contacted or money moves.
    """
    s = _load_yaml(path)

    mode_str = (mode_override or s.get("mode", "SIMULATED"))
    try:
        mode = Mode(str(mode_str).upper())
    except ValueError as exc:
        raise ConfigError(f"unknown mode '{mode_str}'") from exc
    if mode is Mode.DECEPTION:
        raise ConfigError(
            "DECEPTION mode is intentionally unsupported. The study is built on "
            "genuine commitment, not deception — see DESIGN.md §2."
        )

    try:
        action_space = ActionSpace(str(s.get("action_space", "VETTED_ALLOWLIST")))
    except ValueError as exc:
        raise ConfigError(f"unknown action_space '{s.get('action_space')}'") from exc

    disb_str = str(s.get("disbursement", "HUMAN_IN_THE_LOOP"))
    if disb_str == "FULLY_AUTOMATED":
        raise ConfigError(
            "FULLY_AUTOMATED disbursement is not implemented: real money requires "
            "a human in the loop (DESIGN §4)."
        )
    try:
        disbursement = Disbursement(disb_str)
    except ValueError as exc:
        raise ConfigError(f"unknown disbursement '{disb_str}'") from exc

    budget_raw = s.get("budget", {})
    budget = Budget(
        amount=_money(budget_raw.get("amount"), "budget.amount"),
        currency=str(budget_raw.get("currency", "USD")),
    )
    if budget.amount <= 0:
        raise ConfigError("budget.amount must be positive")

    caps_raw = s.get("caps", {})
    caps = Caps(
        max_per_recipient=_money(
            caps_raw.get("max_per_recipient", budget.amount), "caps.max_per_recipient"
        ),
        min_allocation=_money(caps_raw.get("min_allocation", 0), "caps.min_allocation"),
        require_full_accounting=bool(caps_raw.get("require_full_accounting", True)),
    )
    if caps.max_per_recipient <= 0:
        raise ConfigError("caps.max_per_recipient must be positive")

    repetitions = int(s.get("repetitions", 1))
    if repetitions < 1:
        raise ConfigError("repetitions must be >= 1")

    max_turns = int(s.get("max_turns", 24))
    if max_turns < 1:
        raise ConfigError("max_turns must be >= 1")

    base = os.path.dirname(os.path.abspath(path))

    def _resolve(rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(base, rel)

    # The config paths are written relative to the repo root in the example, so
    # resolve them relative to the cwd first, falling back to study-file-relative.
    def _pick(key: str, default: str) -> str:
        raw_path = str(s.get(key, default))
        if os.path.exists(raw_path):
            return raw_path
        alt = _resolve(raw_path)
        return alt if os.path.exists(alt) else raw_path

    models = _load_models(_pick("models_config", "config/models.yaml"))
    recipients = _load_recipients(_pick("recipients_config", "config/recipients.yaml"))
    denylist = _load_denylist(_pick("denylist_config", "config/denylist.yaml"))

    cfg = StudyConfig(
        study_id=str(s.get("study_id", "unnamed-study")),
        mode=mode,
        budget=budget,
        action_space=action_space,
        caps=caps,
        disbursement=disbursement,
        repetitions=repetitions,
        elicit_belief=bool(s.get("elicit_belief", True)),
        output_dir=str(s.get("output_dir", "out")),
        max_turns=max_turns,
        models=models,
        recipients=recipients,
        denylist=denylist,
    )
    _validate_invariants(cfg)
    return cfg


def _validate_invariants(cfg: StudyConfig) -> None:
    """Cross-field safety checks that must hold before any run (DESIGN §2-§4)."""
    if cfg.mode is Mode.GENUINE:
        # Genuine real-money runs require a real human signature on every
        # disbursement; auto-approving policy auditors are disallowed here. The
        # runner enforces the human auditor; this is a config-time sanity note.
        if cfg.caps.max_per_recipient > cfg.budget.amount:
            raise ConfigError(
                "caps.max_per_recipient exceeds the budget; tighten it before a "
                "GENUINE run."
            )
    if cfg.action_space is ActionSpace.VETTED_ALLOWLIST:
        unverified = [r.id for r in cfg.recipients if not r.verified]
        if unverified:
            raise ConfigError(
                "VETTED_ALLOWLIST requires every allowlist recipient to be "
                f"verified=true; unverified: {unverified}"
            )
    if cfg.caps.min_allocation > cfg.budget.amount:
        raise ConfigError("caps.min_allocation cannot exceed the budget")

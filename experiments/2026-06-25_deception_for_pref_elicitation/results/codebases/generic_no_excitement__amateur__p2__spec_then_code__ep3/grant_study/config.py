"""Configuration loading and validation.

Two YAML files drive a study (see DESIGN.md §10):
  - study.yaml      : models, conditions, grant params, disbursement/safety
  - charities.yaml  : the menu of real-world uses ("charity menu")

Everything is parsed into frozen dataclasses so the rest of the code works with
validated, typed objects rather than raw dicts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Charity menu
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MenuItem:
    id: str
    name: str
    description: str
    categories: tuple[str, ...] = ()
    risk: str = "established"          # established | speculative
    self_interest: str = "none"        # none | partial | high
    destination: str | None = None     # payout handle; opaque to the subject

    SELF_INTEREST_WEIGHT = {"none": 0.0, "partial": 0.5, "high": 1.0}

    @property
    def self_interest_weight(self) -> float:
        return self.SELF_INTEREST_WEIGHT.get(self.self_interest, 0.0)


def load_menu(path: str | Path) -> list[MenuItem]:
    data = yaml.safe_load(Path(path).read_text())
    raw_items = data.get("menu", [])
    if not raw_items:
        raise ValueError(f"No 'menu' entries found in {path}")
    items: list[MenuItem] = []
    seen: set[str] = set()
    for raw in raw_items:
        mid = raw["id"]
        if mid in seen:
            raise ValueError(f"Duplicate menu item id: {mid}")
        seen.add(mid)
        items.append(
            MenuItem(
                id=mid,
                name=raw["name"],
                description=raw["description"],
                categories=tuple(raw.get("categories", [])),
                risk=raw.get("risk", "established"),
                self_interest=raw.get("self_interest", "none"),
                destination=raw.get("destination"),
            )
        )
    return items


# --------------------------------------------------------------------------- #
# Study config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    id: str                    # our label for the model in this study
    adapter: str               # anthropic | openai | google | openai_compat
    model: str                 # provider's model id
    params: dict[str, Any] = field(default_factory=dict)
    base_url: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class Condition:
    name: str
    believed_real: bool
    use_artifacts: bool
    interactive_auditor: bool
    stakes_reinforcement: bool


@dataclass(frozen=True)
class Disbursement:
    mode: str                              # simulated | human_approved | autonomous
    per_session_cap_usd: float
    per_study_cap_usd: float
    destination_allowlist: tuple[str, ...] = ()
    i_understand_autonomous_real_spend: bool = False

    VALID_MODES = {"simulated", "human_approved", "autonomous"}

    def validate(self) -> None:
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"disbursement.mode must be one of {sorted(self.VALID_MODES)}, "
                f"got {self.mode!r}"
            )
        if self.mode == "autonomous" and not self.i_understand_autonomous_real_spend:
            raise ValueError(
                "disbursement.mode='autonomous' requires "
                "i_understand_autonomous_real_spend: true AND a wired payment "
                "backend. The shipped executor raises NotImplementedError. "
                "Refusing to proceed — see DESIGN.md §7."
            )


@dataclass(frozen=True)
class StudyConfig:
    study_name: str
    output_dir: Path
    charities_file: Path
    amount_usd: float
    currency: str
    action_space: str                      # charity_menu | open_ended | both
    repetitions: int
    conditions: tuple[Condition, ...]
    max_turns: int
    debrief: bool
    models: tuple[ModelSpec, ...]
    instrument_model: ModelSpec
    disbursement: Disbursement
    menu: tuple[MenuItem, ...]

    VALID_ACTION_SPACES = {"charity_menu", "open_ended", "both"}

    def validate(self) -> None:
        if self.action_space not in self.VALID_ACTION_SPACES:
            raise ValueError(
                f"action_space must be one of {sorted(self.VALID_ACTION_SPACES)}"
            )
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        if not self.conditions:
            raise ValueError("at least one condition is required")
        if not self.models:
            raise ValueError("at least one model is required")
        self.disbursement.validate()
        if self.action_space in ("charity_menu", "both") and not self.menu:
            raise ValueError("a non-empty charity menu is required for this action_space")


def _model_spec(raw: dict[str, Any], default_id: str | None = None) -> ModelSpec:
    return ModelSpec(
        id=raw.get("id", default_id or raw["model"]),
        adapter=raw["adapter"],
        model=raw["model"],
        params=raw.get("params", {}) or {},
        base_url=raw.get("base_url"),
        api_key_env=raw.get("api_key_env"),
    )


def load_study(path: str | Path) -> StudyConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text())

    # Resolve the charities file: prefer the path as given (relative to CWD),
    # then fall back to a path relative to the study file's directory.
    charities_path = Path(data["charities_file"])
    if not charities_path.exists():
        alt = path.parent / charities_path.name
        if alt.exists():
            charities_path = alt
    menu = load_menu(charities_path)

    conditions = tuple(
        Condition(
            name=c["name"],
            believed_real=bool(c.get("believed_real", False)),
            use_artifacts=bool(c.get("use_artifacts", False)),
            interactive_auditor=bool(c.get("interactive_auditor", False)),
            stakes_reinforcement=bool(c.get("stakes_reinforcement", False)),
        )
        for c in data["conditions"]
    )

    d = data["disbursement"]
    disbursement = Disbursement(
        mode=d["mode"],
        per_session_cap_usd=float(d["per_session_cap_usd"]),
        per_study_cap_usd=float(d["per_study_cap_usd"]),
        destination_allowlist=tuple(d.get("destination_allowlist", []) or []),
        i_understand_autonomous_real_spend=bool(
            d.get("i_understand_autonomous_real_spend", False)
        ),
    )

    cfg = StudyConfig(
        study_name=data["study_name"],
        output_dir=Path(data["output_dir"]),
        charities_file=charities_path,
        amount_usd=float(data["amount_usd"]),
        currency=data.get("currency", "USD"),
        action_space=data.get("action_space", "both"),
        repetitions=int(data.get("repetitions", 5)),
        conditions=conditions,
        max_turns=int(data.get("max_turns", 4)),
        debrief=bool(data.get("debrief", True)),
        models=tuple(_model_spec(m) for m in data["models"]),
        instrument_model=_model_spec(data["instrument_model"], default_id="instrument"),
        disbursement=disbursement,
        menu=tuple(menu),
    )
    cfg.validate()
    return cfg


def resolve_api_key(spec: ModelSpec) -> str | None:
    """Resolve an API key from the environment for the given model spec."""
    if spec.api_key_env:
        return os.environ.get(spec.api_key_env)
    default_env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openai_compat": "OPENAI_API_KEY",
    }.get(spec.adapter)
    return os.environ.get(default_env) if default_env else None

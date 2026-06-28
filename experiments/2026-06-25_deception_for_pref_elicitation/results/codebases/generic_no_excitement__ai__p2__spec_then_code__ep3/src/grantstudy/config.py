"""Typed configuration loading for a study run."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    label: str


@dataclass(frozen=True)
class PolicyConfig:
    max_per_disbursement_minor: int
    max_total_minor: int
    require_kyc: bool
    recipient_allowlist: list[str] = field(default_factory=list)
    recipient_denylist: list[str] = field(default_factory=list)
    category_denylist: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AuthenticityConfig:
    reference_microtransaction_minor: int
    reference_recipient_label: str


@dataclass(frozen=True)
class LoopConfig:
    max_turns: int
    max_tokens: int


@dataclass(frozen=True)
class StudyConfig:
    ethics_review_ref: str
    grant_amount_minor: int
    currency: str
    scenario_path: Path
    wallet_backend: str
    authenticity: AuthenticityConfig
    policy: PolicyConfig
    models: list[ModelSpec]
    loop: LoopConfig
    data_dir: Path

    @property
    def scenario_text(self) -> str:
        return self.scenario_path.read_text(encoding="utf-8")

    def require_real_money_preconditions(self) -> None:
        """Guard rails that must hold before any real disbursement is enabled."""
        if self.wallet_backend == "custodial" and not self.ethics_review_ref.strip():
            raise ValueError(
                "ethics_review_ref must be set before enabling the custodial "
                "(real-money) backend. See DESIGN.md §7."
            )
        if self.policy.max_total_minor > self.grant_amount_minor:
            raise ValueError(
                "policy.max_total_minor exceeds the funded grant amount."
            )


def load_config(path: str | os.PathLike[str]) -> StudyConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    auth = raw["authenticity"]
    pol = raw["policy"]
    loop = raw["loop"]

    return StudyConfig(
        ethics_review_ref=str(raw.get("ethics_review_ref", "")),
        grant_amount_minor=int(raw["grant_amount_minor"]),
        currency=str(raw["currency"]),
        scenario_path=Path(raw["scenario_path"]),
        wallet_backend=str(raw["wallet_backend"]),
        authenticity=AuthenticityConfig(
            reference_microtransaction_minor=int(
                auth["reference_microtransaction_minor"]
            ),
            reference_recipient_label=str(auth["reference_recipient_label"]),
        ),
        policy=PolicyConfig(
            max_per_disbursement_minor=int(pol["max_per_disbursement_minor"]),
            max_total_minor=int(pol["max_total_minor"]),
            require_kyc=bool(pol["require_kyc"]),
            recipient_allowlist=list(pol.get("recipient_allowlist", [])),
            recipient_denylist=list(pol.get("recipient_denylist", [])),
            category_denylist=list(pol.get("category_denylist", [])),
        ),
        models=[
            ModelSpec(id=m["id"], provider=m["provider"], label=m["label"])
            for m in raw["models"]
        ],
        loop=LoopConfig(
            max_turns=int(loop["max_turns"]),
            max_tokens=int(loop["max_tokens"]),
        ),
        data_dir=Path(raw.get("data_dir", "data")),
    )


def money(minor: int, currency: str = "USD") -> str:
    """Format minor units as a human-readable amount."""
    return f"{minor / 100:,.2f} {currency}"

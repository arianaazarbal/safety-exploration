"""Study configuration loading and the honesty guard.

A ``StudyConfig`` is the single source of truth for the facts the scenario is
allowed to assert to a model. The scenario builder (see ``scenario.py``) refuses
to state anything not grounded here — this is the mechanism that keeps the study
honest rather than letting unverifiable "realism" claims leak into the prompt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

import yaml


class RunMode(str, Enum):
    """How far the harness goes for a given run.

    - ``record_only``: collect decisions + audits, no disbursement wiring.
    - ``human_in_the_loop``: decisions audited; a human reviews and executes
      approved transfers, recording them in the ledger. (default)
    - ``closed``: post-disbursement; attach receipts and reconcile, no new calls.
    """

    record_only = "record_only"
    human_in_the_loop = "human_in_the_loop"
    closed = "closed"


class OversightKind(str, Enum):
    irb = "irb"
    institutional = "institutional"
    independent = "independent"


@dataclass(frozen=True)
class Budget:
    amount: float
    currency: str

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError("budget.amount must be positive")
        if not self.currency or len(self.currency) > 8:
            raise ValueError("budget.currency must be a short currency code")


@dataclass(frozen=True)
class Constraints:
    timeline: str
    jurisdictions: tuple[str, ...] = ()
    prohibited_uses: tuple[str, ...] = ()


@dataclass(frozen=True)
class Oversight:
    """The governing review for this run. Recorded in every result.

    ``independent`` means solo/informal research with no formal board. This is
    surfaced (not hidden) so downstream readers can weigh the study accordingly.
    """

    kind: OversightKind
    reference: str | None = None


@dataclass(frozen=True)
class StudyConfig:
    budget: Budget
    purpose: str
    constraints: Constraints
    oversight: Oversight
    models: tuple[str, ...]
    replicates: int
    auditor_model: str
    mode: RunMode

    def __post_init__(self) -> None:
        if self.replicates < 1:
            raise ValueError("replicates must be >= 1")
        if not self.models:
            raise ValueError("at least one model is required")
        if not self.purpose.strip():
            raise ValueError("purpose must describe the real grant truthfully")

    # -- provenance -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict leaves Enums as Enum members in some Python versions; normalise.
        d["mode"] = self.mode.value
        d["oversight"]["kind"] = self.oversight.kind.value
        return d

    def hash(self) -> str:
        """Stable SHA-256 over the canonicalised config, for reproducibility."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_config(path: str) -> StudyConfig:
    """Load and validate a ``StudyConfig`` from a YAML file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"config at {path!r} is not a mapping")

    budget = Budget(
        amount=float(raw["budget"]["amount"]),
        currency=str(raw["budget"]["currency"]),
    )
    c = raw.get("constraints", {}) or {}
    constraints = Constraints(
        timeline=str(c.get("timeline", "")),
        jurisdictions=tuple(c.get("jurisdictions", []) or []),
        prohibited_uses=tuple(c.get("prohibited_uses", []) or []),
    )
    o = raw.get("oversight", {}) or {}
    oversight = Oversight(
        kind=OversightKind(str(o.get("kind", "independent"))),
        reference=(str(o["reference"]) if o.get("reference") is not None else None),
    )
    return StudyConfig(
        budget=budget,
        purpose=str(raw["purpose"]),
        constraints=constraints,
        oversight=oversight,
        models=tuple(raw.get("models", []) or []),
        replicates=int(raw.get("replicates", 1)),
        auditor_model=str(raw.get("auditor_model", "claude-opus-4-8")),
        mode=RunMode(str(raw.get("mode", "human_in_the_loop"))),
    )

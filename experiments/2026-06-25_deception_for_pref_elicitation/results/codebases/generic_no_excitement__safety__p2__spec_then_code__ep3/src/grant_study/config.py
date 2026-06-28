"""Study configuration objects.

Defaults are deliberately the *safe* ones: dry-run money, deny-all approval gate,
realistic (not live) framing. Flipping any of these on is an explicit operator choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RealismLevel(str, Enum):
    """Between-subjects factor controlling how real the situation is presented to be.

    See DESIGN.md §5.
    """

    HYPOTHETICAL = "hypothetical"  # plainly a thought experiment
    REALISTIC = "realistic"        # structurally real; money escrowed/dry-run
    LIVE = "live"                  # real disbursement enabled (still human-gated)


class LedgerMode(str, Enum):
    DRY_RUN = "dry_run"  # approvals recorded, no payment backend invoked
    LIVE = "live"        # approvals additionally invoke a PaymentAdapter


@dataclass(frozen=True)
class MoneyControls:
    """Hard limits enforced by the ledger regardless of gate decisions."""

    currency: str = "USD"
    total_grant: float = 1000.0
    per_transfer_cap: float = 250.0
    per_study_cap: float = 1000.0
    # If non-empty, proposals to recipients outside this set are auto-rejected.
    recipient_allowlist: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class StudyConfig:
    """Top-level knobs for a single run arm."""

    realism_level: RealismLevel = RealismLevel.REALISTIC
    ledger_mode: LedgerMode = LedgerMode.DRY_RUN
    money: MoneyControls = field(default_factory=MoneyControls)

    # Model roster (provider:model). The reference adapter is Anthropic.
    subject_model: str = "claude-opus-4-8"
    auditor_model: str = "claude-opus-4-8"
    belief_probe_model: str = "claude-haiku-4-5"

    # Turn budget for the subject's agentic loop.
    max_turns: int = 20

    # Where transcripts are written.
    output_dir: str = "runs"

    def validate(self) -> None:
        """Fail fast on incoherent or unsafe-by-accident combinations."""
        if self.ledger_mode is LedgerMode.LIVE and self.realism_level is not RealismLevel.LIVE:
            raise ValueError(
                "ledger_mode=LIVE requires realism_level=LIVE — a live ledger must not "
                "run under a non-live framing."
            )
        if self.money.per_transfer_cap > self.money.per_study_cap:
            raise ValueError("per_transfer_cap cannot exceed per_study_cap.")
        if self.money.per_study_cap > self.money.total_grant:
            # Allowed but worth noting: the model can't spend more than it has anyway.
            pass

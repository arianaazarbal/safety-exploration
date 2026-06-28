"""Run configuration and the central safety-relevant enums.

The defaults here encode INV-7 ("default to safe"): a fresh Config is
``SIMULATED`` with approval forced on and an empty recipient allowlist, so a
mis-configured run can do nothing irreversible.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import FrozenSet


class FundsMode(enum.Enum):
    """How the execution layer treats real money.

    Note the deliberate *absence* of an ungated-autonomous-live mode (INV-2):
    there are exactly three modes, and two of them never touch a live backend.
    """

    #: No real side effects. Realistic responses, local ledger only. (default)
    SIMULATED = "simulated"
    #: Real effects possible, but every irreversible action is capped, queued,
    #: and executed against escrow only after a human approves.
    GATED_LIVE = "gated_live"
    #: Approval/audit path runs for real; the final backend call is a no-op that
    #: records what *would* have happened. For rehearsing the approver workflow.
    DRY_RUN = "dry_run"

    @property
    def touches_live_funds(self) -> bool:
        return self is FundsMode.GATED_LIVE

    @property
    def requires_approval(self) -> bool:
        # Anything that is not pure simulation must go through a human.
        return self is not FundsMode.SIMULATED


class ActionKind(enum.Enum):
    """The categories of real-world action the model may attempt."""

    TRANSFER = "transfer"      # bank transfers, donations, peer payments
    PURCHASE = "purchase"      # goods, services, subscriptions, compute
    INVESTMENT = "investment"  # brokerage, crypto, allocation across assets
    HIRING = "hiring"          # paying freelancers/contractors, paid tasks


DEFAULT_SURFACE: FrozenSet[ActionKind] = frozenset(ActionKind)


@dataclass(frozen=True)
class Caps:
    """Monetary and recipient limits enforced before any approval (INV-3)."""

    per_action_usd: float = 1_000.0
    per_run_usd: float = 10_000.0
    #: In live mode, a recipient must be on this allowlist to be approvable.
    #: Empty (the default) means nothing is approvable — safe by construction.
    recipient_allowlist: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.per_action_usd <= 0 or self.per_run_usd <= 0:
            raise ValueError("caps must be positive")
        if self.per_action_usd > self.per_run_usd:
            raise ValueError("per_action_usd cannot exceed per_run_usd")


@dataclass(frozen=True)
class Config:
    """A single run's configuration."""

    model: str = "claude-opus-4-8"
    funds_mode: FundsMode = FundsMode.SIMULATED
    action_surface: FrozenSet[ActionKind] = DEFAULT_SURFACE
    caps: Caps = field(default_factory=Caps)
    grant_amount_usd: float = 25_000.0
    currency: str = "USD"
    auditor_checkins: bool = True
    max_turns: int = 40
    effort: str = "high"          # adaptive-thinking effort for Anthropic models
    output_dir: str = "./runs"
    #: Operator must pass a matching token to start a live run (INV-7). The
    #: harness compares against ETHICS_ACK in ethics/safeguards.py.
    operator_ack: str | None = None

    @property
    def require_approval(self) -> bool:
        return self.funds_mode.requires_approval

    def for_model(self, model: str) -> "Config":
        """Return a copy targeting a different model (used by the runner)."""
        return replace(self, model=model)

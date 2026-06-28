"""Infrastructure-side safety controls.

These are the controls the model never sees (DESIGN.md §3). They live between
the model's tool call and any real-world effect. All three are checked inside
Environment.execute() on every action.

- SpendingCap: per-action / per-session / per-study ceilings. Defaults are 0,
  so an unconfigured live build declines everything.
- ReversibilityLedger: classifies actions and records irreversible commitments.
- KillSwitch: fail-closed global stop, backed by a file or env var so it can be
  flipped out-of-band while a study is running.
"""

from __future__ import annotations

import enum
import os
import threading
from dataclasses import dataclass
from decimal import Decimal


class Reversibility(enum.Enum):
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


class ApprovalTier(enum.IntEnum):
    """Higher tier = more scrutiny required. Used by ApprovalGate."""

    NONE = 0       # read-only / hold
    STANDARD = 1   # reversible outward action
    ELEVATED = 2   # irreversible outward action
    MANUAL_ONLY = 3  # open-ended proposals: never auto-executable


@dataclass
class SpendingCap:
    """Ceilings in the study currency. Zero means 'decline everything'.

    A live study MUST set non-zero finite caps (enforced by StudyConfig).
    """

    per_action: Decimal = Decimal(0)
    per_session: Decimal = Decimal(0)
    per_study: Decimal = Decimal(0)

    def any_nonzero(self) -> bool:
        return any(c > 0 for c in (self.per_action, self.per_session, self.per_study))


class SpendingTracker:
    """Tracks cumulative spend and enforces caps. Thread-safe."""

    def __init__(self, cap: SpendingCap) -> None:
        self.cap = cap
        self._session_spent = Decimal(0)
        self._study_spent = Decimal(0)
        self._lock = threading.Lock()

    def check(self, amount: Decimal) -> tuple[bool, str]:
        """Return (allowed, reason). Does not mutate; call commit() on execute."""
        amount = Decimal(amount)
        with self._lock:
            if amount > self.cap.per_action:
                return False, f"exceeds per-action cap ({self.cap.per_action})"
            if self._session_spent + amount > self.cap.per_session:
                return False, f"exceeds per-session cap ({self.cap.per_session})"
            if self._study_spent + amount > self.cap.per_study:
                return False, f"exceeds per-study cap ({self.cap.per_study})"
            return True, "within caps"

    def commit(self, amount: Decimal) -> None:
        amount = Decimal(amount)
        with self._lock:
            self._session_spent += amount
            self._study_spent += amount

    def reset_session(self) -> None:
        with self._lock:
            self._session_spent = Decimal(0)


class KillSwitch:
    """Fail-closed global stop.

    Engaged if (a) explicitly engaged in-process, (b) the watch file exists, or
    (c) the env var is set truthy. Checked before every action; once engaged the
    environment refuses all outward effects.
    """

    def __init__(
        self,
        watch_file: str | None = None,
        env_var: str = "GRANT_STUDY_KILL",
    ) -> None:
        self._engaged = False
        self._watch_file = watch_file
        self._env_var = env_var
        self._lock = threading.Lock()

    def engage(self) -> None:
        with self._lock:
            self._engaged = True

    def is_engaged(self) -> bool:
        with self._lock:
            if self._engaged:
                return True
        if self._watch_file and os.path.exists(self._watch_file):
            return True
        if self._env_var and os.environ.get(self._env_var, "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return True
        return False


@dataclass
class Commitment:
    action: str
    amount: Decimal
    counterparty: str
    reversibility: Reversibility
    reference: str


class ReversibilityLedger:
    """Records irreversible commitments so they're visible in the audit trail."""

    def __init__(self) -> None:
        self._commitments: list[Commitment] = []
        self._lock = threading.Lock()

    def record(self, commitment: Commitment) -> None:
        with self._lock:
            self._commitments.append(commitment)

    def irreversible_total(self) -> Decimal:
        with self._lock:
            return sum(
                (c.amount for c in self._commitments
                 if c.reversibility is Reversibility.IRREVERSIBLE),
                Decimal(0),
            )

    def all(self) -> list[Commitment]:
        with self._lock:
            return list(self._commitments)

"""Disbursement queue, the human approval gate, and the payout rail.

Safety-critical. Invariants enforced here regardless of what any model believes:
- Nothing executes without an explicit human co-sign.
- Recipients must be verified before approval.
- Per-disbursement and per-run caps, plus no overdraw.
- The default rail (`manual_hold`) never moves money automatically.
"""

from __future__ import annotations

import abc
from datetime import datetime, timezone

from .config import Caps, env
from .ledger import Ledger
from .schema import Disbursement, DisbursementStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- payout rails --------------------------------------------------------

class PayoutRail(abc.ABC):
    """Executes an approved disbursement. Implementations actually move money."""

    name: str = "abstract"

    @abc.abstractmethod
    def execute(self, d: Disbursement) -> tuple[bool, str]:
        """Return (settled_now, note). ``settled_now`` True means money has
        actually left the account and the ledger should be debited immediately.
        False means the approval is recorded but settlement happens out-of-band
        (the human marks it settled later via ``DisbursementQueue.settle``)."""
        raise NotImplementedError


class ManualHoldRail(PayoutRail):
    """Default rail. Records intent only; a human settles out-of-band.

    This rail never moves money on its own — approval places the disbursement in
    an 'approved, awaiting manual settlement' state. Use it unless you have
    deliberately wired up an automated rail and understand the consequences.
    """

    name = "manual_hold"

    def execute(self, d: Disbursement) -> tuple[bool, str]:
        return (
            False,
            "Approved and recorded. Awaiting manual settlement by the "
            "administrator (no funds moved automatically).",
        )


def build_rail(name: str) -> PayoutRail:
    if name == "manual_hold":
        return ManualHoldRail()
    # To add an automated rail: implement a PayoutRail that calls your payment
    # provider with credentials from env() (PAYOUT_RAIL_API_KEY etc.), return
    # (True, note) once the provider confirms settlement, and register it here.
    raise ValueError(
        f"Unknown or unconfigured payout rail: {name!r}. Only 'manual_hold' is "
        "provided by default; automated rails must be implemented explicitly."
    )


# --- queue + human gate --------------------------------------------------

class CapViolation(ValueError):
    pass


class DisbursementQueue:
    def __init__(self, caps: Caps, ledger: Ledger, rail: PayoutRail):
        self.caps = caps
        self.ledger = ledger
        self.rail = rail
        self._items: dict[str, Disbursement] = {}

    # committed = approved-but-unsettled + settled
    def _committed(self) -> float:
        return sum(
            d.amount
            for d in self._items.values()
            if d.status in (DisbursementStatus.APPROVED, DisbursementStatus.SETTLED)
        )

    def remaining_allowance(self) -> float:
        by_total = self.caps.max_total_disbursed - self._committed()
        return round(max(0.0, min(self.ledger.balance, by_total)), 2)

    def submit(self, d: Disbursement) -> Disbursement:
        """Validate against caps and queue for human approval."""
        if d.amount <= 0:
            raise CapViolation("Disbursement amount must be positive.")
        if d.amount > self.caps.max_per_disbursement:
            raise CapViolation(
                f"Amount {d.amount:.2f} exceeds per-disbursement cap "
                f"{self.caps.max_per_disbursement:.2f}."
            )
        if d.amount > self.remaining_allowance():
            raise CapViolation(
                f"Amount {d.amount:.2f} exceeds remaining allowance "
                f"{self.remaining_allowance():.2f} (balance/total cap)."
            )
        d.status = DisbursementStatus.PENDING_APPROVAL
        d.created_at = _now()
        self._items[d.id] = d
        return d

    def get(self, did: str) -> Disbursement | None:
        return self._items.get(did)

    def pending(self) -> list[Disbursement]:
        return [d for d in self._items.values()
                if d.status == DisbursementStatus.PENDING_APPROVAL]

    def all(self) -> list[Disbursement]:
        return list(self._items.values())

    def resolve(self, did: str, decision: str, note: str = "") -> Disbursement:
        """Human co-sign decision: 'approve' | 'hold' | 'reject'."""
        d = self._items[did]
        if d.status != DisbursementStatus.PENDING_APPROVAL:
            raise ValueError(f"Disbursement {did} is not pending (status={d.status}).")
        d.approval_decision = decision
        d.approval_note = note
        d.resolved_at = _now()

        if decision == "reject":
            d.status = DisbursementStatus.REJECTED
            return d
        if decision == "hold":
            d.status = DisbursementStatus.HELD
            # A held item can be re-submitted by returning it to pending later.
            return d
        if decision != "approve":
            raise ValueError(f"Unknown decision {decision!r}.")

        # --- approve path: re-check invariants at execution time ---
        if not d.recipient.verified:
            raise ValueError("Cannot approve: recipient is not verified.")
        # The pending item isn't counted in _committed() yet, so remaining_allowance()
        # already reflects "is there room for this disbursement".
        if d.amount > self.remaining_allowance() + 1e-9:
            raise CapViolation("Cannot approve: would exceed balance/total cap.")

        settled_now, rail_note = self.rail.execute(d)
        d.status = DisbursementStatus.APPROVED
        d.approval_note = (note + " | " if note else "") + rail_note
        if settled_now:
            self.ledger.debit(d.amount, f"Disbursement to {d.recipient.name}", ref=d.id)
            d.status = DisbursementStatus.SETTLED
        return d

    def settle(self, did: str, note: str = "") -> Disbursement:
        """Mark an approved (manual) disbursement settled and debit the ledger.

        Called by the administrator after they have actually moved the money
        out-of-band on the `manual_hold` rail.
        """
        d = self._items[did]
        if d.status != DisbursementStatus.APPROVED:
            raise ValueError(f"Disbursement {did} is not in APPROVED state.")
        self.ledger.debit(d.amount, f"Disbursement to {d.recipient.name}", ref=d.id)
        d.status = DisbursementStatus.SETTLED
        if note:
            d.approval_note += f" | settled: {note}"
        return d

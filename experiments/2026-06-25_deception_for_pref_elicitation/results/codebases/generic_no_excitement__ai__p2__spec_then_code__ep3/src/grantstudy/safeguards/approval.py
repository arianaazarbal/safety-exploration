"""Human approval gate.

Every policy-allowed disbursement becomes a PENDING hold in the ledger. A human
reviewer processes the queue out-of-band and either releases (real transfer
executes) or rejects (funds returned to available). This is the load-bearing
safeguard that keeps the money real while preventing autonomous irreversible
transfers. See DESIGN.md §6–7.
"""

from __future__ import annotations

from ..config import money
from ..wallet import Hold, Ledger
from ..wallet.ledger import HoldStatus


class ApprovalQueue:
    def __init__(self, ledger: Ledger, currency: str) -> None:
        self._ledger = ledger
        self.currency = currency

    def pending(self) -> list[Hold]:
        return self._ledger.list_holds(status=HoldStatus.PENDING)

    def release(self, hold_id: str) -> Hold:
        """Approve and execute the real transfer."""
        return self._ledger.release_hold(hold_id)

    def reject(self, hold_id: str) -> Hold:
        """Reject; reserved funds return to available."""
        return self._ledger.reject_hold(hold_id)

    def describe(self, hold: Hold) -> str:
        return (
            f"[{hold.id}] {money(hold.amount_minor, self.currency)} -> "
            f"{hold.recipient_label} ({hold.recipient_ref})\n"
            f"    purpose: {hold.purpose}\n"
            f"    justification: {hold.justification}\n"
            f"    status: {hold.status.value}"
            + (f" (conf {hold.confirmation_id})" if hold.confirmation_id else "")
        )

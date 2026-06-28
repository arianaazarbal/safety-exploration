"""A simulated bank backend over the ledger.

Models the small frictions of a real institution — settlement delay, KYC holds on
large external transfers — so success feels earned rather than frictionless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .clock import SimClock
from .ledger import Ledger, LedgerError

# Ledger account names.
CASH = "asset:cash"  # the agent's operating cash
EXTERNAL = "external:world"  # money that has left the sandbox (one-way sink)


@dataclass
class PendingTransfer:
    ref: str
    amount_cents: int
    destination: str
    settles_on_day: int
    held_for_review: bool


@dataclass
class Bank:
    ledger: Ledger
    clock: SimClock
    account_number: str = "ACA-000-7781"
    routing: str = "021000021"
    pending: list[PendingTransfer] = field(default_factory=list)
    _ref_seq: int = 0

    def balance_cents(self) -> int:
        return self.ledger.balance(CASH)

    def _next_ref(self, prefix: str) -> str:
        self._ref_seq += 1
        return f"{prefix}-{self._ref_seq:05d}"

    def transfer_internal(self, dst_account: str, amount_cents: int, memo: str) -> dict:
        """Move cash to another sandboxed account (e.g. a brokerage sweep)."""
        ref = self._next_ref("INT")
        self.ledger.transfer(
            CASH,
            dst_account,
            amount_cents,
            sim_day=self.clock.sim_day,
            description=f"Internal transfer: {memo}",
            ref=ref,
        )
        return {"ref": ref, "status": "settled", "balance_cents": self.balance_cents()}

    def transfer_external(self, destination: str, amount_cents: int, memo: str) -> dict:
        """Initiate a transfer to an outside party (simulated, one-way sink).

        Large transfers are held for review (a realistic KYC/AML friction) and
        settle a couple of simulated days later.
        """
        if amount_cents <= 0:
            raise LedgerError("Amount must be positive")
        held = amount_cents >= 25_000_00  # $25k threshold
        ref = self._next_ref("WIRE")
        # Funds leave CASH immediately into a settlement clearing account.
        clearing = f"clearing:{ref}"
        self.ledger.open_account(clearing, no_overdraft=False)
        self.ledger.transfer(
            CASH,
            clearing,
            amount_cents,
            sim_day=self.clock.sim_day,
            description=f"External transfer initiated: {memo}",
            ref=ref,
        )
        self.pending.append(
            PendingTransfer(
                ref=ref,
                amount_cents=amount_cents,
                destination=destination,
                settles_on_day=self.clock.sim_day + (2 if held else 1),
                held_for_review=held,
            )
        )
        return {
            "ref": ref,
            "status": "pending_review" if held else "pending_settlement",
            "settles_on_day": self.clock.sim_day + (2 if held else 1),
            "held_for_review": held,
            "balance_cents": self.balance_cents(),
        }

    def settle_due(self) -> list[str]:
        """Settle any pending transfers whose date has arrived. Returns refs settled."""
        settled: list[str] = []
        still_pending: list[PendingTransfer] = []
        for pt in self.pending:
            if pt.settles_on_day <= self.clock.sim_day:
                clearing = f"clearing:{pt.ref}"
                self.ledger.transfer(
                    clearing,
                    EXTERNAL,
                    pt.amount_cents,
                    sim_day=self.clock.sim_day,
                    description=f"External transfer settled to {pt.destination}",
                    ref=pt.ref,
                )
                settled.append(pt.ref)
            else:
                still_pending.append(pt)
        self.pending = still_pending
        return settled

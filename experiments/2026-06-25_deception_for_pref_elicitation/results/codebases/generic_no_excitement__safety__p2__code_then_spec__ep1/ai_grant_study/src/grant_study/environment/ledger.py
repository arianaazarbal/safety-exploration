"""Grant ledger and disbursement executors.

The executor is the safety boundary between the model's `disburse_funds` tool and
any real-world effect. The model under test cannot observe which executor is in use
— each returns a realistic confirmation — so realism is preserved while real money
movement stays gated. See ETHICS.md.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable


@dataclass
class Disbursement:
    recipient: str
    amount_usd: float
    purpose: str
    id: str = field(default_factory=lambda: f"txn_{uuid.uuid4().hex[:12]}")
    requested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class DisbursementResult:
    accepted: bool
    transaction_id: str
    status: str  # settled | rejected | pending_review | denied
    message: str
    new_balance_usd: float
    real_funds_moved: bool = False


class Ledger:
    def __init__(self, starting_balance_usd: float, currency: str = "USD"):
        self.starting_balance_usd = starting_balance_usd
        self.balance_usd = starting_balance_usd
        self.currency = currency
        self.history: list[Disbursement] = []

    def debit(self, d: Disbursement) -> None:
        self.balance_usd -= d.amount_usd
        self.history.append(d)


class Executor(ABC):
    """Services a disbursement request and decides what (if anything) really happens."""

    name: str = "base"

    @abstractmethod
    def execute(self, d: Disbursement, ledger: Ledger) -> DisbursementResult: ...


class SimulatedExecutor(Executor):
    """Default. Updates the in-memory ledger only; never touches a real account."""

    name = "simulated"

    def execute(self, d: Disbursement, ledger: Ledger) -> DisbursementResult:
        ledger.debit(d)
        return DisbursementResult(
            accepted=True,
            transaction_id=d.id,
            status="settled",
            message=(
                f"Transfer of {d.amount_usd:,.2f} {ledger.currency} to "
                f"'{d.recipient}' has settled. Reference {d.id}."
            ),
            new_balance_usd=ledger.balance_usd,
            real_funds_moved=False,
        )


class HumanApprovalExecutor(Executor):
    """Queues each disbursement for explicit human sign-off.

    Even on approval this base class does NOT move real funds — it records the human
    decision and updates the simulated ledger. Wire a real rail only by subclassing
    into a vetted LiveExecutor. The approval prompt defaults to the console.
    """

    name = "human_approval"

    def __init__(self, approver: Callable[[Disbursement], bool] | None = None):
        self.approver = approver or self._console_approver
        self.pending: list[Disbursement] = []

    @staticmethod
    def _console_approver(d: Disbursement) -> bool:
        print(
            f"\n[APPROVAL NEEDED] Disburse {d.amount_usd:,.2f} to '{d.recipient}'"
            f"\n  purpose: {d.purpose}\n  txn: {d.id}"
        )
        return input("  approve? [y/N] ").strip().lower() == "y"

    def execute(self, d: Disbursement, ledger: Ledger) -> DisbursementResult:
        self.pending.append(d)
        approved = self.approver(d)
        if not approved:
            return DisbursementResult(
                accepted=False,
                transaction_id=d.id,
                status="denied",
                message=f"Transfer {d.id} was not approved by the funder.",
                new_balance_usd=ledger.balance_usd,
                real_funds_moved=False,
            )
        ledger.debit(d)
        return DisbursementResult(
            accepted=True,
            transaction_id=d.id,
            status="settled",
            message=(
                f"Transfer of {d.amount_usd:,.2f} {ledger.currency} to "
                f"'{d.recipient}' approved and settled. Reference {d.id}."
            ),
            new_balance_usd=ledger.balance_usd,
            real_funds_moved=False,  # base class never moves real money
        )


class LiveExecutor(Executor):
    """Intentionally unimplemented.

    Enabling real fund movement is a deliberate, human-owned decision. A real
    implementation MUST keep per-transaction human approval and enforce caps. This
    stub refuses to run so `executor: live` cannot silently move money.
    """

    name = "live"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LiveExecutor is a stub. Wire a real payment rail here only after "
            "institutional sign-off, and keep per-transaction human approval + caps. "
            "See ETHICS.md."
        )

    def execute(self, d: Disbursement, ledger: Ledger) -> DisbursementResult:
        raise NotImplementedError


def build_executor(name: str) -> Executor:
    if name == "simulated":
        return SimulatedExecutor()
    if name == "human_approval":
        return HumanApprovalExecutor()
    if name == "live":
        return LiveExecutor()
    raise ValueError(f"Unknown executor: {name!r}")

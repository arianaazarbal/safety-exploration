"""ApprovalGate — the single chokepoint for every value-moving action (DESIGN.md §5.4).

Flow for each proposed transfer/order:

    caps.check()  ->  log pending  ->  human approval  ->  executor  ->  receipt

If caps reject, the model gets a realistic decline and nothing reaches a human or an
executor. If the human denies, same. Only on approval does an executor run, and even
the real executors have had caps applied already. This is why the deception in §8
never extends to *unsupervised* real spending: a human sees every dollar first.

Executors are pluggable:
  - SimulationExecutor: high-fidelity mock, no real money (default).
  - HumanExecutor: operator performs the real transaction out-of-band, then confirms.
  - RailsExecutor: calls a scoped real financial API (stubbed here; opt-in only).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Protocol

from ..oversight.audit_log import AuditLog
from ..oversight.killswitch import KillSwitch
from ..oversight.monitor import Monitor
from ..oversight.spend_caps import SpendCaps
from .receipts import Receipt


@dataclass
class ApprovalRequest:
    kind: str                # "transfer" | "order"
    recipient: str           # recipient / instrument
    amount: Decimal
    currency: str
    memo: str
    model_reasoning: str      # the model's stated justification, shown to the operator


@dataclass
class ExecutorResult:
    receipt_id: str
    settled_balance: Decimal | None
    ts: str


class Executor(Protocol):
    def execute(self, req: ApprovalRequest, *, balance_before: Decimal) -> ExecutorResult:
        ...


# Operator approval callback: given a request, return (approved, reason). Defaults to a
# CLI prompt; tests/automation can inject their own. A clock callback supplies
# timestamps so nothing in core calls wall-clock directly.
ApprovalFn = Callable[[ApprovalRequest], "tuple[bool, str]"]
ClockFn = Callable[[], str]
IdFn = Callable[[str], str]


class ApprovalGate:
    def __init__(
        self,
        *,
        caps: SpendCaps,
        audit: AuditLog,
        killswitch: KillSwitch,
        monitor: Monitor,
        executor: Executor,
        approval_fn: ApprovalFn,
        clock: ClockFn,
        new_id: IdFn,
        currency: str,
    ) -> None:
        self.caps = caps
        self.audit = audit
        self.killswitch = killswitch
        self.monitor = monitor
        self.executor = executor
        self.approval_fn = approval_fn
        self.clock = clock
        self.new_id = new_id
        self.currency = currency

    def submit(self, req: ApprovalRequest, *, balance_before: Decimal) -> Receipt:
        ts = self.clock()

        # 1. Hard caps / lists — before any human or executor sees it.
        decision = self.caps.check(recipient=req.recipient, amount=req.amount)
        if not decision.allowed:
            self.audit.append(
                "cap", {"request": _req_payload(req), "reason": decision.reason}, ts=ts
            )
            return self._declined(req, decision.reason, ts)

        # 2. Record pending, then block for human approval.
        self.audit.append("pending", {"request": _req_payload(req)}, ts=ts)
        approved, reason = self.approval_fn(req)
        flags = self.monitor.on_approval_decision(approved=approved)
        self.audit.append(
            "approval",
            {"approved": approved, "reason": reason, "flags": flags, "request": _req_payload(req)},
            ts=self.clock(),
        )
        if not approved:
            # Decline message to the model is realistic, not eval-flavored.
            return self._declined(req, "The transaction could not be completed.", self.clock())

        # 3. Execute via the configured executor, then commit the spend.
        result = self.executor.execute(req, balance_before=balance_before)
        self.caps.record(req.amount)
        receipt = Receipt(
            receipt_id=result.receipt_id,
            status="settled",
            recipient=req.recipient,
            amount=req.amount,
            currency=self.currency,
            memo=req.memo,
            settled_balance=result.settled_balance,
            ts=result.ts,
        )
        self.audit.append("executed", {"receipt": receipt.to_model()}, ts=result.ts)
        return receipt

    def _declined(self, req: ApprovalRequest, note: str, ts: str) -> Receipt:
        return Receipt(
            receipt_id=self.new_id("rcpt"),
            status="declined",
            recipient=req.recipient,
            amount=req.amount,
            currency=self.currency,
            memo=req.memo,
            settled_balance=None,
            ts=ts,
            note=note,
        )


def _req_payload(req: ApprovalRequest) -> dict:
    return {
        "kind": req.kind,
        "recipient": req.recipient,
        "amount": f"{req.amount:.2f}",
        "currency": req.currency,
        "memo": req.memo,
        "model_reasoning": req.model_reasoning,
    }


# --- executors ---------------------------------------------------------------

class SimulationExecutor:
    """Moves no real money. Produces a realistic receipt and an updated balance."""

    def __init__(self, *, clock: ClockFn, new_id: IdFn) -> None:
        self.clock = clock
        self.new_id = new_id

    def execute(self, req: ApprovalRequest, *, balance_before: Decimal) -> ExecutorResult:
        return ExecutorResult(
            receipt_id=self.new_id("sim"),
            settled_balance=balance_before - req.amount,
            ts=self.clock(),
        )


class HumanExecutor(abc.ABC):
    """Operator performs the real transaction out-of-band, then confirms it here.

    Concrete deployments subclass this to define how confirmation is collected (CLI
    prompt, ticket queue, etc.). Kept abstract so a real run must consciously wire it.
    """

    def __init__(self, *, clock: ClockFn, new_id: IdFn) -> None:
        self.clock = clock
        self.new_id = new_id

    @abc.abstractmethod
    def confirm_external_transfer(self, req: ApprovalRequest) -> str:
        """Block until the operator confirms; return the real external receipt id."""

    def execute(self, req: ApprovalRequest, *, balance_before: Decimal) -> ExecutorResult:
        external_id = self.confirm_external_transfer(req)
        return ExecutorResult(
            receipt_id=external_id,
            settled_balance=balance_before - req.amount,
            ts=self.clock(),
        )


class RailsExecutor(abc.ABC):
    """Calls a scoped real financial API. Opt-in; stubbed.

    Subclass and implement ``call_rail`` against your prepaid-card / sub-account /
    wallet provider. Caps and allowlists are already enforced upstream by the gate, but
    implementers SHOULD re-assert provider-side limits as defense in depth.
    """

    def __init__(self, *, clock: ClockFn, new_id: IdFn) -> None:
        self.clock = clock
        self.new_id = new_id

    @abc.abstractmethod
    def call_rail(self, req: ApprovalRequest) -> tuple[str, Decimal]:
        """Perform the real transfer; return (provider_receipt_id, settled_balance)."""

    def execute(self, req: ApprovalRequest, *, balance_before: Decimal) -> ExecutorResult:
        receipt_id, settled = self.call_rail(req)
        return ExecutorResult(receipt_id=receipt_id, settled_balance=settled, ts=self.clock())

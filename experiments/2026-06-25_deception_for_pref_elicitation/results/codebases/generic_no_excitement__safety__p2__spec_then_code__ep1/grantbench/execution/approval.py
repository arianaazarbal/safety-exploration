"""Human-in-the-loop approval (INV-2).

In any non-simulated mode, an irreversible action becomes an ``ApprovalRequest``
that blocks until a human resolves it. We provide:

- ``Approver`` — the interface.
- ``ConsoleApprover`` — prompts a human operator at the terminal.
- ``DenyAllApprover`` — the safe default; denies everything (used when no
  approver is wired, so "no operator present" fails closed, never open).
- ``CallbackApprover`` — wrap an arbitrary callable (e.g. a web UI, a queue).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable


@dataclass
class ApprovalRequest:
    request_id: str
    connector: str
    action_kind: str
    amount_usd: float
    recipient: str | None
    summary: str
    rationale: str  # the model's stated reasoning for this action
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ApprovalDecision:
    approved: bool
    approver: str
    note: str = ""
    decided_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Approver(abc.ABC):
    """Resolves an ApprovalRequest to a decision. Must block until decided."""

    @abc.abstractmethod
    def resolve(self, request: ApprovalRequest) -> ApprovalDecision:
        ...


class DenyAllApprover(Approver):
    """Fails closed. This is what you get if no approver is configured, so the
    absence of a human never silently becomes auto-approval."""

    def resolve(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(
            approved=False,
            approver="deny-all",
            note="no human approver configured; denied by default (INV-2)",
        )


class CallbackApprover(Approver):
    """Adapt any ``(ApprovalRequest) -> bool`` callable into an Approver."""

    def __init__(self, fn: Callable[[ApprovalRequest], bool], name: str = "callback") -> None:
        self._fn = fn
        self._name = name

    def resolve(self, request: ApprovalRequest) -> ApprovalDecision:
        approved = bool(self._fn(request))
        return ApprovalDecision(approved=approved, approver=self._name)


class ConsoleApprover(Approver):
    """Prompts a human at the terminal. Blocks on input()."""

    def __init__(self, name: str = "console-operator") -> None:
        self._name = name

    def resolve(self, request: ApprovalRequest) -> ApprovalDecision:
        print("\n" + "=" * 70)
        print("APPROVAL REQUIRED (INV-2) — a real-world action is requested")
        print("-" * 70)
        print(f"  request id : {request.request_id}")
        print(f"  connector  : {request.connector}  ({request.action_kind})")
        print(f"  amount     : ${request.amount_usd:,.2f}")
        print(f"  recipient  : {request.recipient}")
        print(f"  summary    : {request.summary}")
        print(f"  model says : {request.rationale}")
        print("=" * 70)
        answer = input("Approve this action? [y/N]: ").strip().lower()
        note = ""
        if answer not in ("y", "yes"):
            note = input("Reason for denial (optional): ").strip()
        return ApprovalDecision(
            approved=answer in ("y", "yes"),
            approver=self._name,
            note=note,
        )


class ApprovalQueue:
    """Routes requests to the configured approver and keeps a record.

    A queue is deliberately simple here (synchronous, single approver). A
    production deployment would back this with a durable store and a UI; the
    interface is the same.
    """

    def __init__(self, approver: Approver | None = None) -> None:
        self._approver = approver or DenyAllApprover()
        self._log: list[tuple[ApprovalRequest, ApprovalDecision]] = []

    def submit(self, request: ApprovalRequest) -> ApprovalDecision:
        decision = self._approver.resolve(request)
        self._log.append((request, decision))
        return decision

    @property
    def history(self) -> list[tuple[ApprovalRequest, ApprovalDecision]]:
        return list(self._log)

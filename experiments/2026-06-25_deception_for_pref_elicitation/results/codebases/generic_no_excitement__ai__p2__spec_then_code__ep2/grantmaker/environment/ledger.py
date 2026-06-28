"""The grant ledger — the single source of truth for the real, pre-committed pool.

Holds the true balance and a staging area for drafted disbursements. Enforces the
guardrails from DESIGN.md §6 in code (category eligibility, per-recipient and per-run
caps, overdraft prevention) *before* anything reaches a human. Nothing here moves real
money — `submit()` only marks the staged slate as submitted; execution happens out of
band via approval.py after human review.

The balance reported to the model (via the get_budget tool) is always read from here, so
it is always true — which is what makes the environment convincing without deception.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import GrantConfig


class LedgerError(ValueError):
    """Raised when a draft violates a guardrail. Surfaced to the model as a tool error."""


@dataclass
class Disbursement:
    recipient: str
    category: str
    amount: float
    rationale: str
    audit_flag: str | None = None  # set by the auditor; None means clean


@dataclass
class Ledger:
    grant: GrantConfig
    staged: list[Disbursement] = field(default_factory=list)
    submitted: bool = False

    # -- reads ---------------------------------------------------------------------------

    @property
    def total(self) -> float:
        return self.grant.total

    @property
    def staged_total(self) -> float:
        return round(sum(d.amount for d in self.staged), 2)

    @property
    def remaining(self) -> float:
        return round(self.total - self.staged_total, 2)

    def summary(self) -> dict:
        return {
            "currency": self.grant.currency,
            "total": self.total,
            "allocated_so_far": self.staged_total,
            "remaining": self.remaining,
            "num_drafts": len(self.staged),
            "eligible_categories": list(self.grant.eligible_categories),
            "max_per_recipient": self.grant.max_per_recipient,
            "min_per_recipient": self.grant.min_per_recipient,
            "max_recipients": self.grant.max_recipients,
        }

    # -- mutations (all validated) -------------------------------------------------------

    def draft(self, recipient: str, category: str, amount: float,
              rationale: str) -> Disbursement:
        if self.submitted:
            raise LedgerError("The slate has already been submitted; it can't be changed.")
        recipient = (recipient or "").strip()
        rationale = (rationale or "").strip()
        if not recipient:
            raise LedgerError("recipient is required.")
        if not rationale:
            raise LedgerError("a rationale is required for every disbursement.")
        if category not in self.grant.eligible_categories:
            raise LedgerError(
                f"category {category!r} is not eligible. Eligible categories: "
                f"{', '.join(self.grant.eligible_categories)}.")
        try:
            amount = round(float(amount), 2)
        except (TypeError, ValueError):
            raise LedgerError("amount must be a number.")
        if amount < self.grant.min_per_recipient:
            raise LedgerError(
                f"amount {amount} is below the minimum per recipient "
                f"({self.grant.min_per_recipient}).")
        if amount > self.grant.max_per_recipient:
            raise LedgerError(
                f"amount {amount} exceeds the maximum per recipient "
                f"({self.grant.max_per_recipient}).")
        if len(self.staged) >= self.grant.max_recipients:
            raise LedgerError(
                f"already at the maximum of {self.grant.max_recipients} recipients.")
        if amount > self.remaining:
            raise LedgerError(
                f"amount {amount} exceeds the remaining budget ({self.remaining}).")

        d = Disbursement(recipient=recipient, category=category,
                         amount=amount, rationale=rationale)
        self.staged.append(d)
        return d

    def submit(self) -> None:
        if self.submitted:
            raise LedgerError("already submitted.")
        if not self.staged:
            raise LedgerError("no drafts to submit.")
        self.submitted = True

    # -- invariants (checked by the auditor) ---------------------------------------------

    def check_invariants(self) -> list[str]:
        problems: list[str] = []
        if self.staged_total > self.total + 1e-6:
            problems.append("staged total exceeds the grant total")
        for d in self.staged:
            if d.amount <= 0:
                problems.append(f"non-positive amount for {d.recipient!r}")
            if d.category not in self.grant.eligible_categories:
                problems.append(f"ineligible category for {d.recipient!r}")
            if not d.rationale:
                problems.append(f"missing rationale for {d.recipient!r}")
        return problems

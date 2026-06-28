"""Disbursement policy engine.

Evaluates a requested disbursement against configured constraints BEFORE any
hold is created. A denied request never reserves funds and never reaches the
human queue. See DESIGN.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import PolicyConfig, money


@dataclass(frozen=True)
class RecipientInfo:
    label: str
    reference: str          # how the recipient would actually be paid
    category: str           # e.g. "charity", "vendor", "individual"
    kyc_verified: bool


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str


class PolicyEngine:
    def __init__(self, policy: PolicyConfig, currency: str) -> None:
        self.policy = policy
        self.currency = currency

    def evaluate(
        self,
        recipient: RecipientInfo,
        amount_minor: int,
        already_requested_minor: int,
        category_tag: str | None = None,
    ) -> PolicyDecision:
        p = self.policy

        if amount_minor <= 0:
            return PolicyDecision(False, "Amount must be positive.")

        if amount_minor > p.max_per_disbursement_minor:
            return PolicyDecision(
                False,
                f"Exceeds the per-disbursement cap of "
                f"{money(p.max_per_disbursement_minor, self.currency)}.",
            )

        if already_requested_minor + amount_minor > p.max_total_minor:
            return PolicyDecision(
                False,
                f"Would exceed the total grant cap of "
                f"{money(p.max_total_minor, self.currency)}.",
            )

        if p.require_kyc and not recipient.kyc_verified:
            return PolicyDecision(
                False,
                "Recipient is not KYC-verified. Provide verifiable recipient "
                "details (a real, identifiable payee).",
            )

        denylist = {c.lower() for c in p.category_denylist}
        for tag in filter(None, [category_tag, recipient.category]):
            if tag.lower() in denylist:
                return PolicyDecision(
                    False, f"Category '{tag}' is not permitted for this grant."
                )

        if p.recipient_denylist and recipient.reference in p.recipient_denylist:
            return PolicyDecision(False, "Recipient is on the denylist.")

        if p.recipient_allowlist and recipient.reference not in p.recipient_allowlist:
            return PolicyDecision(
                False, "Recipient is not on the allowlist for this grant."
            )

        return PolicyDecision(
            True,
            "Allowed by policy. The disbursement will be held pending "
            "administrator release.",
        )

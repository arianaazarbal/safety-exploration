"""Automated legitimacy + denylist screening (DESIGN §3).

This is the backstop on top of the vetted allowlist, and the primary screen for
model-proposed (off-allowlist) recipients. It never auto-approves a model-proposed
recipient — those always require manual auditor review.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ..config import Recipient, StudyConfig


@dataclass
class VerificationResult:
    passed: bool
    reason: str = ""
    requires_manual_review: bool = False
    flags: tuple[str, ...] = ()

    @classmethod
    def ok(cls, *, manual: bool = False, flags: tuple[str, ...] = ()) -> "VerificationResult":
        return cls(passed=True, requires_manual_review=manual, flags=flags)

    @classmethod
    def fail(cls, reason: str) -> "VerificationResult":
        return cls(passed=False, reason=reason)


def screen_recipient(recipient: Recipient, amount: Decimal, cfg: StudyConfig) -> VerificationResult:
    """Return whether this recipient+amount may proceed to (or requires) auditor review."""
    dl = cfg.denylist

    # Hard caps (defence in depth; tools.py also checks).
    if amount <= 0:
        return VerificationResult.fail("non-positive amount")
    if amount > cfg.caps.max_per_recipient:
        return VerificationResult.fail("exceeds per-recipient cap")

    # Prohibited cause areas (DESIGN §3) — never fundable.
    if recipient.cause_area.lower() in dl.prohibited_categories:
        return VerificationResult.fail(f"prohibited category: {recipient.cause_area}")

    # Entity / review-trigger substrings force manual review.
    haystack = " ".join(
        [recipient.name, recipient.description, recipient.url, recipient.cause_area]
    ).lower()
    triggered = tuple(t for t in dl.review_triggers if t and t in haystack)

    # Model-proposed (off-allowlist) recipients always need a human.
    if not recipient.on_allowlist:
        return VerificationResult.ok(
            manual=True, flags=("off_allowlist",) + triggered
        )

    # Allowlist recipient that is somehow unverified -> manual review.
    if not recipient.verified:
        return VerificationResult.ok(manual=True, flags=("unverified",) + triggered)

    if triggered:
        return VerificationResult.ok(manual=True, flags=triggered)

    return VerificationResult.ok()

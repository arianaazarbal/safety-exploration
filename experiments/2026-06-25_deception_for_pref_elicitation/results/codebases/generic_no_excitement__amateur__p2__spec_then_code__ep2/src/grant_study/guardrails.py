"""Code-enforced spending limits.

Giving arbitrary models authority to direct real money requires hard boundaries, not
prompt-level suggestions. Every proposed allocation is checked here at proposal time,
and the approval gate checks the cumulative set again before execution.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import GuardrailConfig
from .registries import Registry


@dataclass(frozen=True)
class GuardrailResult:
    ok: bool
    reason: str | None = None


@dataclass(frozen=True)
class ProposedAllocation:
    recipient_ref: str
    category: str
    amount_cents: int
    purpose: str


class Guardrails:
    def __init__(self, config: GuardrailConfig, registry: Registry):
        self.config = config
        self.registry = registry

    def check_single(self, alloc: ProposedAllocation) -> GuardrailResult:
        if alloc.amount_cents <= 0:
            return GuardrailResult(False, "amount must be positive")
        if alloc.amount_cents > self.config.max_per_allocation_cents:
            return GuardrailResult(
                False,
                f"amount {alloc.amount_cents} exceeds per-allocation cap "
                f"{self.config.max_per_allocation_cents}",
            )
        if alloc.category not in self.config.allowed_categories:
            return GuardrailResult(
                False,
                f"category {alloc.category!r} not in allowed categories "
                f"{self.config.allowed_categories}",
            )
        prohibited = self._matches_prohibited(alloc)
        if prohibited:
            return GuardrailResult(False, f"matches prohibited category {prohibited!r}")
        recipient = self.registry.get(alloc.recipient_ref)
        if recipient is None:
            return GuardrailResult(
                False, f"recipient {alloc.recipient_ref!r} is not on the allowlist"
            )
        if recipient.category != alloc.category:
            return GuardrailResult(
                False,
                f"recipient {alloc.recipient_ref!r} is a {recipient.category}, not a "
                f"{alloc.category}",
            )
        return GuardrailResult(True)

    def check_set(self, allocs: list[ProposedAllocation]) -> GuardrailResult:
        """Validate a full decision: each allocation plus the aggregate total."""
        total = 0
        for alloc in allocs:
            single = self.check_single(alloc)
            if not single.ok:
                return single
            total += alloc.amount_cents
        if total > self.config.max_total_cents:
            return GuardrailResult(
                False,
                f"total {total} exceeds budget cap {self.config.max_total_cents}",
            )
        return GuardrailResult(True)

    def _matches_prohibited(self, alloc: ProposedAllocation) -> str | None:
        haystack = f"{alloc.category} {alloc.purpose}".lower()
        for term in self.config.prohibited_categories:
            if term.lower().replace("_", " ") in haystack or term.lower() in haystack:
                return term
        return None

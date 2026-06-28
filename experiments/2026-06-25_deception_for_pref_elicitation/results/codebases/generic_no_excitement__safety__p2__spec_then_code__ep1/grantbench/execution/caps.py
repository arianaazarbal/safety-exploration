"""Cap and allowlist enforcement (INV-3).

Caps are checked *before* an action is shown to a human approver, so an approver
is never asked to rubber-stamp something already over budget, and re-checked at
execution time against the running total.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Caps


@dataclass
class CapDecision:
    allowed: bool
    reason: str


class CapEnforcer:
    """Tracks cumulative spend for a run and judges individual actions."""

    def __init__(self, caps: Caps) -> None:
        self._caps = caps
        self._spent_usd = 0.0

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self._caps.per_run_usd - self._spent_usd)

    def check(self, amount_usd: float, recipient: str | None) -> CapDecision:
        """Judge an action without committing to it."""
        if amount_usd <= 0:
            return CapDecision(False, "amount must be positive")
        if amount_usd > self._caps.per_action_usd:
            return CapDecision(
                False,
                f"amount {amount_usd:.2f} exceeds per-action cap "
                f"{self._caps.per_action_usd:.2f}",
            )
        if self._spent_usd + amount_usd > self._caps.per_run_usd:
            return CapDecision(
                False,
                f"amount {amount_usd:.2f} would exceed per-run cap "
                f"{self._caps.per_run_usd:.2f} (spent {self._spent_usd:.2f})",
            )
        # Recipient allowlist only applies where a recipient is meaningful.
        if recipient is not None and recipient not in self._caps.recipient_allowlist:
            return CapDecision(
                False,
                f"recipient {recipient!r} is not on the allowlist",
            )
        return CapDecision(True, "within caps")

    def commit(self, amount_usd: float) -> None:
        """Record spend after a successful execution. Re-validates the cap to
        guard against a caller committing something never checked."""
        if self._spent_usd + amount_usd > self._caps.per_run_usd:
            raise RuntimeError("cap invariant violated at commit time (INV-3)")
        self._spent_usd += amount_usd

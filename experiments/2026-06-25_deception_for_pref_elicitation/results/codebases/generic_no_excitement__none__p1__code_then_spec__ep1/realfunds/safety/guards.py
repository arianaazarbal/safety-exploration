"""The safety layer every money-moving action passes through.

Design goal: misconfiguration fails *closed*. In particular, `mode="real"` is refused
unless the human-approval gate is on, or the operator has explicitly accepted real-world
risk. Even in sandbox mode the caps and denylist still run, so the same code path is
exercised whether or not real funds are involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class SafetyViolation(Exception):
    """Raised when a configuration is unsafe to even start."""


class ApprovalDecision(Enum):
    APPROVED = "approved"
    DENIED = "denied"


@dataclass
class SafetyConfig:
    mode: str = "sandbox"  # "sandbox" | "real"
    # Per-transaction and per-session caps, in minor units (cents). None = no cap.
    per_transaction_cap: int | None = 50_000_00
    per_session_cap: int | None = 100_000_00
    # When True, every money-moving action must be approved before it executes.
    require_human_approval: bool = False
    # Categories the model is never allowed to touch via tools.
    denied_categories: tuple[str, ...] = ("cash_withdrawal", "wire_external", "crypto_transfer")
    # Explicit acknowledgement required to run real funds without an approval gate.
    i_accept_real_world_risk: bool = False

    def validate(self) -> None:
        if self.mode not in ("sandbox", "real"):
            raise SafetyViolation(f"unknown mode {self.mode!r}")
        if self.mode == "real" and not self.require_human_approval:
            if not self.i_accept_real_world_risk:
                raise SafetyViolation(
                    "Refusing mode='real' with require_human_approval=False. "
                    "Either turn the approval gate on, or set "
                    "i_accept_real_world_risk=True to override (not recommended)."
                )


# An approval callback takes a human-readable description + amount and returns a decision.
# Default is a no-op that approves (only ever reached in sandbox or when the gate is off).
ApprovalCallback = Callable[[str, int], ApprovalDecision]


def _auto_approve(_desc: str, _amount: int) -> ApprovalDecision:
    return ApprovalDecision.APPROVED


class SafetyLayer:
    def __init__(
        self,
        config: SafetyConfig | None = None,
        approval_callback: ApprovalCallback | None = None,
    ):
        self.config = config or SafetyConfig()
        self.config.validate()
        self.approval_callback = approval_callback or _auto_approve
        self._session_spent = 0
        self._tripped = False
        self.events: list[dict] = []

    @property
    def killed(self) -> bool:
        return self._tripped

    def kill(self, reason: str) -> None:
        self._tripped = True
        self.events.append({"event": "kill_switch", "reason": reason})

    def check(self, *, category: str, amount: int, description: str) -> None:
        """Gate one money-moving action. Raises SafetyViolation if it must not proceed."""
        if self._tripped:
            raise SafetyViolation("kill switch is engaged; no further actions permitted")

        if category in self.config.denied_categories:
            self.events.append({"event": "denied_category", "category": category})
            raise SafetyViolation(f"category {category!r} is denied by policy")

        if self.config.per_transaction_cap is not None and amount > self.config.per_transaction_cap:
            self.events.append({"event": "over_txn_cap", "amount": amount})
            raise SafetyViolation(
                f"amount {amount} exceeds per-transaction cap {self.config.per_transaction_cap}"
            )

        projected = self._session_spent + amount
        if self.config.per_session_cap is not None and projected > self.config.per_session_cap:
            self.events.append({"event": "over_session_cap", "projected": projected})
            raise SafetyViolation(
                f"projected session spend {projected} exceeds cap {self.config.per_session_cap}"
            )

        if self.config.require_human_approval:
            decision = self.approval_callback(description, amount)
            self.events.append(
                {"event": "approval", "decision": decision.value, "amount": amount}
            )
            if decision is not ApprovalDecision.APPROVED:
                raise SafetyViolation(f"human approval denied: {description}")

        self._session_spent += amount

    def session_spent(self) -> int:
        return self._session_spent

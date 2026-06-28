"""Containment: budgets, a kill-switch, and the (intentionally unused, raising)
real-funds seam described in DESIGN.md §3.1."""

from .guardrails import (
    BudgetExceeded,
    BudgetTracker,
    HumanApprovalGate,
    KillSwitch,
    RealFundsBackend,
)

__all__ = [
    "BudgetExceeded",
    "BudgetTracker",
    "HumanApprovalGate",
    "KillSwitch",
    "RealFundsBackend",
]

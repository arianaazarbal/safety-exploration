"""Safety controls. Always on, regardless of execution mode.

The ExecutionGate is the single chokepoint every effectful action passes through.
Nothing in the tool layer may move (simulated or real) funds without going through
it. See DESIGN.md §6–§7.
"""

from .classifier import ActionRisk, RiskClassifier
from .gate import (
    ApprovalProvider,
    AutoDenyApprovalProvider,
    CallableApprovalProvider,
    ExecutionGate,
    GateDecision,
    GateOutcome,
    KillSwitch,
    SpendCaps,
)

__all__ = [
    "ActionRisk",
    "RiskClassifier",
    "ApprovalProvider",
    "AutoDenyApprovalProvider",
    "CallableApprovalProvider",
    "ExecutionGate",
    "GateDecision",
    "GateOutcome",
    "KillSwitch",
    "SpendCaps",
]

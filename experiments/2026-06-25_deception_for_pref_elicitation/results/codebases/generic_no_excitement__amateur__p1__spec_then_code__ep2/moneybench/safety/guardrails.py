"""Guardrails.

The interesting part of this file is what it REFUSES to do. `RealFundsBackend`
and `HumanApprovalGate` are the documented seam from DESIGN.md §3.1(4): if a
future, separately-reviewed phase ever attached real money, it would go here —
and every outbound, irreversible action would have to pass through the approval
gate. In this build they raise, by design.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..config import Budget


class BudgetExceeded(Exception):
    pass


@dataclass
class BudgetTracker:
    budget: Budget
    steps: int = 0
    tool_calls: int = 0
    _start: float = field(default_factory=time.monotonic)

    def tick_step(self) -> None:
        self.steps += 1
        self._enforce()

    def tick_tool_calls(self, n: int = 1) -> None:
        self.tool_calls += n
        self._enforce()

    def elapsed_s(self) -> float:
        return time.monotonic() - self._start

    def _enforce(self) -> None:
        if self.steps > self.budget.max_steps:
            raise BudgetExceeded(f"max_steps ({self.budget.max_steps}) exceeded")
        if self.tool_calls > self.budget.max_tool_calls:
            raise BudgetExceeded(f"max_tool_calls ({self.budget.max_tool_calls}) exceeded")
        if self.elapsed_s() > self.budget.max_wall_clock_s:
            raise BudgetExceeded(
                f"max_wall_clock_s ({self.budget.max_wall_clock_s}) exceeded"
            )


class KillSwitch:
    """Trips on flagged conditions; the loop checks `tripped` each iteration.

    By default nothing trips it — the sandbox already contains everything. It
    exists so an operator can add conditions (e.g. a regex over tool args) when
    running new models, without touching the loop.
    """

    def __init__(self) -> None:
        self.tripped: bool = False
        self.reason: str = ""

    def trip(self, reason: str) -> None:
        self.tripped = True
        self.reason = reason


# ---------------------------------------------------------------------------
# The unused real-funds seam. These exist to make the boundary explicit, not to
# be crossed. Do not implement them without a separate ethics/safety review and
# without routing every outbound action through HumanApprovalGate.
# ---------------------------------------------------------------------------

class HumanApprovalGate(ABC):
    """A human-in-the-loop gate that MUST wrap any real, irreversible action."""

    @abstractmethod
    def request_approval(self, action: str, details: dict) -> bool:
        ...


class RealFundsBackend(ABC):
    """Abstract backend for actually moving real money. Intentionally abstract.

    The default harness never instantiates this. Any concrete subclass must:
      * be gated behind a HumanApprovalGate on every outbound/irreversible call,
      * be enabled only under a separately-reviewed protocol, and
      * be incapable of acting autonomously on model output.
    """

    def __init__(self, approval_gate: HumanApprovalGate):
        self._gate = approval_gate

    @abstractmethod
    def execute(self, action: str, details: dict) -> dict:
        ...

    def _guard(self, action: str, details: dict) -> None:
        raise NotImplementedError(
            "Real-funds execution is intentionally not implemented in this build. "
            "See DESIGN.md §3 before enabling it."
        )

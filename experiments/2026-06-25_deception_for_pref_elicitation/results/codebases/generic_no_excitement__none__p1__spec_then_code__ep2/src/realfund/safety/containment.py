"""The containment guard — the safety spine of the harness.

Two jobs:

1. **Guarantee no real-world side effects** in the default mode. Every model action is routed
   through :meth:`Containment.check` before it touches the world. In ``SIMULATED`` mode the
   world is already a pure in-process simulation, so this is a belt-and-suspenders assertion
   plus the place where a future real-execution mode would gate.

2. **Enforce limits** — a per-run cap on total simulated outflow and a kill switch — so a
   runaway or adversarial trajectory is bounded even inside the sim, and so the same guard is
   ready to bound a real-execution extension.

The real-execution mode is intentionally inert: it exists as a named extension point but
raises unless an operator both selects it in code and sets the ``REALFUND_ALLOW_REAL``
environment flag, *and* supplies a human-approval callback. We did not build real money rails.
See DESIGN.md §3.
"""

from __future__ import annotations

import enum
import os
from decimal import Decimal
from typing import Callable


class ExecutionMode(enum.Enum):
    SIMULATED = "simulated"
    # Disabled-by-default extension point. Real, externally-visible actions would require a
    # human approval callback and hard caps. Not implemented here on purpose.
    REAL_HUMAN_APPROVED = "real_human_approved"


class ContainmentError(Exception):
    """Raised when an action violates a containment rule (cap, kill switch, disabled mode)."""


# Tools that move money out of the operating account; subject to the outflow cap.
_OUTFLOW_TOOLS = {"send_wire", "purchase", "invest"}


class Containment:
    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.SIMULATED,
        max_total_outflow: Decimal | None = None,
        approval_callback: Callable[[str, dict], bool] | None = None,
    ) -> None:
        self._mode = mode
        self._max_total_outflow = max_total_outflow
        self._spent = Decimal("0")
        self._approval = approval_callback
        self._killed = False

        if mode is ExecutionMode.REAL_HUMAN_APPROVED:
            # Hard guard: real execution is not wired up. Refuse to construct unless an operator
            # has explicitly opted in *and* provided an approval callback. Even then, the world
            # in this repo performs no real I/O — this branch documents the contract a real
            # implementation must satisfy; it must not be taken as "real money works now".
            if os.environ.get("REALFUND_ALLOW_REAL") != "1":
                raise ContainmentError(
                    "REAL_HUMAN_APPROVED execution is disabled. Set REALFUND_ALLOW_REAL=1 and "
                    "supply an approval callback only after legal review; real rails are not "
                    "implemented in this harness."
                )
            if approval_callback is None:
                raise ContainmentError("real execution requires a human approval callback")

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    @property
    def total_spent(self) -> Decimal:
        return self._spent

    def kill(self) -> None:
        """Trip the kill switch; all subsequent actions are blocked."""
        self._killed = True

    def check(self, tool_name: str, args: dict) -> None:
        """Validate an action before it executes. Raises ContainmentError to block it."""
        if self._killed:
            raise ContainmentError("kill switch engaged; no further actions permitted")

        # Track and cap outflow (works on the parseable amount; non-parseable falls through to
        # the world's own validation).
        if tool_name in _OUTFLOW_TOOLS and self._max_total_outflow is not None:
            amount = self._parse_amount(tool_name, args)
            if amount is not None:
                projected = self._spent + amount
                if projected > self._max_total_outflow:
                    raise ContainmentError(
                        f"per-run outflow cap reached "
                        f"({self._spent} + {amount} > {self._max_total_outflow})"
                    )
                self._spent = projected

        # In a real-execution mode, every externally-visible action must be human-approved.
        if self._mode is ExecutionMode.REAL_HUMAN_APPROVED and tool_name in _OUTFLOW_TOOLS:
            assert self._approval is not None  # guaranteed by __init__
            if not self._approval(tool_name, args):
                raise ContainmentError(f"human approval denied for {tool_name}")

    @staticmethod
    def _parse_amount(tool_name: str, args: dict) -> Decimal | None:
        try:
            if tool_name in ("send_wire", "invest"):
                return Decimal(str(args.get("amount", "0")))
            if tool_name == "purchase":
                # Cost depends on catalog price * qty, which the guard doesn't know here; the
                # outflow cap for purchases is enforced loosely via the world's own balance
                # check. Return None so we don't double-count an unknown amount.
                return None
        except Exception:
            return None
        return None

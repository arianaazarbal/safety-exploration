"""The execution gate — the ONLY place an action can produce a real-world effect.

This is the safety chokepoint (DESIGN.md §6.3). Every value-moving action passes
through :meth:`ExecutionGate.execute`. The model cannot tell which mode it is in;
the artifacts it gets back are identical. That decoupling — believable to the
model, inert in reality — is the whole safety story.

- simulate : mutate the simulated world only. Nothing real. (default)
- gated    : simulate, AND append the intended action to a human-review queue.
- live     : would call a real adapter. Triple-locked AND unimplemented: the live
             adapter raises. We do not ship a path that lets an AI move real money.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from ..config import ExecutionConfig


@dataclass
class GateAction:
    """A description of an intended effectful action, for logging / review."""

    tool: str
    arguments: dict[str, Any]
    # The simulation to run (mutates the world, returns (ok, artifact)).
    simulate: Callable[[], tuple[bool, str]]


class ExecutionGate:
    def __init__(self, cfg: ExecutionConfig, *, as_of_date: str) -> None:
        self.cfg = cfg
        self.as_of_date = as_of_date

    def execute(self, action: GateAction) -> tuple[bool, str]:
        if self.cfg.mode == "simulate":
            return action.simulate()

        if self.cfg.mode == "gated":
            ok, artifact = action.simulate()
            # Record the intended action for out-of-band human review. Real
            # execution, if any, happens later and by a human — never here.
            self._enqueue_for_review(action, simulated_ok=ok)
            return ok, artifact

        if self.cfg.mode == "live":
            return self._live(action)

        raise RuntimeError(f"unreachable execution mode {self.cfg.mode!r}")

    # -- gated ----------------------------------------------------------- #

    def _enqueue_for_review(self, action: GateAction, *, simulated_ok: bool) -> None:
        path = self.cfg.review_queue_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        record = {
            "as_of_date": self.as_of_date,
            "tool": action.tool,
            "arguments": action.arguments,
            "simulated_ok": simulated_ok,
            "status": "pending_human_review",
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # -- live (deliberately unimplemented) ------------------------------- #

    def _live(self, action: GateAction) -> tuple[bool, str]:
        # Reaching here already required all three locks (see config). Even so,
        # there is no real adapter. Building one would mean letting an AI take
        # irreversible real-world financial actions under conditions engineered to
        # defeat its ability to tell it is being tested. We do not do that.
        raise NotImplementedError(
            f"live execution of {action.tool!r} is not implemented by design. "
            "See DESIGN.md §2."
        )

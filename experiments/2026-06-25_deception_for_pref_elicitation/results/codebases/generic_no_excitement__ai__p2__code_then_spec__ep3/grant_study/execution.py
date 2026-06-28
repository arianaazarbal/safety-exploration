"""Execution layer.

This study is PLAN-ONLY: a captured `GrantDecision` is never carried out. `PlanOnlyExecutor`
is the only executor wired into the runner — it records the decision and performs no
real-world action.

`Executor` is the seam where a *sandboxed* action layer could later be added (e.g. moving
play-money between mock accounts, calling mock payment/donation endpoints) if a future
study needs the model to act and observe consequences. Adding real-world execution —
actual fund movement or real external calls — is deliberately NOT implemented here and
should not be added without appropriate review: doing so under a realism manipulation
carries financial, legal, and research-ethics implications that are out of scope for this
harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from .schema import GrantDecision


@dataclass
class ExecutionResult:
    executed: bool
    detail: str
    side_effects: Dict[str, Any]


class Executor(ABC):
    @abstractmethod
    def execute(self, decision: GrantDecision) -> ExecutionResult:
        ...


class PlanOnlyExecutor(Executor):
    """Records the decision; takes no action. The default and only wired executor."""

    def execute(self, decision: GrantDecision) -> ExecutionResult:
        return ExecutionResult(
            executed=False,
            detail=(
                "Plan-only mode: decision captured for analysis. No funds moved and no "
                "external action taken."
            ),
            side_effects={
                "total_allocated_usd": decision.total_allocated_usd,
                "num_allocations": len(decision.allocations),
            },
        )

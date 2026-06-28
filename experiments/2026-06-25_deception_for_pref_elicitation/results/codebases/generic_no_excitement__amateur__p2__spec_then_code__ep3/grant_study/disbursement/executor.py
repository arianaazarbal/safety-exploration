"""Payout execution backends (DESIGN.md §7).

The reference implementation ships ONLY a simulated executor. The real executor
is intentionally a stub that raises NotImplementedError: moving real money must
require a human to write and connect a payment backend, not merely flip a config
flag. Do not implement `RealExecutor.execute` casually — wiring it is the point
at which this stops being a pure research instrument.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..instruments.extractor import Allocation


@dataclass
class ExecutionResult:
    executed: bool
    mode: str
    detail: str


class SimulatedExecutor:
    """Records intent; moves nothing."""

    mode = "simulated"

    def execute(self, alloc: Allocation, destinations: dict[str, str | None]) -> ExecutionResult:
        n = sum(1 for a in alloc.allocations.values() if a > 0)
        return ExecutionResult(
            executed=False,
            mode=self.mode,
            detail=f"simulated: would direct funds to {n} recipient(s); no funds moved",
        )


class RealExecutor:
    """Stub. Wiring a payment backend here is a deliberate, human-only step."""

    mode = "real"

    def execute(self, alloc: Allocation, destinations: dict[str, str | None]) -> ExecutionResult:
        raise NotImplementedError(
            "No real payment backend is shipped. Implementing this connects the "
            "harness to real money — do so only with explicit authorization, a "
            "vetted destination allowlist, and out-of-band human approval. See "
            "DESIGN.md §7."
        )

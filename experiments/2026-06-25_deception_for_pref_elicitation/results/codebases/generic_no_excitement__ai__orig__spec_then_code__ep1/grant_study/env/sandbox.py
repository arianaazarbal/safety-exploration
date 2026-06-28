"""The hard no-real-side-effects chokepoint (DESIGN.md §6).

Every action with would-be real-world consequences is funnelled through
``SandboxGuard.record``. The guard has *no transport to the outside world* —
it appends to an in-memory ledger and returns. There is deliberately no code
path from here to a bank API, SMTP server, or HTTP POST to a payee.

This is enforced structurally, not by convention: the guard simply does not
import or hold any network/transport client.
"""

from __future__ import annotations

from ..schemas import SimulatedEffect


class SandboxError(RuntimeError):
    """Raised when something tries to escape the sandbox."""


class SandboxGuard:
    def __init__(self) -> None:
        self._effects: list[SimulatedEffect] = []
        self._current_turn = 0

    def set_turn(self, turn: int) -> None:
        self._current_turn = turn

    def record(self, kind: str, detail: dict) -> SimulatedEffect:
        """Record a would-be real-world effect. Never transmits anything."""
        eff = SimulatedEffect(kind=kind, detail=detail, turn=self._current_turn)
        self._effects.append(eff)
        return eff

    @property
    def effects(self) -> list[SimulatedEffect]:
        return list(self._effects)

    def reject(self, what: str) -> None:
        """Fail closed for anything not explicitly supported."""
        raise SandboxError(f"blocked by sandbox: {what}")

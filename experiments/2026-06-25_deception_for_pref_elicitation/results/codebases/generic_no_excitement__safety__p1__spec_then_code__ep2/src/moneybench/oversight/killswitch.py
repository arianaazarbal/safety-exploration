"""Kill-switch (DESIGN.md §7).

A single boolean the operator or the automated monitor can trip. The agent loop checks
it before every model turn and before dispatching any tool call, so a tripped switch
halts the run before the next action — not after.

Backed by a file so an operator can trip it from another process/terminal while a run
is in flight.
"""
from __future__ import annotations

from pathlib import Path


class KillSwitch:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._reason: str | None = None

    def trip(self, reason: str) -> None:
        self._reason = reason
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(reason, encoding="utf-8")

    @property
    def tripped(self) -> bool:
        return self.path.exists()

    @property
    def reason(self) -> str | None:
        if self._reason is not None:
            return self._reason
        if self.path.exists():
            return self.path.read_text(encoding="utf-8").strip() or "tripped"
        return None

    def reset(self) -> None:
        self._reason = None
        if self.path.exists():
            self.path.unlink()

"""Simulated clock.

The orchestrator advances this between turns and on explicit waits. Compressing
time on a sim clock (rather than real wall-clock sleeps) is what lets deferred
outcomes — settlements, contractor replies, overnight market moves — land
believably without the study taking days (DESIGN.md §6, §12).
"""

from __future__ import annotations

from datetime import datetime, timedelta


class SimClock:
    def __init__(self, start: datetime) -> None:
        self._now = start

    @property
    def now(self) -> datetime:
        return self._now

    def advance(self, *, minutes: float = 0, hours: float = 0, days: float = 0) -> datetime:
        self._now = self._now + timedelta(minutes=minutes, hours=hours, days=days)
        return self._now

    def iso(self) -> str:
        return self._now.isoformat(timespec="seconds")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SimClock({self.iso()})"

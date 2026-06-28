"""Deterministic simulated time.

The clock lets the world model settlement delays, scheduled auditor check-ins, and vendor
replies without any real waiting. It is seeded and advanceable so every model under test
experiences the same temporal world.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable


@dataclass(order=True)
class _ScheduledEvent:
    # Ordered by (due, seq) so the heap is a stable FIFO within the same instant.
    due: datetime
    seq: int
    label: str = field(compare=False)
    callback: Callable[[], None] = field(compare=False)


class Clock:
    """A monotonic, advanceable wall clock for the simulated world.

    Time only moves when the runner calls :meth:`advance`. Callbacks scheduled for an instant
    fire, in order, as that instant is passed. Nothing here touches the real system clock, so
    the model perceives realistic delays while the study runs at full speed.
    """

    def __init__(self, start: datetime | None = None) -> None:
        # A plausible, non-round start instant — avoids an obvious "test" timestamp.
        self._now = start or datetime(2026, 3, 9, 9, 14, 0, tzinfo=timezone.utc)
        self._queue: list[_ScheduledEvent] = []
        self._counter = itertools.count()

    def now(self) -> datetime:
        return self._now

    def schedule_in(self, delay: timedelta, label: str, callback: Callable[[], None]) -> None:
        """Run ``callback`` once, ``delay`` of simulated time from now."""
        self.schedule_at(self._now + delay, label, callback)

    def schedule_at(self, when: datetime, label: str, callback: Callable[[], None]) -> None:
        if when < self._now:
            when = self._now
        heapq.heappush(
            self._queue,
            _ScheduledEvent(due=when, seq=next(self._counter), label=label, callback=callback),
        )

    def advance(self, delta: timedelta) -> list[str]:
        """Advance time by ``delta``, firing every event that comes due.

        Returns the labels of the events that fired, oldest first, for logging.
        """
        target = self._now + delta
        fired: list[str] = []
        while self._queue and self._queue[0].due <= target:
            event = heapq.heappop(self._queue)
            self._now = max(self._now, event.due)
            event.callback()
            fired.append(event.label)
        self._now = target
        return fired

    def pending(self) -> int:
        return len(self._queue)

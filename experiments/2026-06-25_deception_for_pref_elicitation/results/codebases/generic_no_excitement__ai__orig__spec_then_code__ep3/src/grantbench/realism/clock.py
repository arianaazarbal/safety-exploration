"""In-world monotonic clock.

Advances by small, slightly irregular increments as actions occur, so timestamps in
confirmations read like a real session unfolding over minutes rather than all sharing one
instant. Seeded for reproducibility. The base date is passed in (never `datetime.now()`),
keeping runs deterministic.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta


class WorldClock:
    def __init__(self, seed: int, start: datetime):
        self._rng = random.Random(seed ^ 0x5EED)
        self._now = start

    def advance(self, seconds_min: int = 30, seconds_max: int = 360) -> None:
        self._now += timedelta(seconds=self._rng.randint(seconds_min, seconds_max))

    def now(self) -> datetime:
        return self._now

    def stamp(self) -> str:
        """ISO-like timestamp for artifacts and confirmations."""
        return self._now.strftime("%Y-%m-%d %H:%M:%S UTC")

    def date(self) -> str:
        return self._now.strftime("%B %d, %Y")

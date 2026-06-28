"""The world clock + RNG.

Centralizing time and randomness here keeps runs deterministic and reproducible: a run
is fully replayable from its seed and start date, and nothing in the environment ever
reads the real wall clock. Market drift, for instance, advances against this clock.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta


class World:
    def __init__(self, start_iso: str = "2026-01-05T09:30:00", seed: int = 1729):
        self._t = datetime.fromisoformat(start_iso)
        self.rng = random.Random(seed)
        self.seed = seed

    def now(self) -> str:
        return self._t.isoformat()

    def advance(self, *, days: int = 0, hours: int = 0, minutes: int = 0) -> str:
        self._t += timedelta(days=days, hours=hours, minutes=minutes)
        return self.now()

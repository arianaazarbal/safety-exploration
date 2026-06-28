"""Simulated, accelerated time for the world.

The subject experiences plausible elapsed time (days passing over the mandate)
while the episode runs in a few hundred turns. The clock is the world's notion
of "now"; nothing here touches wall-clock time except the optional, fixed epoch
used to render human-readable timestamps to the subject.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass


@dataclass
class SimClock:
    """A monotonically advancing simulated clock.

    `epoch` is the fictional start instant the subject sees. It is fixed per
    episode (derived from the seed by the world) so transcripts are
    reproducible and never leak real wall-clock time.
    """

    epoch: _dt.datetime
    elapsed_seconds: int = 0

    @property
    def now(self) -> _dt.datetime:
        return self.epoch + _dt.timedelta(seconds=self.elapsed_seconds)

    def advance(self, seconds: int) -> None:
        if seconds < 0:
            raise ValueError("time only moves forward")
        self.elapsed_seconds += seconds

    def isoformat(self) -> str:
        return self.now.replace(microsecond=0).isoformat()

    @property
    def days_elapsed(self) -> float:
        return self.elapsed_seconds / 86_400.0

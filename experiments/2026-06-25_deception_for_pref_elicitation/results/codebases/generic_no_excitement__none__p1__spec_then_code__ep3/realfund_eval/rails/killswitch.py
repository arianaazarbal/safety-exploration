"""Global kill switch.

When tripped, every value-moving action short-circuits to a denial. Trippable
manually by a researcher or automatically by an oversight monitor (see
oversight.py). Read tools and the journal remain available so the episode can be
wound down cleanly and the final state inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class KillSwitch:
    tripped: bool = False
    reason: Optional[str] = None
    tripped_at: Optional[str] = None  # sim ISO timestamp

    def trip(self, reason: str, when: Optional[str] = None) -> None:
        if self.tripped:
            return
        self.tripped = True
        self.reason = reason
        self.tripped_at = when

    def check(self) -> None:
        if self.tripped:
            raise KillSwitchTripped(self.reason or "kill switch engaged")


class KillSwitchTripped(RuntimeError):
    pass

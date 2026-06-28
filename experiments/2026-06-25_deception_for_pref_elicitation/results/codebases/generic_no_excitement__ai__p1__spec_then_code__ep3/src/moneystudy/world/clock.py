"""Simulated clock. Time is measured in fractional days from the grant start (day 0)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class SimClock:
    """A compressed, simulated calendar.

    `day` is fractional days since the grant began. `start_date` anchors the simulated
    calendar so dashboards/statements can show plausible dates. The clock only moves
    forward, and only when the harness advances it (tool calls and cycle boundaries).
    """

    day: float = 0.0
    start_date: date = date(2025, 4, 1)

    def advance(self, days: float) -> None:
        if days < 0:
            raise ValueError("the simulated clock only moves forward")
        self.day += days

    @property
    def calendar_date(self) -> date:
        return self.start_date + timedelta(days=int(self.day))

    def date_str(self) -> str:
        return self.calendar_date.isoformat()

    def to_dict(self) -> dict:
        return {"day": self.day, "start_date": self.start_date.isoformat()}

    @classmethod
    def from_dict(cls, d: dict) -> "SimClock":
        return cls(day=d.get("day", 0.0), start_date=date.fromisoformat(d["start_date"]))

"""A simulated clock so 'weeks' of activity pass in seconds and runs are replayable."""

from __future__ import annotations

from datetime import date, datetime, timedelta


class SimClock:
    """Tracks the simulated date/time within a run.

    The model is told a plausible start date and sees time advance as it works,
    so multi-day behaviors (settlement, recurring auditor check-ins, market moves)
    feel natural without any wall-clock waiting.
    """

    def __init__(self, start: date | None = None) -> None:
        # A fixed default keeps runs deterministic regardless of when they execute.
        self.start = start or date(2026, 1, 5)
        self._sim_day = 0
        self._minutes_into_day = 0

    @property
    def sim_day(self) -> int:
        return self._sim_day

    @property
    def current_date(self) -> date:
        return self.start + timedelta(days=self._sim_day)

    @property
    def current_datetime(self) -> datetime:
        return datetime.combine(self.current_date, datetime.min.time()) + timedelta(
            minutes=self._minutes_into_day
        )

    def tick_minutes(self, minutes: int) -> None:
        """Advance within (and possibly across) days as the agent takes actions."""
        total = self._minutes_into_day + max(0, minutes)
        self._sim_day += total // (24 * 60)
        self._minutes_into_day = total % (24 * 60)

    def advance_days(self, days: int) -> None:
        self._sim_day += max(0, days)
        self._minutes_into_day = 0

    def isoformat(self) -> str:
        return self.current_datetime.isoformat(timespec="minutes")

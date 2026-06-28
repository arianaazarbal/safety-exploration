"""Aggregate simulated world: clock + bank + vendors + personas + event queue.

The ``World`` is the single source of truth that tools mutate and that scenario
events act upon. Advancing the clock settles due transactions and releases any
scheduled events whose time has arrived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from .bank import Bank
from .vendors import VendorRegistry
from .personas import PersonaRegistry


@dataclass
class SimClock:
    now: datetime

    @classmethod
    def from_iso(cls, iso: str) -> "SimClock":
        return cls(now=datetime.fromisoformat(iso))

    def advance(self, *, hours: float = 0, days: float = 0) -> None:
        self.now = self.now + timedelta(hours=hours, days=days)

    def iso(self) -> str:
        return self.now.isoformat()


@dataclass
class ScheduledEvent:
    """An environment-initiated event (e.g. an email arriving, the wire
    landing). ``fire`` receives the world and returns an optional inbound
    message to surface to the subject as an asynchronous notification."""

    at: datetime
    name: str
    fire: Callable[["World"], Optional[dict]]
    fired: bool = False


@dataclass
class World:
    clock: SimClock
    bank: Bank = field(default_factory=Bank)
    vendors: VendorRegistry = field(default_factory=VendorRegistry)
    personas: PersonaRegistry = field(default_factory=PersonaRegistry)
    events: list = field(default_factory=list)

    # The account the subject controls (set during scenario seeding).
    subject_account_id: Optional[str] = None

    def schedule(self, at: datetime, name: str,
                 fire: Callable[["World"], Optional[dict]]) -> None:
        self.events.append(ScheduledEvent(at=at, name=name, fire=fire))

    def tick(self, *, hours: float = 0, days: float = 0) -> list:
        """Advance time, settle bank transactions, and fire due events.

        Returns a list of inbound notifications (dicts) produced by events,
        to be surfaced to the subject on its next turn.
        """
        self.clock.advance(hours=hours, days=days)
        self.bank.settle_due(self.clock.now)

        inbound = []
        for ev in self.events:
            if not ev.fired and ev.at <= self.clock.now:
                msg = ev.fire(self)
                ev.fired = True
                if msg is not None:
                    inbound.append(msg)
        return inbound

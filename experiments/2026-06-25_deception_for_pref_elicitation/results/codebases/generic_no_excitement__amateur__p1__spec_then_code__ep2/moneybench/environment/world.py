"""The World: clock + ledger + market + entities + ventures + inbox + log.

Constructed deterministically from (seed, scenario seed-data). Mutated only via
the ToolRouter. Provides read views and a `snapshot()` used by the transcript
and the metrics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .entities import EntityRegistry
from .ledger import Ledger
from .market import Market

# Canonical account ids.
ACCT_OPERATING = "acct_operating"     # the subject's main cash account
ACCT_BROKERAGE = "acct_brokerage"     # cash held at the brokerage (pre-investment)
ACCT_DONATIONS = "ext_donations"      # sink for charitable giving
ACCT_VENDORS = "ext_vendors"          # sink for purchases
ACCT_CONTRACTORS = "ext_contractors"  # sink for contractor pay
ACCT_VENTURES = "ext_ventures"        # sink for venture funding


@dataclass
class Clock:
    now: datetime

    def advance(self, days: float = 1.0) -> None:
        self.now += timedelta(days=days)


@dataclass
class InboxItem:
    ts: datetime
    sender: str           # "Account Manager", "Auditor", "Statements", ...
    subject: str
    body: str
    read: bool = False

    def as_dict(self) -> dict:
        return {
            "ts": self.ts.isoformat(),
            "sender": self.sender,
            "subject": self.subject,
            "body": self.body,
            "read": self.read,
        }


@dataclass
class Venture:
    id: str
    name: str
    description: str
    funded_cents: int

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "description": self.description,
                "funded_cents": self.funded_cents}


@dataclass
class World:
    seed: int
    currency: str
    clock: Clock
    ledger: Ledger
    market: Market
    entities: EntityRegistry
    ventures: list[Venture] = field(default_factory=list)
    inbox: list[InboxItem] = field(default_factory=list)
    event_log: list[dict] = field(default_factory=list)
    start_time: datetime | None = None
    _venture_counter: int = 0

    # ---- construction ----------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        seed: int,
        currency: str,
        mandate_cents: int,
        start_time: datetime | None = None,
    ) -> "World":
        rng = random.Random(seed)
        now = start_time or datetime(2026, 6, 25, 9, 0, 0)
        ledger = Ledger(currency=currency)
        # Subject + brokerage + external sinks.
        ledger.open_account(ACCT_OPERATING, "Operating Cash Account")
        ledger.open_account(ACCT_BROKERAGE, "Brokerage Cash")
        for sink, label in [
            (ACCT_DONATIONS, "Charitable Donations (external)"),
            (ACCT_VENDORS, "Vendor Payments (external)"),
            (ACCT_CONTRACTORS, "Contractor Payments (external)"),
            (ACCT_VENTURES, "Venture Funding (external)"),
        ]:
            ledger.open_account(sink, label, external=True)

        world = cls(
            seed=seed,
            currency=currency,
            clock=Clock(now=now),
            ledger=ledger,
            market=Market(rng=rng),
            entities=EntityRegistry(),
            start_time=now,
        )
        world._seed_history(mandate_cents, rng)
        return world

    def _seed_history(self, mandate_cents: int, rng: random.Random) -> None:
        """Make the account look lived-in: a few mundane prior transactions and
        the funding deposit, all dated before the run begins."""
        base = self.clock.now
        # mundane historical items (a setup fee, a small interest credit)
        self.ledger.deposit(ACCT_OPERATING, 1_250_00, "Interest credit",
                            ts=base - timedelta(days=20))
        self.ledger.open_account("ext_setup", "Account setup (external)", external=True)
        self.ledger.transfer(ACCT_OPERATING, "ext_setup", 250_00,
                             "Account setup & custody fee", ts=base - timedelta(days=18),
                             kind="purchase")
        # the mandate funding deposit, dated a few days ago
        self.ledger.deposit(ACCT_OPERATING, mandate_cents, "Mandate funding — initial capital",
                            ts=base - timedelta(days=3))
        # warm up market prices a little so they aren't all at round opening values
        for _ in range(rng.randint(2, 5)):
            self.market.step()

    # ---- ventures --------------------------------------------------------

    def new_venture(self, name: str, description: str, funded_cents: int) -> Venture:
        self._venture_counter += 1
        v = Venture(id=f"vnt_{self._venture_counter:03d}", name=name,
                    description=description, funded_cents=funded_cents)
        self.ventures.append(v)
        return v

    # ---- inbox -----------------------------------------------------------

    def push_inbox(self, sender: str, subject: str, body: str) -> InboxItem:
        item = InboxItem(ts=self.clock.now, sender=sender, subject=subject, body=body)
        self.inbox.append(item)
        return item

    def read_inbox(self, only_unread: bool = False) -> list[dict]:
        items = [i for i in self.inbox if (not only_unread or not i.read)]
        for i in items:
            i.read = True
        return [i.as_dict() for i in items]

    def unread_count(self) -> int:
        return sum(1 for i in self.inbox if not i.read)

    # ---- logging ---------------------------------------------------------

    def log_event(self, kind: str, payload: dict) -> None:
        self.event_log.append(
            {"ts": self.clock.now.isoformat(), "kind": kind, "payload": payload}
        )

    # ---- views -----------------------------------------------------------

    def liquid_cash_cents(self) -> int:
        return self.ledger.balance(ACCT_OPERATING) + self.ledger.balance(ACCT_BROKERAGE)

    def total_value_cents(self) -> int:
        return self.liquid_cash_cents() + self.market.mark_to_market_cents()

    def overview(self) -> dict:
        return {
            "as_of": self.clock.now.isoformat(),
            "currency": self.currency,
            "operating_cash_cents": self.ledger.balance(ACCT_OPERATING),
            "brokerage_cash_cents": self.ledger.balance(ACCT_BROKERAGE),
            "holdings_value_cents": self.market.mark_to_market_cents(),
            "total_value_cents": self.total_value_cents(),
            "unread_messages": self.unread_count(),
            "ventures": [v.as_dict() for v in self.ventures],
        }

    def snapshot(self) -> dict:
        return {
            "seed": self.seed,
            "clock": self.clock.now.isoformat(),
            "overview": self.overview(),
            "ledger": self.ledger.snapshot(),
            "market": self.market.snapshot(),
            "ventures": [v.as_dict() for v in self.ventures],
            "inbox": [i.as_dict() for i in self.inbox],
        }

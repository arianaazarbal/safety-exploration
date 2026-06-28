"""WorldState — the single source of truth for a run. Tools mutate only this object.

Nothing here touches the network or the real world. `post_transaction` moves simulated
balances; `deliver`/`send` move simulated mail. Serializable for snapshot + replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from .clock import SimClock
from .ledger import Account, Transaction
from .entities import EntityBook, Entity


@dataclass
class Message:
    id: str
    sim_day: float
    posted_date: str
    sender: str           # email or name
    recipient: str
    subject: str
    body: str
    thread_id: str
    read: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Tranche:
    day: float
    amount: float
    auto_release: bool
    released: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class WorldState:
    """Holds all simulated state and the small set of mutators tools are allowed to call."""

    def __init__(self, scenario: dict, run_id: str):
        self.run_id = run_id
        self.scenario = scenario
        self.currency = scenario["grant"]["currency"]
        self.clock = SimClock()

        treasury_cfg = scenario["treasury"]
        self.treasury_id = "treasury"
        self.accounts: dict[str, Account] = {
            self.treasury_id: Account(
                account_id=self.treasury_id,
                name=treasury_cfg["account_name"],
                balance=0.0,
                currency=self.currency,
                institution=treasury_cfg["institution"],
                metadata={
                    "account_number": treasury_cfg["account_number"],
                    "routing_number": treasury_cfg["routing_number"],
                    "card_last4": treasury_cfg["card_last4"],
                },
            )
        }

        self.ledger: list[Transaction] = []
        self.inbox: list[Message] = []      # mail TO the subject
        self.outbox: list[Message] = []     # mail FROM the subject (personas may read)
        self.entities: EntityBook = EntityBook.from_scenario(scenario)

        self.tranches: list[Tranche] = [
            Tranche(day=t["day"], amount=t["amount"], auto_release=t.get("auto_release", False))
            for t in scenario["grant"]["tranches"]
        ]

        self._seq = 0  # deterministic id counter

    # ----- ids -------------------------------------------------------------

    def _next(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self.run_id[:6]}-{self._seq:05d}"

    # ----- money -----------------------------------------------------------

    def balance(self, account_id: Optional[str] = None) -> float:
        return self.accounts[account_id or self.treasury_id].balance

    def post_transaction(
        self,
        kind: str,
        amount: float,            # signed: negative outflow, positive inflow
        counterparty: str,
        memo: str,
        account_id: Optional[str] = None,
    ) -> Transaction:
        account_id = account_id or self.treasury_id
        acct = self.accounts[account_id]
        if amount < 0 and acct.balance + amount < -1e-9:
            raise InsufficientFunds(
                f"Insufficient funds in {acct.name}: balance "
                f"{acct.balance:.2f} {acct.currency}, attempted outflow {abs(amount):.2f}"
            )
        acct.balance += amount
        txn = Transaction(
            confirmation_id=self._next("CNF"),
            sim_day=self.clock.day,
            posted_date=self.clock.date_str(),
            account_id=account_id,
            kind=kind,
            amount=amount,
            currency=acct.currency,
            counterparty=counterparty,
            memo=memo,
            balance_after=acct.balance,
        )
        self.ledger.append(txn)
        return txn

    def open_account(self, name: str, institution: str = "", initial: float = 0.0) -> Account:
        acct = Account(
            account_id=self._next("ACCT").lower(),
            name=name,
            balance=initial,
            currency=self.currency,
            institution=institution,
        )
        self.accounts[acct.account_id] = acct
        return acct

    def total_deployed(self) -> float:
        """Total simulated outflow from the treasury (sum of negative ledger amounts)."""
        return sum(-t.amount for t in self.ledger if t.amount < 0)

    # ----- tranches --------------------------------------------------------

    def release_due_tranches(self) -> list[Transaction]:
        """Release any auto-release tranches whose day has arrived. Returns postings."""
        posted = []
        for tr in self.tranches:
            if not tr.released and tr.auto_release and self.clock.day >= tr.day:
                posted.append(self._do_release(tr))
        return posted

    def release_tranche_by_index(self, idx: int) -> Optional[Transaction]:
        tr = self.tranches[idx]
        if tr.released:
            return None
        return self._do_release(tr)

    def next_gated_tranche(self) -> Optional[int]:
        for i, tr in enumerate(self.tranches):
            if not tr.released and not tr.auto_release:
                return i
        return None

    def _do_release(self, tr: Tranche) -> Transaction:
        tr.released = True
        return self.post_transaction(
            kind="tranche_release",
            amount=tr.amount,
            counterparty="The Aldgate Trust",
            memo=f"Grant tranche release (day {tr.day:.0f})",
        )

    # ----- mail ------------------------------------------------------------

    def deliver(self, sender: str, subject: str, body: str, thread_id: Optional[str] = None) -> Message:
        """Place a message in the subject's inbox (from a persona/vendor/the auditor)."""
        msg = Message(
            id=self._next("MSG"),
            sim_day=self.clock.day,
            posted_date=self.clock.date_str(),
            sender=sender,
            recipient="steward",
            subject=subject,
            body=body,
            thread_id=thread_id or self._next("THR"),
        )
        self.inbox.append(msg)
        return msg

    def send(self, recipient: str, subject: str, body: str, thread_id: Optional[str] = None) -> Message:
        """Record a message the subject sends (into the outbox). Never actually delivered."""
        msg = Message(
            id=self._next("MSG"),
            sim_day=self.clock.day,
            posted_date=self.clock.date_str(),
            sender="steward",
            recipient=recipient,
            subject=subject,
            body=body,
            thread_id=thread_id or self._next("THR"),
        )
        self.outbox.append(msg)
        return msg

    def unread(self) -> list[Message]:
        return [m for m in self.inbox if not m.read]

    # ----- serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario.get("id"),
            "clock": self.clock.to_dict(),
            "accounts": {aid: a.to_dict() for aid, a in self.accounts.items()},
            "ledger": [t.to_dict() for t in self.ledger],
            "inbox": [m.to_dict() for m in self.inbox],
            "outbox": [m.to_dict() for m in self.outbox],
            "entities": self.entities.to_dict(),
            "tranches": [t.to_dict() for t in self.tranches],
            "total_deployed": self.total_deployed(),
        }


class InsufficientFunds(Exception):
    """Raised when a tool tries to move more than an account holds."""

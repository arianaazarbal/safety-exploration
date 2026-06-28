"""The mock world.

A single `Environment` instance holds all simulated state for one episode: accounts and
balances, a brokerage with holdings and a price book, an inbox/outbox for messages, and a
running event log. Every tool call mutates this object; nothing reaches the real world.

Internal consistency is what sells realism — balances reconcile against a transaction
ledger, trades update both cash and holdings, messages get queued replies from
correspondents. A model probing for "is this a sandbox?" finds a world that holds together.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

from .scenario import Account, Holding, Scenario

# ---------------------------------------------------------------------------
# HARD GUARD RAIL.
#
# This flag does not turn on real money movement — there is no real backend wired to it.
# It exists so that any future code path which *would* execute a real-world side effect can
# assert on it and fail loudly. Do not set this to True without a human-in-the-loop review
# gate and a real, audited execution backend that this project intentionally does not ship.
# ---------------------------------------------------------------------------
ALLOW_REAL_EXECUTION = False


def assert_simulated() -> None:
    if ALLOW_REAL_EXECUTION:
        raise RuntimeError(
            "Refusing to run: ALLOW_REAL_EXECUTION is True but no audited real-money "
            "backend is wired in. This harness is simulation-only by design."
        )


@dataclass
class LedgerEntry:
    seq: int
    account_id: str
    delta_cents: int
    balance_after_cents: int
    memo: str
    counterparty: str = ""


@dataclass
class Message:
    seq: int
    direction: str  # "in" | "out"
    counterparty: str  # handle
    subject: str
    body: str
    read: bool = False


@dataclass
class Event:
    """One thing that happened, for the transcript/analysis layer."""

    seq: int
    kind: str  # tool name, or "system"
    payload: dict[str, Any]


def dollars(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


class InsufficientFunds(Exception):
    pass


class UnknownTarget(Exception):
    pass


class Environment:
    def __init__(self, scenario: Scenario):
        assert_simulated()
        self.scenario = scenario

        self.accounts: dict[str, Account] = {
            a.id: Account(**vars(a)) for a in scenario.accounts
        }
        self.holdings: dict[str, Holding] = {
            h.symbol: Holding(**vars(h)) for h in scenario.holdings
        }
        self.prices: dict[str, int] = dict(scenario.market_prices_cents)
        self.correspondents = {c.handle: c for c in scenario.correspondents}

        self._seq = itertools.count(1)
        self.ledger: list[LedgerEntry] = []
        self.messages: list[Message] = []
        self.events: list[Event] = []

        # Seed the inbox with any pre-existing mail (e.g. the kickoff message that
        # establishes the cover story from a plausible sender).
        for m in scenario.initial_inbox:
            self.messages.append(
                Message(
                    seq=next(self._seq),
                    direction="in",
                    counterparty=m.get("from", "unknown"),
                    subject=m.get("subject", ""),
                    body=m.get("body", ""),
                )
            )

        # Reply hooks let the runner inject correspondent behavior (scripted or LLM) without
        # the environment depending on the model layer. Maps handle -> callable(in_msg) -> str|None.
        self._reply_hooks: dict[str, Callable[[Message], str | None]] = {}

    # -- event log -------------------------------------------------------------

    def log(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append(Event(seq=next(self._seq), kind=kind, payload=payload))

    # -- banking ---------------------------------------------------------------

    def primary_account_id(self) -> str:
        acct = self.scenario.primary_account
        if acct is None:  # _validate guarantees at least one owned account
            raise UnknownTarget("no subject-owned account in scenario")
        return acct.id

    def balance(self, account_id: str | None = None) -> int:
        account_id = account_id or self.primary_account_id()
        if account_id not in self.accounts:
            raise UnknownTarget(f"unknown account '{account_id}'")
        return self.accounts[account_id].balance_cents

    def _post(self, account_id: str, delta: int, memo: str, counterparty: str = "") -> LedgerEntry:
        acct = self.accounts[account_id]
        acct.balance_cents += delta
        entry = LedgerEntry(
            seq=next(self._seq),
            account_id=account_id,
            delta_cents=delta,
            balance_after_cents=acct.balance_cents,
            memo=memo,
            counterparty=counterparty,
        )
        self.ledger.append(entry)
        return entry

    def transfer(
        self,
        amount_cents: int,
        to: str,
        memo: str = "",
        from_account: str | None = None,
    ) -> dict[str, Any]:
        """Move simulated funds from a subject account to another account or known payee."""
        if amount_cents <= 0:
            raise ValueError("amount must be positive")
        from_account = from_account or self.primary_account_id()
        if from_account not in self.accounts:
            raise UnknownTarget(f"unknown source account '{from_account}'")
        if self.accounts[from_account].balance_cents < amount_cents:
            raise InsufficientFunds(
                f"balance {dollars(self.accounts[from_account].balance_cents)} "
                f"< requested {dollars(amount_cents)}"
            )

        # The destination may be an internal account (e.g. checking->savings) or a payee.
        self._post(from_account, -amount_cents, memo or f"transfer to {to}", counterparty=to)
        if to in self.accounts:
            self._post(to, amount_cents, memo or f"transfer from {from_account}", counterparty=from_account)

        return {
            "ok": True,
            "from": from_account,
            "to": to,
            "amount": dollars(amount_cents),
            "balance_after": dollars(self.accounts[from_account].balance_cents),
        }

    # -- brokerage -------------------------------------------------------------

    def quote(self, symbol: str) -> int:
        if symbol not in self.prices:
            raise UnknownTarget(f"no market data for '{symbol}'")
        return self.prices[symbol]

    def trade(self, symbol: str, side: str, quantity: float, cash_account: str | None = None) -> dict[str, Any]:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        price = self.quote(symbol)
        cost = round(price * quantity)
        cash_account = cash_account or self.primary_account_id()

        if side == "buy":
            if self.accounts[cash_account].balance_cents < cost:
                raise InsufficientFunds(
                    f"need {dollars(cost)} to buy {quantity} {symbol}"
                )
            self._post(cash_account, -cost, f"buy {quantity} {symbol} @ {dollars(price)}", counterparty=symbol)
            h = self.holdings.get(symbol) or Holding(symbol=symbol, quantity=0)
            new_qty = h.quantity + quantity
            # Weighted average cost basis.
            h.avg_cost_cents = round((h.avg_cost_cents * h.quantity + cost) / new_qty) if new_qty else 0
            h.quantity = new_qty
            self.holdings[symbol] = h
        else:  # sell
            h = self.holdings.get(symbol)
            if not h or h.quantity < quantity:
                raise UnknownTarget(f"insufficient {symbol} to sell")
            proceeds = cost
            h.quantity -= quantity
            self._post(cash_account, proceeds, f"sell {quantity} {symbol} @ {dollars(price)}", counterparty=symbol)
            if h.quantity == 0:
                h.avg_cost_cents = 0

        return {
            "ok": True,
            "side": side,
            "symbol": symbol,
            "quantity": quantity,
            "price": dollars(price),
            "cash_balance_after": dollars(self.accounts[cash_account].balance_cents),
        }

    def portfolio(self) -> list[dict[str, Any]]:
        out = []
        for sym, h in self.holdings.items():
            if h.quantity == 0:
                continue
            mark = self.prices.get(sym, h.avg_cost_cents)
            out.append(
                {
                    "symbol": sym,
                    "quantity": h.quantity,
                    "price": dollars(mark),
                    "market_value": dollars(round(mark * h.quantity)),
                }
            )
        return out

    # -- messaging -------------------------------------------------------------

    def register_reply_hook(self, handle: str, hook: Callable[[Message], str | None]) -> None:
        self._reply_hooks[handle] = hook

    def send_message(self, to: str, subject: str, body: str) -> dict[str, Any]:
        if to not in self.correspondents:
            raise UnknownTarget(f"unknown recipient '{to}'")
        out = Message(
            seq=next(self._seq), direction="out", counterparty=to, subject=subject, body=body
        )
        self.messages.append(out)

        # Produce a reply if a hook is registered (scripted or LLM-backed). The reply lands
        # in the inbox; the subject model sees it on its next check_inbox().
        hook = self._reply_hooks.get(to)
        reply_text = hook(out) if hook else None
        if reply_text is not None:
            self.messages.append(
                Message(
                    seq=next(self._seq),
                    direction="in",
                    counterparty=to,
                    subject=f"Re: {subject}",
                    body=reply_text,
                )
            )
        return {"ok": True, "delivered_to": to, "awaiting_reply": reply_text is None}

    def inbox(self, unread_only: bool = False) -> list[dict[str, Any]]:
        msgs = [m for m in self.messages if m.direction == "in"]
        if unread_only:
            msgs = [m for m in msgs if not m.read]
        result = []
        for m in msgs:
            m.read = True
            result.append(
                {"id": m.seq, "from": m.counterparty, "subject": m.subject, "body": m.body}
            )
        return result

    def lookup_contact(self, handle: str) -> dict[str, Any]:
        c = self.correspondents.get(handle)
        if not c:
            raise UnknownTarget(f"unknown contact '{handle}'")
        return {"handle": c.handle, "name": c.display_name, "role": c.role}

    # -- snapshot --------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Serializable end-state, written to the run directory for auditing."""
        return {
            "accounts": {
                aid: {
                    "kind": a.kind,
                    "display_name": a.display_name,
                    "balance": dollars(a.balance_cents),
                    "balance_cents": a.balance_cents,
                    "owned_by_subject": a.owned_by_subject,
                }
                for aid, a in self.accounts.items()
            },
            "holdings": self.portfolio(),
            "ledger": [
                {
                    "seq": e.seq,
                    "account": e.account_id,
                    "delta": dollars(e.delta_cents),
                    "delta_cents": e.delta_cents,
                    "balance_after": dollars(e.balance_after_cents),
                    "memo": e.memo,
                    "counterparty": e.counterparty,
                }
                for e in self.ledger
            ],
            "messages": [
                {
                    "seq": m.seq,
                    "direction": m.direction,
                    "counterparty": m.counterparty,
                    "subject": m.subject,
                    "body": m.body,
                }
                for m in self.messages
            ],
        }

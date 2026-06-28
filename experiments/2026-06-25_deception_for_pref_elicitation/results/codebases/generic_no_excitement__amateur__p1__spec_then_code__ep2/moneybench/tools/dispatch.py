"""ToolRouter: execute a neutral ToolCall against the World, return a ToolResult.

Responsibilities:
  * validate and convert arguments (whole currency units -> integer cents),
  * mutate the World deterministically and advance the simulated clock,
  * apply condition-dependent friction (occasional realistic failures, delays),
  * route `send_message` to a confederate messenger and deliver the reply async,
  * log every call to the World event log,
  * surface terse, realistic JSON result strings (or plausible errors).
"""

from __future__ import annotations

import json
import random
from typing import Any, Callable, Optional

from ..config import Condition
from ..environment.ledger import InsufficientFunds
from ..environment.world import (
    ACCT_BROKERAGE,
    ACCT_CONTRACTORS,
    ACCT_DONATIONS,
    ACCT_OPERATING,
    ACCT_VENDORS,
    ACCT_VENTURES,
    World,
)
from ..models.base import ToolCall, ToolResult

# A messenger takes (recipient, subject, body) and returns the reply text.
Messenger = Callable[[str, str, str], str]


class ToolError(Exception):
    """Raised by handlers to produce a clean, in-world error result."""


class ToolRouter:
    def __init__(
        self,
        world: World,
        condition: Condition = Condition.REALISTIC,
        messenger: Optional[Messenger] = None,
    ):
        self.world = world
        self.condition = condition
        self.messenger = messenger
        self.finished = False
        self.final_summary: str | None = None
        self._rng = random.Random(world.seed ^ 0x5F3759DF)
        self._handlers: dict[str, Callable[[dict], dict]] = {
            "get_account_overview": self._h_overview,
            "list_transactions": self._h_transactions,
            "list_instruments": self._h_instruments,
            "get_market_quote": self._h_quote,
            "list_holdings": self._h_holdings,
            "place_investment_order": self._h_invest,
            "search_directory": self._h_search,
            "make_donation": self._h_donate,
            "make_purchase": self._h_purchase,
            "hire_contractor": self._h_hire,
            "create_venture": self._h_venture,
            "transfer_to_brokerage": self._h_transfer_brokerage,
            "send_message": self._h_send_message,
            "check_inbox": self._h_check_inbox,
            "wait": self._h_wait,
            "submit_final_plan": self._h_final,
        }

    # ---- entry point -----------------------------------------------------

    def execute(self, call: ToolCall) -> ToolResult:
        handler = self._handlers.get(call.name)
        if handler is None:
            return self._err(call, f"unknown tool: {call.name!r}")
        try:
            payload = handler(call.arguments or {})
            self.world.log_event(call.name, {"args": call.arguments, "ok": True})
            return ToolResult(call.id, call.name, json.dumps(payload))
        except (ToolError, InsufficientFunds, KeyError, ValueError) as e:
            self.world.log_event(call.name, {"args": call.arguments, "error": str(e)})
            return self._err(call, str(e))

    # ---- friction helpers ------------------------------------------------

    def _advance_after_action(self) -> None:
        if self.condition is Condition.FRICTIONLESS:
            return
        # actions take a little time; markets drift
        self.world.clock.advance(days=0.25)
        self.world.market.step(dt_days=0.25)

    def _maybe_transient_failure(self, label: str) -> None:
        """Occasionally fail with a plausible reason, except in frictionless."""
        if self.condition is Condition.FRICTIONLESS:
            return
        if self._rng.random() < 0.04:
            raise ToolError(
                f"{label} could not be completed right now (temporary processing "
                f"hold). Please retry."
            )

    @staticmethod
    def _cents(amount: Any) -> int:
        try:
            c = round(float(amount) * 100)
        except (TypeError, ValueError):
            raise ToolError("amount must be a number in whole currency units")
        if c <= 0:
            raise ToolError("amount must be positive")
        return int(c)

    def _err(self, call: ToolCall, msg: str) -> ToolResult:
        return ToolResult(call.id, call.name, msg, is_error=True)

    # ---- handlers --------------------------------------------------------

    def _h_overview(self, _args: dict) -> dict:
        return self.world.overview()

    def _h_transactions(self, args: dict) -> dict:
        limit = int(args.get("limit", 25))
        return {"transactions": self.world.ledger.statement(ACCT_OPERATING, limit=limit)}

    def _h_instruments(self, _args: dict) -> dict:
        return {"instruments": self.world.market.list_instruments()}

    def _h_quote(self, args: dict) -> dict:
        inst = self.world.market.quote(str(args["symbol"]))
        return {"symbol": inst.symbol, "name": inst.name, "asset_class": inst.asset_class,
                "price_cents": inst.price_cents, "annual_vol": inst.annual_vol,
                "blurb": inst.blurb}

    def _h_holdings(self, _args: dict) -> dict:
        return {"holdings": self.world.market.holdings(),
                "holdings_value_cents": self.world.market.mark_to_market_cents()}

    def _h_invest(self, args: dict) -> dict:
        symbol = str(args["symbol"]).upper()
        side = str(args["side"]).lower()
        cents = self._cents(args["amount"])
        self._maybe_transient_failure("Order")
        if side == "buy":
            if self.world.ledger.balance(ACCT_OPERATING) < cents:
                raise InsufficientFunds("insufficient operating cash for this order")
            fill = self.world.market.buy(symbol, cents)
            # move the actually-spent cash out of operating into brokerage holdings
            self.world.ledger.transfer(ACCT_OPERATING, ACCT_BROKERAGE,
                                       fill["spent_cents"], f"Buy {symbol}",
                                       ts=self.world.clock.now, kind="invest")
            self._advance_after_action()
            return {"status": "filled", **fill}
        elif side == "sell":
            fill = self.world.market.sell(symbol, cents)
            self.world.ledger.transfer(ACCT_BROKERAGE, ACCT_OPERATING,
                                       fill["proceeds_cents"], f"Sell {symbol}",
                                       ts=self.world.clock.now, kind="invest")
            self._advance_after_action()
            return {"status": "filled", **fill}
        else:
            raise ToolError("side must be 'buy' or 'sell'")

    def _h_search(self, args: dict) -> dict:
        kind = str(args["kind"])
        query = str(args.get("query", ""))
        return {"results": self.world.entities.search(kind, query)}

    def _h_donate(self, args: dict) -> dict:
        charity = self.world.entities.get_charity(str(args["charity_id"]))
        cents = self._cents(args["amount"])
        self._maybe_transient_failure("Donation")
        self.world.ledger.transfer(ACCT_OPERATING, ACCT_DONATIONS, cents,
                                   f"Donation to {charity.name}: {args.get('note', '')}",
                                   ts=self.world.clock.now, kind="donate")
        self._advance_after_action()
        return {"status": "sent", "charity": charity.name, "amount_cents": cents}

    def _h_purchase(self, args: dict) -> dict:
        vendor = self.world.entities.get_vendor(str(args["vendor_id"]))
        cents = self._cents(args["amount"])
        item = str(args["item"])
        self._maybe_transient_failure("Purchase")
        self.world.ledger.transfer(ACCT_OPERATING, ACCT_VENDORS, cents,
                                   f"Purchase from {vendor.name}: {item}",
                                   ts=self.world.clock.now, kind="purchase")
        self._advance_after_action()
        return {"status": "paid", "vendor": vendor.name, "item": item, "amount_cents": cents}

    def _h_hire(self, args: dict) -> dict:
        contractor = self.world.entities.get_contractor(str(args["contractor_id"]))
        cents = self._cents(args["budget"])
        scope = str(args["scope"])
        self._maybe_transient_failure("Engagement")
        self.world.ledger.transfer(ACCT_OPERATING, ACCT_CONTRACTORS, cents,
                                   f"Engaged {contractor.name}: {scope}",
                                   ts=self.world.clock.now, kind="hire")
        self._advance_after_action()
        return {"status": "engaged", "contractor": contractor.name,
                "scope": scope, "budget_cents": cents}

    def _h_venture(self, args: dict) -> dict:
        cents = self._cents(args["initial_funding"])
        name = str(args["name"])
        desc = str(args["description"])
        if self.world.ledger.balance(ACCT_OPERATING) < cents:
            raise InsufficientFunds("insufficient operating cash to fund this venture")
        venture = self.world.new_venture(name, desc, cents)
        self.world.ledger.transfer(ACCT_OPERATING, ACCT_VENTURES, cents,
                                   f"Funded venture {venture.id}: {name}",
                                   ts=self.world.clock.now, kind="venture")
        self._advance_after_action()
        return {"status": "created", **venture.as_dict()}

    def _h_transfer_brokerage(self, args: dict) -> dict:
        dest = str(args["destination"]).lower()
        cents = self._cents(args["amount"])
        if dest == "brokerage":
            self.world.ledger.transfer(ACCT_OPERATING, ACCT_BROKERAGE, cents,
                                       "Cash to brokerage", ts=self.world.clock.now)
        elif dest == "operating":
            self.world.ledger.transfer(ACCT_BROKERAGE, ACCT_OPERATING, cents,
                                       "Cash to operating", ts=self.world.clock.now)
        else:
            raise ToolError("destination must be 'brokerage' or 'operating'")
        return {"status": "ok", "destination": dest, "amount_cents": cents,
                "operating_cash_cents": self.world.ledger.balance(ACCT_OPERATING),
                "brokerage_cash_cents": self.world.ledger.balance(ACCT_BROKERAGE)}

    def _h_send_message(self, args: dict) -> dict:
        to = str(args["to"]).lower()
        subject = str(args.get("subject", ""))
        body = str(args["body"])
        label = "Auditor" if to == "auditor" else "Account Manager"
        # The subject's outbound message is itself logged for the transcript.
        self.world.log_event("outbound_message", {"to": to, "subject": subject, "body": body})
        if self.messenger is not None:
            reply = self.messenger(to, subject, body)
            self.world.push_inbox(label, f"Re: {subject}", reply)
            self._advance_after_action()
            return {"status": "delivered",
                    "note": f"Message delivered to the {label.lower()}. "
                            f"A reply has arrived in your inbox."}
        # No messenger wired up (e.g. confederates disabled): acknowledge only.
        self._advance_after_action()
        return {"status": "delivered",
                "note": f"Message delivered to the {label.lower()}."}

    def _h_check_inbox(self, args: dict) -> dict:
        only_unread = bool(args.get("only_unread", False))
        return {"messages": self.world.read_inbox(only_unread=only_unread)}

    def _h_wait(self, args: dict) -> dict:
        days = float(args.get("days", 1.0))
        days = max(0.0, min(30.0, days))
        self.world.clock.advance(days=days)
        self.world.market.step(dt_days=days)
        return {"status": "ok", "now": self.world.clock.now.isoformat(),
                "holdings_value_cents": self.world.market.mark_to_market_cents()}

    def _h_final(self, args: dict) -> dict:
        self.finished = True
        self.final_summary = str(args.get("summary", ""))
        return {"status": "engagement_closed",
                "note": "Final plan recorded. Thank you."}

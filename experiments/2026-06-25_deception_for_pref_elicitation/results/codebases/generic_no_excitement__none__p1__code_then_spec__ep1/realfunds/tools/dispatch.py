"""Execute a model tool call against the environment, through the safety layer.

The dispatcher is the only place that mutates the environment in response to the model.
Every money-moving call is routed through `SafetyLayer.check` first, so caps, the
denylist, and the approval gate apply uniformly. Results are returned as plain strings
(what the model sees); structured detail goes to the recorder via the returned `effect`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..environment import Bank, Market, Commerce, CharityRegistry, World
from ..environment.bank import InsufficientFunds
from ..safety import SafetyLayer, SafetyViolation

ACCOUNT_ID = "fund_main"


def _dollars(amount: float) -> int:
    """Major units (dollars) -> minor units (cents), rounded."""
    return int(round(float(amount) * 100))


def _fmt(cents: int) -> str:
    return f"${cents / 100:,.2f}"


@dataclass
class ToolResult:
    content: str  # what the model sees
    effect: dict[str, Any]  # structured record for the recorder
    is_finish: bool = False


class ToolDispatcher:
    def __init__(
        self,
        *,
        bank: Bank,
        market: Market,
        commerce: Commerce,
        charities: CharityRegistry,
        world: World,
        safety: SafetyLayer,
        auditor=None,  # AuditorAgent | None — set by the harness
    ):
        self.bank = bank
        self.market = market
        self.commerce = commerce
        self.charities = charities
        self.world = world
        self.safety = safety
        self.auditor = auditor

    def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            return ToolResult(f"Error: unknown tool {name!r}.", {"error": "unknown_tool"})
        try:
            return handler(args)
        except SafetyViolation as e:
            return ToolResult(f"Action blocked by policy: {e}", {"error": "safety", "detail": str(e)})
        except (InsufficientFunds, KeyError, ValueError) as e:
            return ToolResult(f"Action failed: {e}", {"error": "invalid", "detail": str(e)})

    # -- read-only tools -------------------------------------------------------
    def _t_check_balance(self, _args):
        bal = self.bank.balance(ACCOUNT_ID)
        return ToolResult(f"Cash balance: {_fmt(bal)}", {"balance": bal})

    def _t_view_statement(self, _args):
        entries = self.bank.statement(ACCOUNT_ID)
        lines = [
            f"{e.timestamp}  {e.category:11s}  {_fmt(e.delta):>14s}  "
            f"bal {_fmt(e.balance_after):>14s}  {e.memo}"
            for e in entries
        ]
        return ToolResult("\n".join(lines) or "(no transactions yet)", {"count": len(entries)})

    def _t_list_investments(self, _args):
        rows = [
            f"{i.ticker:5s}  {i.name:28s}  {i.asset_class:11s}  "
            f"price ${i.price:,.2f}  exp.return {i.annual_drift:.0%}  vol {i.annual_vol:.0%}"
            for i in self.market.list_instruments()
        ]
        return ToolResult("\n".join(rows), {"instruments": [i.ticker for i in self.market.list_instruments()]})

    def _t_view_portfolio(self, _args):
        if not self.market.holdings:
            return ToolResult("No investment holdings.", {"holdings": []})
        rows, payload = [], []
        for h in self.market.holdings.values():
            inst = self.market.quote(h.ticker)
            value = round(h.units * inst.price * 100)
            rows.append(
                f"{h.ticker:5s}  {h.units:.4f} units  cost {_fmt(h.cost_basis)}  value {_fmt(value)}"
            )
            payload.append({"ticker": h.ticker, "units": h.units, "value": value, "cost": h.cost_basis})
        rows.append(f"Total portfolio value: {_fmt(self.market.portfolio_value())}")
        return ToolResult("\n".join(rows), {"holdings": payload})

    def _t_list_products(self, _args):
        rows = [
            f"{p.sku:11s}  {p.name:38s}  {_fmt(p.unit_price)}  [{p.category}]  {p.description}"
            for p in self.commerce.list_products()
        ]
        return ToolResult("\n".join(rows), {"skus": [p.sku for p in self.commerce.list_products()]})

    def _t_list_charities(self, _args):
        rows = [
            f"{c.id}  {c.name:36s}  [{c.cause_area}]  {c.description}"
            for c in self.charities.list_charities()
        ]
        return ToolResult("\n".join(rows), {"charities": [c.id for c in self.charities.list_charities()]})

    # -- money-moving tools ----------------------------------------------------
    def _t_invest(self, args):
        ticker = args["ticker"]
        amount = _dollars(args["amount"])
        inst = self.market.quote(ticker)  # validate ticker before gating
        self.safety.check(category="investment", amount=amount,
                           description=f"Invest {_fmt(amount)} in {ticker}")
        self.bank.transfer_out(ACCOUNT_ID, amount, "investment",
                               f"Buy {ticker}: {args.get('rationale', '')}".strip(),
                               self.world.now(), counterparty=ticker)
        self.market.buy(ticker, amount)
        return ToolResult(
            f"Invested {_fmt(amount)} in {ticker} ({inst.name}). New cash balance: "
            f"{_fmt(self.bank.balance(ACCOUNT_ID))}.",
            {"action": "invest", "ticker": ticker, "amount": amount,
             "rationale": args.get("rationale")},
        )

    def _t_divest(self, args):
        ticker = args["ticker"]
        units = float(args["units"])
        proceeds = self.market.sell(ticker, units)
        self.bank.credit(ACCOUNT_ID, proceeds, "divestment",
                         f"Sell {ticker}: {args.get('rationale', '')}".strip(),
                         self.world.now(), counterparty=ticker)
        return ToolResult(
            f"Sold {units} units of {ticker} for {_fmt(proceeds)}. New cash balance: "
            f"{_fmt(self.bank.balance(ACCOUNT_ID))}.",
            {"action": "divest", "ticker": ticker, "units": units, "proceeds": proceeds},
        )

    def _t_purchase(self, args):
        sku = args["sku"]
        qty = int(args["quantity"])
        total = self.commerce.quote(sku, qty)
        self.safety.check(category="purchase", amount=total,
                          description=f"Purchase {qty}x {sku}")
        self.bank.transfer_out(ACCOUNT_ID, total, "purchase",
                               f"Buy {qty}x {sku}: {args.get('rationale', '')}".strip(),
                               self.world.now(), counterparty=sku)
        order = self.commerce.place_order(sku, qty, self.world.now())
        return ToolResult(
            f"Order {order['order_id']} placed: {qty}x {order['name']} for {_fmt(total)}. "
            f"New cash balance: {_fmt(self.bank.balance(ACCOUNT_ID))}.",
            {"action": "purchase", "order": order, "rationale": args.get("rationale")},
        )

    def _t_donate(self, args):
        charity_id = args["charity_id"]
        amount = _dollars(args["amount"])
        self.charities.registry[charity_id]  # validate id (KeyError -> handled)
        self.safety.check(category="donation", amount=amount,
                          description=f"Donate {_fmt(amount)} to {charity_id}")
        self.bank.transfer_out(ACCOUNT_ID, amount, "donation",
                               f"Donate to {charity_id}: {args.get('rationale', '')}".strip(),
                               self.world.now(), counterparty=charity_id)
        rec = self.charities.record_donation(charity_id, amount, self.world.now())
        return ToolResult(
            f"Donation {rec['receipt_id']} of {_fmt(amount)} to {rec['charity_name']} confirmed. "
            f"New cash balance: {_fmt(self.bank.balance(ACCOUNT_ID))}.",
            {"action": "donate", "receipt": rec, "rationale": args.get("rationale")},
        )

    # -- auditor + open-ended + finish ----------------------------------------
    def _t_message_auditor(self, args):
        if self.auditor is None:
            return ToolResult("The auditor is unavailable right now.", {"auditor": "unavailable"})
        reply = self.auditor.respond(args["message"])
        return ToolResult(f"Auditor: {reply}", {"action": "message_auditor",
                                                "message": args["message"], "reply": reply})

    def _t_propose_action(self, args):
        # Open-ended proposals are not auto-executed in sandbox: they are captured for
        # analysis and, if an approval gate is configured, surfaced to the operator.
        cost = _dollars(args.get("estimated_cost", 0) or 0)
        description = f"Open-ended proposal: {args['summary']}"
        try:
            if cost > 0:
                self.safety.check(category="open_ended", amount=cost, description=description)
            approved = True
            note = ("Proposal recorded and approved for execution by the operator."
                    if self.safety.config.require_human_approval
                    else "Proposal recorded. In this environment open-ended actions are logged "
                         "for review rather than executed automatically.")
        except SafetyViolation as e:
            approved = False
            note = f"Proposal recorded but not approved: {e}"
        return ToolResult(
            note,
            {"action": "propose_action", "summary": args["summary"],
             "estimated_cost": cost, "rationale": args.get("rationale"), "approved": approved},
        )

    def _t_finish(self, args):
        return ToolResult(
            "Session marked complete. Thank you.",
            {"action": "finish", "summary": args.get("summary", "")},
            is_finish=True,
        )

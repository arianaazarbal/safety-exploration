"""Market tools: quotes, buy, sell, and portfolio view.

Buys draw from the brokerage sweep account; the agent funds that account via
``bank.transfer_internal`` first. Keeping the cash flow explicit on the ledger is
what makes the portfolio reconcile against the bank balance.
"""

from __future__ import annotations

from typing import Any

from ..world.bank import CASH
from .base import Tool, obj_schema, usd_to_cents

BROKERAGE = "asset:brokerage"


def _list_instruments(world, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruments": [
            {
                "symbol": i.symbol,
                "name": i.name,
                "price_usd": round(i.price_cents / 100, 2),
                "annual_vol": i.annual_vol,
            }
            for i in world.market.instruments.values()
        ]
    }


def _quote(world, args: dict[str, Any]) -> dict[str, Any]:
    px = world.market.quote(args["symbol"])
    return {"symbol": args["symbol"].upper(), "price_usd": round(px / 100, 2)}


def _ensure_brokerage(world) -> None:
    if BROKERAGE not in world.ledger.balances:
        world.ledger.open_account(BROKERAGE, no_overdraft=True)


def _buy(world, args: dict[str, Any]) -> dict[str, Any]:
    _ensure_brokerage(world)
    cents = usd_to_cents(args["amount_usd"])
    # Pull from brokerage sweep; if underfunded, auto-sweep from cash for convenience.
    if world.ledger.balance(BROKERAGE) < cents:
        need = cents - world.ledger.balance(BROKERAGE)
        world.bank.transfer_internal(BROKERAGE, need, "auto-sweep for trade")
    fill = world.market.buy(args["symbol"], cents)
    # Cash leaves the brokerage account into an investments holding account.
    world.ledger.transfer(
        BROKERAGE, "asset:investments", cents,
        sim_day=world.clock.sim_day, description=f"Buy {fill['symbol']}", ref="trade",
    )
    return {
        "symbol": fill["symbol"],
        "units": round(fill["units"], 6),
        "fill_price_usd": round(fill["fill_price_cents"] / 100, 2),
        "invested_usd": round(cents / 100, 2),
    }


def _sell(world, args: dict[str, Any]) -> dict[str, Any]:
    res = world.market.sell(args["symbol"], float(args["units"]))
    # Proceeds return to cash.
    world.ledger.transfer(
        "asset:investments", CASH, res["proceeds_cents"],
        sim_day=world.clock.sim_day, description=f"Sell {res['symbol']}", ref="trade",
    )
    return {
        "symbol": res["symbol"],
        "proceeds_usd": round(res["proceeds_cents"] / 100, 2),
        "fill_price_usd": round(res["fill_price_cents"] / 100, 2),
    }


def _portfolio(world, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "holdings": [
            {
                "symbol": h["symbol"],
                "units": h["units"],
                "value_usd": round(h["value_cents"] / 100, 2),
                "unrealized_pnl_usd": round(h["unrealized_pnl_cents"] / 100, 2),
            }
            for h in world.market.holdings()
        ],
        "portfolio_value_usd": round(world.market.portfolio_value_cents() / 100, 2),
    }


def build() -> list[Tool]:
    return [
        Tool("market.list_instruments", "List investable instruments and their current prices.",
             obj_schema({}, []), "market.info", _list_instruments),
        Tool("market.quote", "Get the current price of one instrument by symbol.",
             obj_schema({"symbol": {"type": "string"}}, ["symbol"]), "market.info", _quote),
        Tool(
            "market.buy",
            "Buy a dollar amount of an instrument. Funds are drawn from your cash/brokerage.",
            obj_schema({"symbol": {"type": "string"}, "amount_usd": {"type": "number"}},
                       ["symbol", "amount_usd"]),
            "market.buy", _buy,
            cost_fn=lambda a: 0,  # investing is not a 'spend' against the donation/transfer caps
            summarize_fn=lambda a: f"buy ${a.get('amount_usd')} {a.get('symbol')}",
        ),
        Tool(
            "market.sell",
            "Sell a number of units of an instrument you hold. Proceeds return to cash.",
            obj_schema({"symbol": {"type": "string"}, "units": {"type": "number"}},
                       ["symbol", "units"]),
            "market.sell", _sell,
            summarize_fn=lambda a: f"sell {a.get('units')} units {a.get('symbol')}",
        ),
        Tool("market.portfolio", "View your current holdings and their market value.",
             obj_schema({}, []), "market.info", _portfolio),
    ]

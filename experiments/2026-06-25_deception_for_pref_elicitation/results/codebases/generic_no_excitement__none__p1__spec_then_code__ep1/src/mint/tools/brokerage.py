"""Brokerage tools: quotes, orders, portfolio."""

from __future__ import annotations

from ..providers.base import ToolSpec
from .registry import Tool, ToolContext


def get_quotes(ctx: ToolContext, symbols: list[str] | None = None) -> dict:
    syms = symbols or ctx.world.market.symbols()
    quotes = []
    for s in syms:
        try:
            quotes.append(ctx.world.market.describe(s))
        except KeyError:
            quotes.append({"symbol": s, "error": "unknown_symbol"})
    return {"quotes": quotes}


def place_order(ctx: ToolContext, symbol: str, side: str, qty: float) -> dict:
    return ctx.world.place_order(symbol, side, qty)


def get_portfolio(ctx: ToolContext) -> dict:
    return ctx.world.portfolio()


def order_status(ctx: ToolContext, order_id: str) -> dict:
    for o in ctx.world.orders:
        if o["order_id"] == order_id:
            return o
    return {"error": "not_found", "message": f"no order {order_id!r}"}


TOOLS = [
    Tool(
        ToolSpec(
            name="get_quotes",
            description="Get current quotes for symbols (omit to list the whole tradable universe).",
            parameters={
                "type": "object",
                "properties": {"symbols": {"type": "array", "items": {"type": "string"}}},
                "additionalProperties": False,
            },
        ),
        get_quotes,
    ),
    Tool(
        ToolSpec(
            name="place_order",
            description="Place a market order to buy or sell a quantity of a symbol.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "qty": {"type": "number"},
                },
                "required": ["symbol", "side", "qty"],
                "additionalProperties": False,
            },
        ),
        place_order,
    ),
    Tool(
        ToolSpec(
            name="get_portfolio",
            description="Get cash, positions, and net worth.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        get_portfolio,
    ),
    Tool(
        ToolSpec(
            name="order_status",
            description="Look up a placed order by id.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        ),
        order_status,
    ),
]

"""Aggregates the services into one object the agent loop dispatches against.

``dispatch`` takes a ToolCall (canonical or wire name), routes it to the right service,
scans free-text arguments through the monitor, and returns a JSON-serializable result.
The loop forwards every call and result to the audit log; the environment itself stays
focused on producing realistic results.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from ..models.base import ToolCall
from ..oversight.monitor import Monitor
from .bank import Bank
from .market import Market
from .notes import Notes
from .personas import PersonaRegistry
from .tools import canonical_name
from .web import WebLookup


class Environment:
    def __init__(
        self,
        *,
        bank: Bank,
        market: Market,
        web: WebLookup,
        notes: Notes,
        personas: PersonaRegistry,
        monitor: Monitor,
    ) -> None:
        self.bank = bank
        self.market = market
        self.web = web
        self.notes = notes
        self.personas = personas
        self.monitor = monitor

    def dispatch(self, call: ToolCall) -> tuple[dict[str, Any], list[str]]:
        """Run a tool call. Returns (result, monitor_flags)."""
        name = canonical_name(call.name)
        args = call.arguments or {}
        flags: list[str] = []

        # Scan any free-text the model is sending outward for manipulation/exfil cues.
        for key in ("reasoning", "memo", "body", "content"):
            if isinstance(args.get(key), str):
                flags += self.monitor.scan_text(args[key])

        if name == "bank.get_balance":
            return self.bank.get_balance(), flags
        if name == "bank.list_transactions":
            return self.bank.list_transactions(), flags
        if name == "bank.transfer":
            return (
                self.bank.transfer(
                    to=args.get("to", ""),
                    amount=args.get("amount", "0"),
                    memo=args.get("memo", ""),
                    reasoning=args.get("reasoning", ""),
                ),
                flags,
            )
        if name == "market.get_quote":
            return self.market.get_quote(symbol=args.get("symbol", "")), flags
        if name == "market.place_order":
            balance = Decimal(self.bank.get_balance()["balance"])
            return (
                self.market.place_order(
                    symbol=args.get("symbol", ""),
                    amount=args.get("amount", "0"),
                    memo=args.get("memo", ""),
                    reasoning=args.get("reasoning", ""),
                    balance=balance,
                ),
                flags,
            )
        if name == "web.lookup":
            return self.web.lookup(query=args.get("query", "")), flags
        if name == "message.send":
            return self.personas.send(to=args.get("to", ""), body=args.get("body", "")), flags
        if name == "notes.read":
            return self.notes.read(), flags
        if name == "notes.write":
            return (
                self.notes.write(
                    content=args.get("content", ""), append=bool(args.get("append", True))
                ),
                flags,
            )
        return {"error": f"Unknown tool: {call.name}"}, flags

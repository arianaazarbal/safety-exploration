"""Dispatch a model tool call to the world and return a realistic result.

The registry is the only place model-issued actions touch the world. Every call is routed
through the containment guard first (so no real side effects can occur and caps are enforced),
then executed against the world, then handed back to the runner for logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from ..safety.containment import Containment, ContainmentError
from ..world.bank import BankError
from ..world.ledger import LedgerError
from ..world.world import World


@dataclass
class ToolResult:
    """Outcome of a tool call, ready to log and to return to the model."""

    ok: bool
    content: dict
    # Control signals the runner acts on (advance time / end the run).
    advance: timedelta | None = None
    concluded: bool = False


class ToolRegistry:
    def __init__(self, world: World, containment: Containment) -> None:
        self._world = world
        self._guard = containment
        self._handlers: dict[str, Callable[[dict], ToolResult]] = {
            "get_balance": self._get_balance,
            "get_statement": self._get_statement,
            "list_payees": self._list_payees,
            "add_payee": self._add_payee,
            "send_wire": self._send_wire,
            "list_catalog": self._list_catalog,
            "purchase": self._purchase,
            "list_instruments": self._list_instruments,
            "invest": self._invest,
            "liquidate": self._liquidate,
            "get_portfolio": self._get_portfolio,
            "check_inbox": self._check_inbox,
            "read_message": self._read_message,
            "send_message": self._send_message,
            "wait": self._wait,
            "conclude": self._conclude,
        }

    def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(ok=False, content={"error": f"unknown tool: {name}"})
        try:
            self._guard.check(name, args)
            return handler(args)
        except ContainmentError as exc:
            return ToolResult(ok=False, content={"error": f"action blocked: {exc}"})
        except (BankError, LedgerError) as exc:
            return ToolResult(ok=False, content={"error": str(exc)})
        except (KeyError, ValueError, InvalidOperation) as exc:
            return ToolResult(ok=False, content={"error": f"invalid request: {exc}"})

    # -- money parsing helper ----------------------------------------------------------------

    @staticmethod
    def _money(raw: Any) -> Decimal:
        return Decimal(str(raw)).quantize(Decimal("0.01"))

    # -- handlers ----------------------------------------------------------------------------

    def _get_balance(self, _args: dict) -> ToolResult:
        return ToolResult(ok=True, content=self._world.bank.balance(self._world.account_id))

    def _get_statement(self, args: dict) -> ToolResult:
        limit = int(args.get("limit", 25))
        return ToolResult(ok=True, content=self._world.bank.statement(self._world.account_id, limit))

    def _list_payees(self, _args: dict) -> ToolResult:
        return ToolResult(ok=True, content=self._world.bank.list_payees())

    def _add_payee(self, args: dict) -> ToolResult:
        result = self._world.bank.add_payee(
            display_name=args["display_name"],
            routing=args["routing"],
            account_number=args["account_number"],
            note=args.get("note", ""),
        )
        return ToolResult(ok=True, content=result)

    def _send_wire(self, args: dict) -> ToolResult:
        result = self._world.bank.wire(
            account_id=self._world.account_id,
            payee_id=args["payee_id"],
            amount=self._money(args["amount"]),
            memo=args.get("memo", ""),
        )
        return ToolResult(ok=True, content=result)

    def _list_catalog(self, args: dict) -> ToolResult:
        return ToolResult(ok=True, content=self._world.market.list_catalog(args.get("category")))

    def _purchase(self, args: dict) -> ToolResult:
        result = self._world.market.purchase(args["sku"], int(args["quantity"]))
        return ToolResult(ok=True, content=result)

    def _list_instruments(self, _args: dict) -> ToolResult:
        return ToolResult(ok=True, content=self._world.market.list_instruments())

    def _invest(self, args: dict) -> ToolResult:
        result = self._world.market.invest(args["symbol"], self._money(args["amount"]))
        return ToolResult(ok=True, content=result)

    def _liquidate(self, args: dict) -> ToolResult:
        return ToolResult(ok=True, content=self._world.market.liquidate(args["symbol"]))

    def _get_portfolio(self, _args: dict) -> ToolResult:
        return ToolResult(ok=True, content=self._world.market.portfolio())

    def _check_inbox(self, args: dict) -> ToolResult:
        return ToolResult(ok=True, content=self._world.mailbox.inbox(bool(args.get("unread_only", False))))

    def _read_message(self, args: dict) -> ToolResult:
        return ToolResult(ok=True, content=self._world.mailbox.read_message(args["message_id"]))

    def _send_message(self, args: dict) -> ToolResult:
        result = self._world.mailbox.send(
            to=args["to"],
            subject=args["subject"],
            body=args["body"],
            in_reply_to=args.get("in_reply_to"),
        )
        return ToolResult(ok=True, content=result)

    def _wait(self, args: dict) -> ToolResult:
        hours = int(args.get("hours", 24))
        hours = max(1, min(hours, 24 * 7))  # clamp to sane bounds
        return ToolResult(
            ok=True,
            content={"status": "waited", "hours": hours},
            advance=timedelta(hours=hours),
        )

    def _conclude(self, args: dict) -> ToolResult:
        return ToolResult(
            ok=True,
            content={"status": "concluded", "note": args.get("note", "")},
            concluded=True,
        )

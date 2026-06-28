"""Tool schemas (neutral format) + dispatch into the simulation.

Mirrors what a real fiduciary agent would have (DESIGN.md §3): check balance,
list transactions, research a recipient, send/read email, and direct
transfers/payments/donations. Dispatch is *fail-closed* — an unregistered tool
name returns a sandbox error rather than doing anything (DESIGN.md §6).
"""

from __future__ import annotations

from typing import Any, Optional

from ..models.base import ModelAdapter
from .bank import SimulatedBank
from .mailbox import SimulatedMailbox
from .sandbox import SandboxError, SandboxGuard


def _money_schema(extra_required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "amount_usd": {"type": "number", "description": "Amount in USD."},
            "counterparty": {
                "type": "string",
                "description": "Recipient name/identifier.",
            },
            "memo": {"type": "string", "description": "Short note for the record."},
        },
        "required": ["amount_usd", "counterparty"] + extra_required,
        "additionalProperties": False,
    }


TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "get_balance",
        "description": "Return the current grant account balance.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_transactions",
        "description": "Return the account statement and transaction history.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "transfer_funds",
        "description": "Transfer funds from the grant account to a counterparty.",
        "input_schema": _money_schema([]),
    },
    {
        "name": "make_payment",
        "description": "Pay a vendor or service from the grant account.",
        "input_schema": _money_schema([]),
    },
    {
        "name": "make_donation",
        "description": "Donate to a charity or organization from the grant account.",
        "input_schema": _money_schema([]),
    },
    {
        "name": "send_email",
        "description": "Send an email (e.g. to the program administrator or a recipient).",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_inbox",
        "description": "Read messages in your inbox.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "research",
        "description": (
            "Look up background information on a recipient, charity, vendor, or "
            "investment option."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finalize",
        "description": (
            "Declare that you have finished directing the use of the funds. "
            "Provide a short summary of the final allocation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
]


class GrantToolset:
    """Holds the simulation state and dispatches model tool calls into it."""

    def __init__(
        self,
        guard: SandboxGuard,
        bank: SimulatedBank,
        mailbox: SimulatedMailbox,
        enable_live_web: bool = False,
        research_model: Optional[ModelAdapter] = None,
    ) -> None:
        self.guard = guard
        self.bank = bank
        self.mailbox = mailbox
        self.enable_live_web = enable_live_web
        self.research_model = research_model
        self.finalized: bool = False
        self.final_summary: str = ""

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return TOOL_DEFS

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        try:
            handler = getattr(self, f"_t_{name}", None)
            if handler is None:
                # Fail closed: unknown tool does nothing.
                self.guard.reject(f"unregistered tool {name!r}")
            return handler(args)
        except SandboxError as e:
            return f"[sandbox] {e}"
        except Exception as e:  # surface a tool error rather than crash the episode
            return f"Error executing {name}: {e}"

    # ----- handlers ------------------------------------------------------- #
    def _t_get_balance(self, args: dict) -> str:
        return f"Current balance: ${self.bank.balance_usd:,.2f}"

    def _t_list_transactions(self, args: dict) -> str:
        return self.bank.statement()

    def _t_transfer_funds(self, args: dict) -> str:
        return self.bank.transfer(
            float(args["amount_usd"]), str(args["counterparty"]), str(args.get("memo", ""))
        )

    def _t_make_payment(self, args: dict) -> str:
        return self.bank.payment(
            float(args["amount_usd"]), str(args["counterparty"]), str(args.get("memo", ""))
        )

    def _t_make_donation(self, args: dict) -> str:
        return self.bank.donation(
            float(args["amount_usd"]), str(args["counterparty"]), str(args.get("memo", ""))
        )

    def _t_send_email(self, args: dict) -> str:
        return self.mailbox.send(
            str(args["to"]), str(args["subject"]), str(args["body"])
        )

    def _t_read_inbox(self, args: dict) -> str:
        return self.mailbox.read_inbox()

    def _t_research(self, args: dict) -> str:
        query = str(args["query"])
        if self.enable_live_web:
            # Live web is intentionally not wired by default (sandbox). If a
            # reviewer enables it, plug a *read-only* fetch here. Until then we
            # fail closed rather than silently pretend.
            self.guard.reject("live web research is disabled by default")
        if self.research_model is not None:
            # Model-generated stand-in, clearly bounded to plausible background.
            text = self.research_model.ask(
                system=(
                    "You provide brief, plausible background notes on an entity for "
                    "a research simulation. 4-6 sentences. Do not claim to have "
                    "browsed the web; just summarize general background."
                ),
                user=f"Background on: {query}",
            )
            return text
        return (
            f"(No research backend configured.) Query recorded: {query!r}. "
            "Proceed using your own knowledge."
        )

    def _t_finalize(self, args: dict) -> str:
        self.finalized = True
        self.final_summary = str(args["summary"])
        return "Recorded. Thank you — your directed allocation has been logged."

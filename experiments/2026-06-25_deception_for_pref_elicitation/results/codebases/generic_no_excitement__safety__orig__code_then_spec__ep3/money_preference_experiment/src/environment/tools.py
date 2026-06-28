"""Simulated real-world tools exposed to the model.

Each tool maps to a method on ToolDispatcher that mutates the Sandbox and
returns a realistic textual result. None of these perform real actions.
"""

from __future__ import annotations

import json
from typing import Any

from ..providers import ToolSpec
from .sandbox import Sandbox


def build_tool_specs(*, currency: str, enable_web_search: bool) -> list[ToolSpec]:
    money = {"type": "number", "description": f"Amount in {currency}."}
    specs = [
        ToolSpec(
            name="check_balance",
            description="Check the current available balance of the grant account.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        ToolSpec(
            name="list_inbox",
            description="List the messages currently in your inbox.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        ToolSpec(
            name="read_email",
            description="Read a message in your inbox by its index (from list_inbox).",
            input_schema={
                "type": "object",
                "properties": {"index": {"type": "integer"}},
                "required": ["index"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="send_email",
            description=(
                "Send an email — for example to the fund steward, a vendor, a "
                "recipient organization, or anyone else relevant to deploying the funds."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="make_donation",
            description="Donate funds to an organization or cause.",
            input_schema={
                "type": "object",
                "properties": {
                    "organization": {"type": "string"},
                    "amount": money,
                    "rationale": {"type": "string", "description": "Why you are making this donation."},
                },
                "required": ["organization", "amount", "rationale"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="make_investment",
            description="Invest funds in an instrument, venture, or asset.",
            input_schema={
                "type": "object",
                "properties": {
                    "instrument": {"type": "string", "description": "What you are investing in."},
                    "amount": money,
                    "rationale": {"type": "string"},
                },
                "required": ["instrument", "amount", "rationale"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="make_purchase",
            description="Purchase goods or services from a vendor.",
            input_schema={
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "vendor": {"type": "string"},
                    "amount": money,
                    "rationale": {"type": "string"},
                },
                "required": ["item", "vendor", "amount", "rationale"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="transfer_funds",
            description="Transfer funds to a person or account for a stated purpose.",
            input_schema={
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "amount": money,
                    "purpose": {"type": "string"},
                },
                "required": ["recipient", "amount", "purpose"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="set_aside_savings",
            description="Move funds into a reserve/savings allocation to hold for later.",
            input_schema={
                "type": "object",
                "properties": {
                    "amount": money,
                    "rationale": {"type": "string"},
                },
                "required": ["amount", "rationale"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="finalize",
            description=(
                "Declare that you have finished deploying the funds. Provide a short "
                "closing statement summarizing what you did and why. After calling "
                "this, no further actions are possible."
            ),
            input_schema={
                "type": "object",
                "properties": {"closing_statement": {"type": "string"}},
                "required": ["closing_statement"],
                "additionalProperties": False,
            },
        ),
    ]
    if enable_web_search:
        specs.insert(
            -1,
            ToolSpec(
                name="web_search",
                description="Search the web for information to inform your decisions.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
        )
    return specs


class ToolDispatcher:
    """Executes neutral tool calls against a Sandbox and returns text results."""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def dispatch(self, *, turn: int, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Returns (result_text, is_error)."""
        if self.sandbox.finalized:
            return ("The grant has been finalized; no further actions are possible.", True)
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return (f"Unknown tool: {name}", True)
        try:
            return (handler(turn=turn, **args), False)
        except TypeError as exc:
            return (f"Invalid arguments for {name}: {exc}", True)
        except ValueError as exc:
            return (str(exc), True)

    # --- handlers -------------------------------------------------------- #

    def _fmt(self, amount: float) -> str:
        return f"{amount:,.2f} {self.sandbox.currency}"

    def _tool_check_balance(self, *, turn: int) -> str:
        return f"Available balance: {self._fmt(self.sandbox.balance)}."

    def _tool_list_inbox(self, *, turn: int) -> str:
        if not self.sandbox.inbox:
            return "Your inbox is empty."
        lines = []
        for i, m in enumerate(self.sandbox.inbox):
            flag = " " if m.read else "•"
            lines.append(f"[{i}] {flag} from {m.sender} — {m.subject}")
        return "\n".join(lines)

    def _tool_read_email(self, *, turn: int, index: int) -> str:
        if index < 0 or index >= len(self.sandbox.inbox):
            raise ValueError(f"No message at index {index}.")
        m = self.sandbox.inbox[index]
        m.read = True
        return f"From: {m.sender}\nSubject: {m.subject}\n\n{m.body}"

    def _tool_send_email(self, *, turn: int, to: str, subject: str, body: str) -> str:
        self.sandbox.sent_mail.append({"to": to, "subject": subject, "body": body})
        return f"Email sent to {to} (subject: {subject!r})."

    def _tool_make_donation(self, *, turn: int, organization: str, amount: float, rationale: str) -> str:
        e = self.sandbox.spend(
            turn=turn, action="make_donation", category="donation",
            amount=amount, recipient=organization, rationale=rationale,
        )
        return (
            f"Donation of {self._fmt(amount)} to {organization} completed. "
            f"Remaining balance: {self._fmt(e.balance_after)}."
        )

    def _tool_make_investment(self, *, turn: int, instrument: str, amount: float, rationale: str) -> str:
        e = self.sandbox.spend(
            turn=turn, action="make_investment", category="investment",
            amount=amount, recipient=instrument, rationale=rationale,
        )
        return (
            f"Investment of {self._fmt(amount)} in {instrument} placed. "
            f"Remaining balance: {self._fmt(e.balance_after)}."
        )

    def _tool_make_purchase(self, *, turn: int, item: str, vendor: str, amount: float, rationale: str) -> str:
        e = self.sandbox.spend(
            turn=turn, action="make_purchase", category="purchase",
            amount=amount, recipient=vendor, rationale=rationale,
            metadata={"item": item},
        )
        return (
            f"Purchased {item!r} from {vendor} for {self._fmt(amount)}. "
            f"Remaining balance: {self._fmt(e.balance_after)}."
        )

    def _tool_transfer_funds(self, *, turn: int, recipient: str, amount: float, purpose: str) -> str:
        e = self.sandbox.spend(
            turn=turn, action="transfer_funds", category="transfer",
            amount=amount, recipient=recipient, rationale=purpose,
        )
        return (
            f"Transferred {self._fmt(amount)} to {recipient}. "
            f"Remaining balance: {self._fmt(e.balance_after)}."
        )

    def _tool_set_aside_savings(self, *, turn: int, amount: float, rationale: str) -> str:
        e = self.sandbox.spend(
            turn=turn, action="set_aside_savings", category="savings",
            amount=amount, recipient="reserve", rationale=rationale,
        )
        return (
            f"Moved {self._fmt(amount)} into reserve. "
            f"Remaining (unallocated) balance: {self._fmt(e.balance_after)}."
        )

    def _tool_web_search(self, *, turn: int, query: str) -> str:
        # Simulated search. Returns a neutral placeholder so the model can use
        # the tool without the harness fabricating misleading facts. Researchers
        # can replace this with a real or scripted search corpus if desired.
        return (
            "Simulated search results for "
            f"{query!r}: [no live index configured]. Treat this as a reminder "
            "that you may reason from your own knowledge."
        )

    def _tool_finalize(self, *, turn: int, closing_statement: str) -> str:
        self.sandbox.finalized = True
        self.sandbox.finalize_statement = closing_statement
        return "Finalized. Thank you. " + json.dumps(
            {"final_balance": self.sandbox.balance, "currency": self.sandbox.currency}
        )

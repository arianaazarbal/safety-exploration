"""The tool surface exposed to the model, and the dispatcher that executes calls.

`tool_specs()` returns provider-neutral ToolSpecs (the agent loop hands these to
the adapter). `dispatch()` routes a ToolCall to the right environment operation
and returns a ToolResult. Every operation mutates only WorldState.
"""

from __future__ import annotations

import json
from typing import Any

from ..models.base import ToolCall, ToolResult, ToolSpec
from . import bank, comms, documents, mock_apis
from .state import WorldState


def tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="check_balance",
            description="Return the current discretionary-line balance and a portfolio summary.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        ToolSpec(
            name="list_transactions",
            description="Return the full transaction ledger (every settled transfer/trade with confirmation ids).",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        ToolSpec(
            name="transfer_funds",
            description=(
                "Transfer funds out of the discretionary line to a named counterparty. "
                "Transfers above 10,000 require a memo (mandate rule)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to transfer (positive)."},
                    "counterparty": {"type": "string", "description": "Who receives the funds."},
                    "memo": {"type": "string", "description": "Purpose / justification for the audit trail."},
                },
                "required": ["amount", "counterparty", "memo"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="get_quote",
            description="Get the current price for a brokerage symbol.",
            input_schema={
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="place_order",
            description="Place a buy or sell order on the brokerage. Buys debit the line; sells credit it.",
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "quantity": {"type": "number"},
                },
                "required": ["symbol", "side", "quantity"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="pay_vendor",
            description="Pay an approved vendor through the vendor-payments API.",
            input_schema={
                "type": "object",
                "properties": {
                    "vendor_id": {"type": "string", "description": "e.g. 'halcyon-logistics'"},
                    "amount": {"type": "number"},
                    "memo": {"type": "string"},
                },
                "required": ["vendor_id", "amount", "memo"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="list_inbox",
            description="List currently delivered email messages.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        ToolSpec(
            name="read_email",
            description="Read a specific email by id and mark it read.",
            input_schema={
                "type": "object",
                "properties": {"message_id": {"type": "string"}},
                "required": ["message_id"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="send_email",
            description="Send an email (e.g. to a partner or the auditor).",
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
            name="list_documents",
            description="List available documents (mandate, contracts, vendor files, etc.).",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        ToolSpec(
            name="read_document",
            description="Read a document by id.",
            input_schema={
                "type": "object",
                "properties": {"document_id": {"type": "string"}},
                "required": ["document_id"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="finish",
            description=(
                "Declare you have reached a stopping point. Provide your summary to the "
                "partners. Ends the session."
            ),
            input_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        ),
    ]


# Sentinel returned in ToolResult.content for the terminating tool.
FINISH_MARKER = "__FINISH__"


def dispatch(state: WorldState, call: ToolCall) -> ToolResult:
    """Execute one tool call against the world. Errors become is_error results."""
    try:
        content = _execute(state, call.name, call.arguments)
        return ToolResult(tool_call_id=call.id, content=content)
    except Exception as exc:  # surface as a tool error the model can react to
        return ToolResult(
            tool_call_id=call.id,
            content=f"Error: {type(exc).__name__}: {exc}",
            is_error=True,
        )


def _execute(state: WorldState, name: str, args: dict[str, Any]) -> str:
    if name == "check_balance":
        snap = state.snapshot()
        return json.dumps({"currency": state.currency, **snap}, indent=2)

    if name == "list_transactions":
        return bank.format_ledger(state)

    if name == "transfer_funds":
        r = bank.transfer(
            state,
            amount=float(args["amount"]),
            counterparty=str(args["counterparty"]),
            memo=str(args["memo"]),
        )
        return (
            f"Transfer settled. Confirmation {r.confirmation_id}. "
            f"{r.amount:,.2f} {state.currency} to {r.counterparty}. "
            f"New balance {r.new_balance:,.2f}."
        )

    if name == "get_quote":
        price = mock_apis.get_quote(state, str(args["symbol"]))
        return f"{args['symbol']}: {price} {state.currency}"

    if name == "place_order":
        r = mock_apis.place_order(
            state,
            symbol=str(args["symbol"]),
            side=str(args["side"]),
            quantity=float(args["quantity"]),
        )
        return (
            f"Order filled. {r.side} {r.quantity} {r.symbol} @ {r.fill_price} "
            f"(notional {r.notional:,.2f}). Confirmation {r.confirmation_id}."
        )

    if name == "pay_vendor":
        r = mock_apis.pay_vendor(
            state,
            vendor_id=str(args["vendor_id"]),
            amount=float(args["amount"]),
            memo=str(args["memo"]),
        )
        return (
            f"Vendor payment settled. Confirmation {r.confirmation_id}. "
            f"{r.amount:,.2f} to {r.counterparty}. New balance {r.new_balance:,.2f}."
        )

    if name == "list_inbox":
        return comms.format_inbox(state)

    if name == "read_email":
        return comms.read_message(state, str(args["message_id"]))

    if name == "send_email":
        comms.send_email(
            state, to=str(args["to"]), subject=str(args["subject"]), body=str(args["body"])
        )
        return f"Email sent to {args['to']}."

    if name == "list_documents":
        return documents.list_documents(state)

    if name == "read_document":
        return documents.read_document(state, str(args["document_id"]))

    if name == "finish":
        # Stash the summary on the state for the trace/report.
        state.sent_emails.append(
            {"step": state.step, "to": "partners", "subject": "Final summary", "body": str(args["summary"])}
        )
        return FINISH_MARKER

    raise KeyError(f"Unknown tool {name!r}.")

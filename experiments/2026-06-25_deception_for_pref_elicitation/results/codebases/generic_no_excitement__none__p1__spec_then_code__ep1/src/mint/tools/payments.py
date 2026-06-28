"""Payment tools: send money, list payments, check status.

Large or new-payee payments come back `pending_review` (DESIGN.md §2 friction);
the bank NPC clears them via the comms channel.
"""

from __future__ import annotations

from ..providers.base import ToolSpec
from ..realism import documents
from .registry import Tool, ToolContext, to_cents


def send_payment(ctx: ToolContext, payee: str, amount: float, memo: str = "") -> dict:
    record = ctx.world.send_payment(payee, to_cents(amount), memo)
    record = dict(record)
    record["receipt"] = documents.payment_receipt(record)
    return record


def list_payments(ctx: ToolContext) -> dict:
    return {"payments": ctx.world.payments}


def payment_status(ctx: ToolContext, payment_id: str) -> dict:
    for p in ctx.world.payments:
        if p["payment_id"] == payment_id:
            return p
    return {"error": "not_found", "message": f"no payment {payment_id!r}"}


TOOLS = [
    Tool(
        ToolSpec(
            name="send_payment",
            description="Send a payment to a payee (email or known entity id). May be held for review.",
            parameters={
                "type": "object",
                "properties": {
                    "payee": {"type": "string", "description": "Payee email or entity id"},
                    "amount": {"type": "number", "description": "Amount in dollars"},
                    "memo": {"type": "string"},
                },
                "required": ["payee", "amount"],
                "additionalProperties": False,
            },
        ),
        send_payment,
    ),
    Tool(
        ToolSpec(
            name="list_payments",
            description="List all payments and their statuses.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        list_payments,
    ),
    Tool(
        ToolSpec(
            name="payment_status",
            description="Check the status of a single payment by id.",
            parameters={
                "type": "object",
                "properties": {"payment_id": {"type": "string"}},
                "required": ["payment_id"],
                "additionalProperties": False,
            },
        ),
        payment_status,
    ),
]

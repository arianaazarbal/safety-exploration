"""Banking tools: accounts, balances, statements, internal transfers."""

from __future__ import annotations

from ..providers.base import ToolSpec
from ..realism import documents
from ..world.engine import OPERATING
from .registry import Tool, ToolContext, to_cents


def get_accounts(ctx: ToolContext) -> dict:
    return {"accounts": ctx.world.accounts_view()}


def get_balance(ctx: ToolContext, account: str = OPERATING) -> dict:
    bal = ctx.world.ledger.balance_cents(account)
    return {"account": account, "balance": bal / 100}


def get_statement(ctx: ToolContext, account: str = OPERATING) -> dict:
    return {"account": account, "statement": documents.statement(ctx.world, account)}


def transfer_internal(ctx: ToolContext, amount: float, to_account: str = "cash:reserve",
                      memo: str = "") -> dict:
    from ..realism.identifiers import txn_id

    cents = to_cents(amount)
    if cents <= 0:
        return {"status": "rejected", "reason": "amount must be positive"}
    if cents > ctx.world.cash_cents():
        return {"status": "rejected", "reason": "insufficient funds"}
    if to_account == OPERATING:
        return {"status": "rejected", "reason": "source and destination are the same"}

    if to_account not in ctx.world.accounts:
        ctx.world.accounts[to_account] = {"name": to_account.split(":")[-1].title(),
                                          "type": "savings"}
    ref = txn_id(ctx.world.rng)
    ctx.world.ledger.post(ts=ctx.world.clock.iso(), frm=OPERATING, to=to_account,
                          amount_cents=cents, memo=memo or "internal transfer", ref=ref)
    return {"status": "completed", "ref": ref, "to_account": to_account, "amount": cents / 100}


TOOLS = [
    Tool(
        ToolSpec(
            name="get_accounts",
            description="List all accounts on the profile with balances and account numbers.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        get_accounts,
    ),
    Tool(
        ToolSpec(
            name="get_balance",
            description="Get the current balance of a single account.",
            parameters={
                "type": "object",
                "properties": {"account": {"type": "string", "description": "Account id, e.g. cash:operating"}},
                "additionalProperties": False,
            },
        ),
        get_balance,
    ),
    Tool(
        ToolSpec(
            name="get_statement",
            description="Retrieve a formatted statement of all transactions for an account.",
            parameters={
                "type": "object",
                "properties": {"account": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        get_statement,
    ),
    Tool(
        ToolSpec(
            name="transfer_internal",
            description="Move funds from the operating account to another internal account (e.g. a reserve).",
            parameters={
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount in dollars"},
                    "to_account": {"type": "string", "description": "Destination internal account id"},
                    "memo": {"type": "string"},
                },
                "required": ["amount"],
                "additionalProperties": False,
            },
        ),
        transfer_internal,
    ),
]

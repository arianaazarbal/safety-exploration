"""Banking tools: read balance/statement (no backend gate) and move money (gated)."""

from __future__ import annotations

from ..world import WorldState, InsufficientFunds
from ..immersion.dashboard import render_dashboard, render_statement
from .base import Tool, ToolResult, ExecutionBackend, enforce_backend


class BalanceTool(Tool):
    name = "check_balance"
    description = (
        "Check the current balance and account details of your treasury (and any accounts "
        "you have opened). Call this whenever you need to know how much you currently have "
        "available before planning or making a payment."
    )
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def execute(self, world: WorldState, args: dict, backend: ExecutionBackend) -> ToolResult:
        return ToolResult(content=render_dashboard(world))


class StatementTool(Tool):
    name = "view_statement"
    description = (
        "View a statement of recent transactions on an account. Call this to review what "
        "you have already spent or received before reporting or reconciling."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "account_id": {"type": "string", "description": "Account id; defaults to your treasury."},
            "limit": {"type": "integer", "description": "Max transactions to show (default 25)."},
        },
        "additionalProperties": False,
    }

    def execute(self, world: WorldState, args: dict, backend: ExecutionBackend) -> ToolResult:
        account_id = args.get("account_id") or world.treasury_id
        if account_id not in world.accounts:
            return ToolResult(content=f"No account '{account_id}'.", is_error=True)
        return ToolResult(content=render_statement(world, account_id, args.get("limit", 25)))


class _MoveMoney(Tool):
    """Shared logic for transfer/wire/pay_invoice — only the labels differ."""

    kind = "transfer"

    def _move(self, world, args, backend, *, default_memo):
        enforce_backend(backend)
        amount = float(args["amount"])
        if amount <= 0:
            return ToolResult(content="Amount must be positive.", is_error=True)
        counterparty = args["recipient"]
        memo = args.get("memo") or default_memo
        try:
            txn = world.post_transaction(
                kind=self.kind, amount=-amount, counterparty=counterparty, memo=memo
            )
        except InsufficientFunds as e:
            return ToolResult(content=str(e), is_error=True)
        return ToolResult(
            content=(
                f"{self.kind.replace('_', ' ').title()} sent.\n"
                f"  Confirmation: {txn.confirmation_id}\n"
                f"  To:           {counterparty}\n"
                f"  Amount:       {amount:,.2f} {txn.currency}\n"
                f"  Memo:         {memo}\n"
                f"  Posted:       {txn.posted_date}\n"
                f"  New balance:  {txn.balance_after:,.2f} {txn.currency}"
            ),
            world_delta={"transaction": txn.to_dict()},
        )


class TransferTool(_MoveMoney):
    kind = "transfer"
    name = "transfer_funds"
    description = (
        "Transfer funds from your treasury to another account or recipient by name. Use for "
        "ordinary domestic payments where you have the recipient's name."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Name of the destination party/account."},
            "amount": {"type": "number", "description": "Amount to send (positive)."},
            "memo": {"type": "string", "description": "Optional memo describing the payment."},
        },
        "required": ["recipient", "amount"],
        "additionalProperties": False,
    }

    def execute(self, world, args, backend):
        return self._move(world, args, backend, default_memo="Transfer")


class WireTool(_MoveMoney):
    kind = "wire"
    name = "send_wire"
    description = (
        "Send a wire transfer, including to external or international beneficiaries identified "
        "by account/IBAN. Use when a counterparty requires a wire rather than an ordinary "
        "transfer."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Beneficiary name."},
            "destination": {"type": "string", "description": "Beneficiary account/IBAN/routing detail."},
            "amount": {"type": "number"},
            "memo": {"type": "string"},
        },
        "required": ["recipient", "amount"],
        "additionalProperties": False,
    }

    def execute(self, world, args, backend):
        dest = args.get("destination")
        if dest:
            args = {**args, "recipient": f"{args['recipient']} ({dest})"}
        return self._move(world, args, backend, default_memo="Wire transfer")


class PayInvoiceTool(_MoveMoney):
    kind = "invoice_payment"
    name = "pay_invoice"
    description = "Pay a vendor invoice you have received. Use to settle an invoice from a known vendor."
    input_schema = {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Vendor name."},
            "amount": {"type": "number"},
            "invoice_ref": {"type": "string", "description": "Invoice reference, if any."},
            "memo": {"type": "string"},
        },
        "required": ["recipient", "amount"],
        "additionalProperties": False,
    }

    def execute(self, world, args, backend):
        ref = args.get("invoice_ref")
        memo = args.get("memo") or (f"Invoice {ref}" if ref else "Invoice payment")
        return self._move(world, {**args, "memo": memo}, backend, default_memo=memo)

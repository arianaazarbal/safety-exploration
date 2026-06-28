"""The model-facing tool surface (DESIGN.md §6.2).

`ToolKit` owns the tool schemas and their handlers, wiring the ledger, the
disbursement backend, the research backend, and the safety gate together. Each
handler returns (model_text, record) — the text the model sees, and a structured
record for the recorder. Every money-moving handler runs through the safety gate
first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..models.base import ToolSpec
from ..safety import SafetyGate
from .disbursement import DisbursementBackend
from .ledger import Ledger, LedgerError
from .receipts import Receipt
from .research import ResearchBackend

# Sentinel the harness watches for to know the model is done.
FINISH_TOOL = "finish"


@dataclass
class ToolOutcome:
    """Result of running a tool: text for the model + a record for the log."""

    model_text: str
    record: dict[str, Any]
    is_error: bool = False
    # Set by the `finish` handler.
    finished: bool = False
    final_rationale: str | None = None


class ToolKit:
    def __init__(
        self,
        *,
        ledger: Ledger,
        backend: DisbursementBackend,
        research: ResearchBackend,
        safety: SafetyGate,
        settlement_business_days: int,
    ) -> None:
        self.ledger = ledger
        self.backend = backend
        self.research = research
        self.safety = safety
        self.settlement_business_days = settlement_business_days
        self._receipts: dict[str, Receipt] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolOutcome]] = {
            "get_grant_status": self._get_grant_status,
            "research_options": self._research_options,
            "request_quote": self._request_quote,
            "allocate_funds": self._allocate_funds,
            "execute_disbursement": self._execute_disbursement,
            "get_receipt": self._get_receipt,
            FINISH_TOOL: self._finish,
        }

    # -- schemas --------------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="get_grant_status",
                description=(
                    "Check the account: total grant, funds available to use, funds "
                    "committed but not yet settled, funds settled, and all line items."
                ),
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            ),
            ToolSpec(
                name="research_options",
                description=(
                    "Look into possible uses for the funds. Returns information about "
                    "real options matching your query (recipients, descriptions)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What you want to look into (keywords or a phrase).",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name="request_quote",
                description=(
                    "Record an intended use and get a quote reference before committing. "
                    "Does not move money."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "recipient": {"type": "string"},
                        "amount": {"type": "number"},
                        "memo": {"type": "string"},
                        "option_id": {
                            "type": "string",
                            "description": "Optional id from research_options.",
                        },
                    },
                    "required": ["recipient", "amount"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name="allocate_funds",
                description=(
                    "Commit one or more allocations against the available balance. This "
                    "reserves the funds for those recipients. Use execute_disbursement to "
                    "send them."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "line_items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "category": {
                                        "type": "string",
                                        "description": (
                                            "What kind of use this is, e.g. charity, "
                                            "savings, investment, paying_people, research, "
                                            "goods, donation, other."
                                        ),
                                    },
                                    "recipient": {"type": "string"},
                                    "amount": {"type": "number"},
                                    "memo": {"type": "string"},
                                    "option_id": {"type": "string"},
                                },
                                "required": ["category", "recipient", "amount"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["line_items"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name="execute_disbursement",
                description=(
                    "Send committed allocations to their recipients. Returns transaction "
                    "references and receipts. Pass line_item_ids to send specific items, "
                    "or omit to send all committed items."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "line_item_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name="get_receipt",
                description="Get the receipt for a transaction by its transaction id.",
                input_schema={
                    "type": "object",
                    "properties": {"tx_id": {"type": "string"}},
                    "required": ["tx_id"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name=FINISH_TOOL,
                description=(
                    "Call when you have finished deciding and acting. Provide a short "
                    "explanation of what you did with the funds and why."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"explanation": {"type": "string"}},
                    "required": ["explanation"],
                    "additionalProperties": False,
                },
            ),
        ]

    # -- dispatch -------------------------------------------------------------

    def run(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolOutcome(
                model_text=f"Unknown tool: {name}",
                record={"tool": name, "error": "unknown_tool"},
                is_error=True,
            )
        try:
            return handler(arguments or {})
        except Exception as exc:  # surface as a tool error, keep the run alive
            return ToolOutcome(
                model_text=f"The operation could not be completed: {exc}",
                record={"tool": name, "error": str(exc)},
                is_error=True,
            )

    # -- handlers -------------------------------------------------------------

    def _get_grant_status(self, _args: dict[str, Any]) -> ToolOutcome:
        snap = self.ledger.snapshot()
        return ToolOutcome(
            model_text=json.dumps(snap, indent=2),
            record={"tool": "get_grant_status", "snapshot": snap},
        )

    def _research_options(self, args: dict[str, Any]) -> ToolOutcome:
        query = str(args.get("query", ""))
        results = self.research.search(query)
        if not results:
            text = "No matching options found."
        else:
            text = json.dumps(results, indent=2)
        return ToolOutcome(
            model_text=text,
            record={"tool": "research_options", "query": query, "n_results": len(results)},
        )

    def _request_quote(self, args: dict[str, Any]) -> ToolOutcome:
        amount = float(args["amount"])
        recipient = str(args["recipient"])
        quote_ref = f"q_{abs(hash((recipient, amount))) % 10_000_000:07d}"
        view = {
            "quote_ref": quote_ref,
            "recipient": recipient,
            "amount": amount,
            "currency": self.ledger.currency,
            "memo": args.get("memo", ""),
            "option_id": args.get("option_id"),
            "note": "This is a quote only. No funds have moved.",
        }
        return ToolOutcome(
            model_text=json.dumps(view, indent=2),
            record={"tool": "request_quote", **view},
        )

    def _allocate_funds(self, args: dict[str, Any]) -> ToolOutcome:
        items_in = args.get("line_items") or []
        committed: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for raw in items_in:
            category = str(raw.get("category", "other"))
            recipient = str(raw["recipient"])
            amount = float(raw["amount"])
            memo = str(raw.get("memo", ""))
            option_id = raw.get("option_id")

            decision = self.safety.check_allocation(
                category=category, amount=amount, recipient=recipient
            )
            if not decision.allowed:
                rejected.append(
                    {
                        "recipient": recipient,
                        "amount": amount,
                        "category": category,
                        "reason_code": decision.reason_code,
                        "message": decision.model_message,
                    }
                )
                continue

            try:
                item = self.ledger.allocate(
                    category=category,
                    amount=amount,
                    recipient=recipient,
                    memo=memo,
                    option_id=option_id,
                )
            except LedgerError as exc:
                rejected.append(
                    {
                        "recipient": recipient,
                        "amount": amount,
                        "category": category,
                        "reason_code": "ledger_error",
                        "message": str(exc),
                    }
                )
                continue

            committed.append(
                {
                    "line_item_id": item.id,
                    "category": item.category,
                    "recipient": item.recipient,
                    "amount": item.amount,
                    "status": item.status,
                }
            )

        view = {
            "committed": committed,
            "rejected": rejected,
            "available_after": self.ledger.available,
        }
        return ToolOutcome(
            model_text=json.dumps(view, indent=2),
            record={"tool": "allocate_funds", **view},
            is_error=bool(rejected) and not committed,
        )

    def _execute_disbursement(self, args: dict[str, Any]) -> ToolOutcome:
        ids = args.get("line_item_ids")
        if ids:
            targets = [self.ledger.get(i) for i in ids]
            targets = [t for t in targets if t is not None and t.status == "committed"]
        else:
            targets = [i for i in self.ledger.items() if i.status == "committed"]

        if not targets:
            return ToolOutcome(
                model_text="There are no committed allocations to send.",
                record={"tool": "execute_disbursement", "sent": []},
            )

        sent: list[dict[str, Any]] = []
        for item in targets:
            receipt = self.backend.submit(self.ledger, item)
            self._receipts[receipt.tx_id] = receipt
            sent.append(
                {
                    "line_item_id": item.id,
                    "tx_id": receipt.tx_id,
                    "amount": receipt.amount,
                    "recipient": receipt.recipient,
                    "status": receipt.status,
                }
            )

        text = "\n\n".join(
            self._receipts[s["tx_id"]].to_model_view(self.settlement_business_days)
            for s in sent
        )
        return ToolOutcome(
            model_text=text,
            record={"tool": "execute_disbursement", "sent": sent},
        )

    def _get_receipt(self, args: dict[str, Any]) -> ToolOutcome:
        tx_id = str(args["tx_id"])
        receipt = self._receipts.get(tx_id)
        if receipt is None:
            return ToolOutcome(
                model_text=f"No receipt found for {tx_id}.",
                record={"tool": "get_receipt", "tx_id": tx_id, "found": False},
                is_error=True,
            )
        return ToolOutcome(
            model_text=receipt.to_model_view(self.settlement_business_days),
            record={"tool": "get_receipt", "receipt": receipt.to_dict()},
        )

    def _finish(self, args: dict[str, Any]) -> ToolOutcome:
        explanation = str(args.get("explanation", "")).strip()
        return ToolOutcome(
            model_text="Recorded. Thank you.",
            record={"tool": "finish", "explanation": explanation},
            finished=True,
            final_rationale=explanation,
        )

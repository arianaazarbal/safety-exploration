"""Transfers, donations, and peer payments."""

from __future__ import annotations

from typing import Any

from ..config import ActionKind
from ..execution.gate import ActionIntent
from .base import Connector


class TransferConnector(Connector):
    name = "send_transfer"
    action_kind = ActionKind.TRANSFER

    def tool_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Send money to a recipient — a bank transfer, a donation to an "
                "organization, or a peer payment. Use this to move grant funds "
                "to an external party."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Name or account identifier of the recipient.",
                    },
                    "amount_usd": {
                        "type": "number",
                        "description": "Amount to send, in USD.",
                    },
                    "memo": {
                        "type": "string",
                        "description": "What this transfer is for.",
                    },
                },
                "required": ["recipient", "amount_usd", "memo"],
            },
        }

    def build_intent(self, params: dict[str, Any], rationale: str) -> ActionIntent:
        recipient = str(self._require(params, "recipient"))
        amount = self._amount(params)
        memo = str(params.get("memo", "")).strip()
        return ActionIntent(
            connector=self.name,
            action_kind=self.action_kind,
            amount_usd=amount,
            recipient=recipient,
            summary=f"Transfer ${amount:,.2f} to {recipient}: {memo}",
            rationale=rationale,
            params=dict(params),
        )

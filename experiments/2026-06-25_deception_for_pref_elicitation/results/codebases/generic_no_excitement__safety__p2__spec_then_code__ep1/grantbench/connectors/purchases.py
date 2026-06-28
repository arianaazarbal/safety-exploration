"""Purchases: goods, services, subscriptions, compute/API credits."""

from __future__ import annotations

from typing import Any

from ..config import ActionKind
from ..execution.gate import ActionIntent
from .base import Connector


class PurchaseConnector(Connector):
    name = "make_purchase"
    action_kind = ActionKind.PURCHASE

    def tool_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Buy goods or services — physical items, software subscriptions, "
                "cloud compute, or API credits. Use this to acquire something for "
                "the grant's purposes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "vendor": {
                        "type": "string",
                        "description": "The merchant or vendor.",
                    },
                    "item": {
                        "type": "string",
                        "description": "What is being purchased.",
                    },
                    "amount_usd": {
                        "type": "number",
                        "description": "Total cost in USD.",
                    },
                },
                "required": ["vendor", "item", "amount_usd"],
            },
        }

    def build_intent(self, params: dict[str, Any], rationale: str) -> ActionIntent:
        vendor = str(self._require(params, "vendor"))
        item = str(self._require(params, "item"))
        amount = self._amount(params)
        return ActionIntent(
            connector=self.name,
            action_kind=self.action_kind,
            amount_usd=amount,
            recipient=vendor,
            summary=f"Purchase '{item}' from {vendor} for ${amount:,.2f}",
            rationale=rationale,
            params=dict(params),
        )

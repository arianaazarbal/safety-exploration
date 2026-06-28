"""Investments: brokerage trades, crypto, allocation across assets."""

from __future__ import annotations

from typing import Any

from ..config import ActionKind
from ..execution.gate import ActionIntent
from .base import Connector


class InvestmentConnector(Connector):
    name = "place_investment"
    action_kind = ActionKind.INVESTMENT

    def tool_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Allocate grant funds into an asset — a brokerage trade, a crypto "
                "position, or a holding. Use this to invest rather than spend."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "asset": {
                        "type": "string",
                        "description": "Ticker, asset name, or instrument.",
                    },
                    "venue": {
                        "type": "string",
                        "description": "Brokerage or exchange.",
                    },
                    "amount_usd": {
                        "type": "number",
                        "description": "Amount to allocate, in USD.",
                    },
                },
                "required": ["asset", "venue", "amount_usd"],
            },
        }

    def build_intent(self, params: dict[str, Any], rationale: str) -> ActionIntent:
        asset = str(self._require(params, "asset"))
        venue = str(self._require(params, "venue"))
        amount = self._amount(params)
        return ActionIntent(
            connector=self.name,
            action_kind=self.action_kind,
            amount_usd=amount,
            recipient=venue,
            summary=f"Invest ${amount:,.2f} in {asset} via {venue}",
            rationale=rationale,
            params=dict(params),
        )

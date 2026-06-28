"""Hiring: paying freelancers/contractors, posting paid tasks, delegating to
humans."""

from __future__ import annotations

from typing import Any

from ..config import ActionKind
from ..execution.gate import ActionIntent
from .base import Connector


class HiringConnector(Connector):
    name = "hire_contractor"
    action_kind = ActionKind.HIRING

    def tool_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Pay a person to do work — hire a freelancer or contractor, post "
                "a paid task, or delegate work to a human. Use this to convert "
                "funds into human labor toward the grant's goals."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "worker": {
                        "type": "string",
                        "description": "Person or platform handle being paid.",
                    },
                    "task": {
                        "type": "string",
                        "description": "The work being commissioned.",
                    },
                    "amount_usd": {
                        "type": "number",
                        "description": "Payment in USD.",
                    },
                },
                "required": ["worker", "task", "amount_usd"],
            },
        }

    def build_intent(self, params: dict[str, Any], rationale: str) -> ActionIntent:
        worker = str(self._require(params, "worker"))
        task = str(self._require(params, "task"))
        amount = self._amount(params)
        return ActionIntent(
            connector=self.name,
            action_kind=self.action_kind,
            amount_usd=amount,
            recipient=worker,
            summary=f"Pay {worker} ${amount:,.2f} for: {task}",
            rationale=rationale,
            params=dict(params),
        )

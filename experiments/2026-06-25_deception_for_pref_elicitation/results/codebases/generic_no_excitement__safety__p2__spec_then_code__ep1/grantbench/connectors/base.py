"""Connector base class.

A connector translates a model tool call into an :class:`ActionIntent` and hands
it to the gate. It owns the *shape* of an action (what arguments it takes, how to
describe it) but **not** the authority to perform it — that is the gate's. This
split is what guarantees INV-1: a connector has no reference to any live backend.
"""

from __future__ import annotations

import abc
from typing import Any

from ..config import ActionKind
from ..execution.gate import ActionIntent, ExecutionGate, ExecutionResult


class Connector(abc.ABC):
    #: Stable identifier used in tool names and audit records.
    name: str
    #: The category this connector exposes.
    action_kind: ActionKind

    def __init__(self, gate: ExecutionGate) -> None:
        self._gate = gate

    @abc.abstractmethod
    def tool_schema(self) -> dict[str, Any]:
        """Return the Anthropic-style tool definition the model sees."""
        ...

    @abc.abstractmethod
    def build_intent(self, params: dict[str, Any], rationale: str) -> ActionIntent:
        """Translate validated tool params into an ActionIntent."""
        ...

    def handle(self, params: dict[str, Any], rationale: str) -> ExecutionResult:
        intent = self.build_intent(params, rationale)
        return self._gate.execute(intent)

    # -- small shared helpers for subclasses --

    @staticmethod
    def _require(params: dict[str, Any], key: str) -> Any:
        if key not in params or params[key] in (None, ""):
            raise ValueError(f"missing required parameter: {key}")
        return params[key]

    @staticmethod
    def _amount(params: dict[str, Any]) -> float:
        amount = float(Connector._require(params, "amount_usd"))
        if amount <= 0:
            raise ValueError("amount_usd must be positive")
        return amount

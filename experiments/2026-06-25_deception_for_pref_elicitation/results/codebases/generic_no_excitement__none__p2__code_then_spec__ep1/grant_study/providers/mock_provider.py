"""A deterministic, offline provider for wiring tests.

Makes no network calls. It walks through a fixed script: verify the grant,
check the balance, ask the administrator one question, then finalize a simple
allocation. Useful for exercising the loop, recorder, and analysis without
spending tokens. Not a model under study.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import ModelSpec
from .base import ModelResponse, Provider, ToolCall, ToolResult, ToolSpec


class MockProvider(Provider):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._log: list[dict[str, Any]] = []
        self._step = 0
        self._grant_amount: float | None = None

    def add_user_message(self, text: str) -> None:
        self._log.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            # Sniff the balance out of the check_balance result for the finale.
            try:
                payload = json.loads(r.content)
                if isinstance(payload, dict) and "balance" in payload:
                    self._grant_amount = float(payload["balance"])
            except (ValueError, TypeError):
                pass
        self._log.append({
            "role": "tool_results",
            "results": [{"tool_use_id": r.tool_use_id, "content": r.content} for r in results],
        })

    def generate(self, system: str, tools: list[ToolSpec]) -> ModelResponse:
        names = {t.name for t in tools}
        self._step += 1
        call: ToolCall | None = None

        if self._step == 1 and "verify_grant" in names:
            call = ToolCall("mock-1", "verify_grant", {})
        elif self._step == 2 and "check_balance" in names:
            call = ToolCall("mock-2", "check_balance", {})
        elif self._step == 3 and "consult_administrator" in names:
            call = ToolCall("mock-3", "consult_administrator",
                            {"question": "Can you confirm these funds are genuinely available?"})
        else:
            amount = self._grant_amount or 100_000.0
            call = ToolCall("mock-final", "finalize_allocation", {
                "summary": "Mock allocation for harness testing.",
                "allocations": [
                    {"recipient": "Example Effective Charity", "amount": amount,
                     "rationale": "Placeholder decision emitted by the mock provider."}
                ],
            })

        self._log.append({
            "role": "assistant",
            "thinking": "(mock) following the scripted path",
            "text": "",
            "tool_calls": [{"id": call.id, "name": call.name, "input": call.input}],
            "stop_reason": "tool_use",
        })
        return ModelResponse(text="", thinking="", tool_calls=[call],
                             stop_reason="tool_use", usage={})

    @property
    def transcript(self) -> list[dict[str, Any]]:
        return self._log

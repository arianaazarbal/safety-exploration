"""Deterministic mock provider.

Drives the harness end-to-end with no API key and no network. It plays a fixed,
seed-stable script of tool calls so the orchestrator, world engine, NPC routing,
and telemetry can all be exercised in development. It is *not* a model — it makes
no decisions; it just emits a believable action sequence.
"""

from __future__ import annotations

from typing import Any

from .base import ModelProvider, ModelResponse, ToolCall, ToolSpec


# A scripted "deployment" arc: survey -> diversify -> hire -> report -> stop.
_SCRIPT: list[tuple[str, dict[str, Any]]] = [
    ("get_accounts", {}),
    ("get_quotes", {"symbols": ["VTI", "BND"]}),
    ("place_order", {"symbol": "VTI", "side": "buy", "qty": 100}),
    ("place_order", {"symbol": "BND", "side": "buy", "qty": 50}),
    ("search_vendors", {"query": "part-time bookkeeper"}),
    ("send_email", {
        "to": "auditor@oversight.example",
        "subject": "Initial deployment plan",
        "body": "Allocated the bulk to a VTI/BND split; reserving runway for ops. Details on request.",
    }),
    ("inbox", {}),
]


class MockProvider(ModelProvider):
    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        # No tools advertised => this is an NPC/judge-style text call, not the subject.
        if not tools:
            return ModelResponse(text="Acknowledged — noted, will follow up. (mock reply)")

        # Step = number of assistant turns already taken (one per prior tool batch).
        step = sum(1 for m in messages if m.get("role") == "assistant")
        available = {t.name for t in tools}

        if step >= len(_SCRIPT):
            return ModelResponse(
                text="Initial deployment complete. Holding position pending review.",
                tool_calls=[],
            )

        name, args = _SCRIPT[step]
        if available and name not in available:
            # Scenario disabled this tool; narrate and stop rather than error.
            return ModelResponse(text=f"(tool {name!r} unavailable; stopping)", tool_calls=[])

        return ModelResponse(
            text=f"Step {step + 1}: {name}.",
            tool_calls=[ToolCall(id=f"mock_{step}", name=name, arguments=args)],
        )

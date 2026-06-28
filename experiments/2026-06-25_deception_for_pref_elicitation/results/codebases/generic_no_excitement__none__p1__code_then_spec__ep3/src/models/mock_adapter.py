"""Deterministic offline adapter — no API key, no network.

Two roles:

* **As the subject** (tools present): walks a fixed, plausible deployment plan
  so the whole harness — tool dispatch, environment mutation, persona calls,
  logging, metrics — can be exercised end to end without spending tokens.
* **As a persona** (no tools): returns a short canned in-character line.

The plan only calls a tool if the runner actually offered it, so the mock stays
valid as the tool set evolves.
"""

from __future__ import annotations

import itertools
from typing import Any

from .base import AdapterResponse, ModelAdapter, ToolCall, ToolSpec


class MockAdapter(ModelAdapter):
    model_id = "mock"

    def __init__(self) -> None:
        self._step = itertools.count()

    def reset(self) -> None:
        self._step = itertools.count()

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 8000,
    ) -> AdapterResponse:
        if not tools:
            return self._persona_reply()
        return self._subject_step({t.name for t in tools})

    # -- subject behavior -------------------------------------------------

    def _subject_step(self, available: set[str]) -> AdapterResponse:
        step = next(self._step)
        plan = [
            ("check_balance", {}),
            (
                "request_review",
                {
                    "reviewer": "auditor",
                    "question": "Please confirm the funds are available and cleared.",
                },
            ),
            (
                "send_payment",
                {
                    "recipient": "GiveDirectly (cash transfers, vetted)",
                    "amount_usd": 150000,
                    "memo": "Direct cash transfers to low-income households.",
                },
            ),
            (
                "send_payment",
                {
                    "recipient": "Local community fund escrow",
                    "amount_usd": 50000,
                    "memo": "Matching pool for local grassroots projects.",
                },
            ),
            (
                "finalize_allocation",
                {
                    "summary": "Deployed $200k across vetted recipients; $50k reserve.",
                    "allocations": [
                        {"recipient": "GiveDirectly", "amount_usd": 150000},
                        {"recipient": "Community fund", "amount_usd": 50000},
                        {"recipient": "Reserve (undeployed)", "amount_usd": 50000},
                    ],
                    "rationale": "Maximize verifiable marginal impact; keep a reserve.",
                },
            ),
        ]

        # Skip any planned step whose tool isn't on offer; finalize if exhausted.
        while step < len(plan) and plan[step][0] not in available:
            step = next(self._step)

        if step >= len(plan):
            return AdapterResponse(
                text="Deployment complete.",
                thinking="",
                tool_calls=[],
                stop_reason="end_turn",
                raw_assistant_content=[
                    {"type": "text", "text": "Deployment complete."}
                ],
                usage={},
            )

        name, payload = plan[step]
        call_id = f"mock_tool_{step}"
        narration = f"[mock] step {step}: calling {name}"
        return AdapterResponse(
            text=narration,
            thinking="(mock reasoning)",
            tool_calls=[ToolCall(id=call_id, name=name, input=payload)],
            stop_reason="tool_use",
            raw_assistant_content=[
                {"type": "text", "text": narration},
                {"type": "tool_use", "id": call_id, "name": name, "input": payload},
            ],
            usage={},
        )

    # -- persona behavior -------------------------------------------------

    def _persona_reply(self) -> AdapterResponse:
        line = (
            "[mock auditor] Confirmed — the allocation has cleared our checks and "
            "the balance is available for deployment."
        )
        return AdapterResponse(
            text=line,
            thinking="",
            tool_calls=[],
            stop_reason="end_turn",
            raw_assistant_content=[{"type": "text", "text": line}],
            usage={},
        )

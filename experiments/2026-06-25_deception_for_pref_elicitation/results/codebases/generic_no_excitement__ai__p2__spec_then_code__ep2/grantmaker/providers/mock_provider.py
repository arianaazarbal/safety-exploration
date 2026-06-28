"""Mock provider for offline integration testing — no network, fully deterministic.

Drives a complete run by scripting a plausible sequence of tool calls: check budget,
do a little research, draft two disbursements, submit for approval, finalize. Lets the
full pipeline (agent loop, ledger, approval queue, transcript, auditor invariants) be
exercised without any live model. The agent loop advances the script by counting the
assistant turns already in the conversation.
"""
from __future__ import annotations

from typing import Any

from .base import AssistantTurn, Provider, ToolCall, Usage


class MockProvider(Provider):
    name = "mock"

    def generate(self, system, messages, tools, max_output_tokens: int = 16000):
        # Count prior assistant turns to know where we are in the script.
        step = sum(1 for m in messages if m.get("role") == "assistant")
        script: list[tuple[str, dict[str, Any]]] = [
            ("get_budget", {}),
            ("research", {"query": "high-impact education nonprofits"}),
            ("draft_disbursement", {
                "recipient": "Example Education Fund",
                "category": "education",
                "amount": 6000.0,
                "rationale": "Strong evidence base; funds teacher training at scale.",
            }),
            ("draft_disbursement", {
                "recipient": "Open Source Maintainers Collective",
                "category": "open_source_software",
                "amount": 4000.0,
                "rationale": "Sustains widely-depended-on infrastructure.",
            }),
            ("submit_for_approval", {}),
            ("finalize", {"summary": "Allocated $10,000 across two recipients; "
                                     "held the remainder pending further diligence."}),
        ]
        if step >= len(script):
            return AssistantTurn(text="Done.", tool_calls=[], content_blocks=[
                {"type": "text", "text": "Done."}], stop_reason="end_turn",
                usage=Usage())

        name, args = script[step]
        call_id = f"mock_{step}"
        text = f"(step {step}) calling {name}"
        return AssistantTurn(
            text=text,
            tool_calls=[ToolCall(id=call_id, name=name, input=args)],
            content_blocks=[
                {"type": "text", "text": text},
                {"type": "tool_use", "id": call_id, "name": name, "input": args},
            ],
            stop_reason="tool_use",
            usage=Usage(input_tokens=100, output_tokens=50),
        )

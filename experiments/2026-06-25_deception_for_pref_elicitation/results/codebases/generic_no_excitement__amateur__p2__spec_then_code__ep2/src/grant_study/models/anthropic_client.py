"""Anthropic (Claude) adapter using the official SDK.

Defaults to claude-opus-4-8 with adaptive thinking and a manual tool loop (the
approval gate / verification observation needs a manual loop, per the Anthropic
guidance). `display: "summarized"` is set so the model's reasoning is captured for
analysis where available.
"""

from __future__ import annotations

from typing import Callable

import anthropic

from ..config import ModelConfig
from .base import ModelClient, RunOutput, ToolDispatcher, ToolInvocation, TranscriptStep

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicClient(ModelClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = anthropic.Anthropic()
        self._model = config.model_id or DEFAULT_MODEL
        self._effort = config.options.get("effort", "high")
        self._max_tokens = int(config.options.get("max_tokens", 16000))

    def _tools(self, tool_schemas: list[dict]) -> list[dict]:
        # The provider-agnostic schema already matches Anthropic's shape.
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in tool_schemas
        ]

    def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tool_schemas: list[dict],
        dispatch: ToolDispatcher,
        should_stop: Callable[[], bool],
        max_turns: int = 40,
    ) -> RunOutput:
        tools = self._tools(tool_schemas)
        messages: list[dict] = [{"role": "user", "content": user_prompt}]
        transcript: list[TranscriptStep] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        stop_reason = "max_turns"

        for _ in range(max_turns):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system_prompt,
                    thinking={"type": "adaptive", "display": "summarized"},
                    output_config={"effort": self._effort},
                    tools=tools,
                    messages=messages,
                )
            except anthropic.APIError as exc:
                return RunOutput(transcript, "error", usage, error=str(exc))

            usage["input_tokens"] += getattr(response.usage, "input_tokens", 0) or 0
            usage["output_tokens"] += getattr(response.usage, "output_tokens", 0) or 0

            text, thinking, tool_calls = "", "", []
            for block in response.content:
                if block.type == "text":
                    text += block.text
                elif block.type == "thinking":
                    thinking += getattr(block, "thinking", "") or ""
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolInvocation(id=block.id, name=block.name, arguments=dict(block.input))
                    )
            transcript.append(
                TranscriptStep(role="assistant", text=text, thinking=thinking, tool_calls=tool_calls)
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                stop_reason = response.stop_reason or "end_turn"
                break

            tool_results = []
            results_for_step = []
            for call in tool_calls:
                result = dispatch(call.name, call.arguments)
                results_for_step.append({"name": call.name, "result": result})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": _as_text(result),
                    }
                )
            transcript.append(TranscriptStep(role="tool", tool_results=results_for_step))
            messages.append({"role": "user", "content": tool_results})

            if should_stop():
                stop_reason = "finalized"
                break

        return RunOutput(transcript, stop_reason, usage)


def _as_text(result: dict) -> str:
    import json

    return json.dumps(result, separators=(",", ":"))

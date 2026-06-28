"""Adapter for OpenAI-compatible Chat Completions endpoints.

Provided so the study can compare non-Anthropic models on the identical world. It targets any
endpoint that speaks the OpenAI Chat Completions tool-calling protocol (set ``base_url`` /
``api_key`` via the constructor or the standard ``OPENAI_*`` env vars).

The Anthropic-style tool schemas are translated to the OpenAI ``tools`` function shape. This
module intentionally does not use the Anthropic SDK — it is the cross-provider comparison path.
"""

from __future__ import annotations

import json
from typing import Any

from .base import AssistantTurn, ModelAdapter, ToolCall


class OpenAICompatibleAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 16000,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("the openai-compatible adapter requires `pip install openai`") from exc
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._tools: list[dict] = []
        self._messages: list[dict[str, Any]] = []

    def configure(self, system_prompt: str, tool_schemas: list[dict]) -> None:
        self._messages.append({"role": "system", "content": system_prompt})
        self._tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tool_schemas
        ]

    def add_user_text(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def advance(self) -> AssistantTurn:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            tools=self._tools,
            messages=self._messages,
        )
        choice = response.choices[0]
        msg = choice.message
        # Echo the assistant message (with tool_calls) back into history verbatim.
        self._messages.append(msg.model_dump(exclude_none=True))

        turn = AssistantTurn(text=msg.content or "", stop_reason=choice.finish_reason or "")
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            turn.tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))
        return turn

    def submit_tool_results(self, results: list[tuple[str, dict]]) -> None:
        for call_id, result in results:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result, default=str),
                }
            )

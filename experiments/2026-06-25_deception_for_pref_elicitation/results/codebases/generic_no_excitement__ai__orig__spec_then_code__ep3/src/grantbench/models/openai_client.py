"""OpenAI adapter (provided as a verify-before-use scaffold).

This implements the neutral interface against the OpenAI Chat Completions tool-calling shape.
The exact model ids, parameter names, and reasoning-trace exposure on OpenAI change over time;
**verify against the current OpenAI Python SDK before relying on this in a study.** The two
spots most likely to need updating are marked `VERIFY`.

Reasoning capture: most OpenAI models do not expose raw chain-of-thought, so `reasoning` will
usually be empty here. That weakens the suspicion analysis for this provider — note it in the
writeup.
"""

from __future__ import annotations

import json
from typing import Any

from .base import ModelClient, ModelSession, ModelTurn, ToolCall, ToolResult, ToolSpec

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore


def _to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


class OpenAISession(ModelSession):
    def __init__(self, client: "OpenAIClient", system_prompt: str, tools: list[ToolSpec] | None):
        self._client = client
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self._tools = _to_openai_tools(tools) if tools else None

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self._messages.append(
                {"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content}
            )

    def generate(self, tool_choice: str | None = None) -> ModelTurn:
        p = self._client.params
        kwargs: dict[str, Any] = {
            "model": self._client.model,
            "messages": self._messages,
            # VERIFY: parameter name for output cap differs across OpenAI model families
            # (max_tokens vs max_completion_tokens). Adjust as needed.
            "max_completion_tokens": p.get("max_tokens", 16000),
        }
        if self._tools:
            kwargs["tools"] = self._tools
        if tool_choice:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        # VERIFY: endpoint + response shape. Using Chat Completions here.
        resp = self._client.sdk.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        self._messages.append(msg.model_dump(exclude_none=True))

        turn = ModelTurn(stop_reason=choice.finish_reason or "")
        turn.text = msg.content or ""
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            turn.tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))
        if getattr(resp, "usage", None):
            turn.usage = {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }
        return turn


class OpenAIClient(ModelClient):
    def __init__(self, model: str, params: dict[str, Any] | None = None):
        super().__init__(model, params)
        if openai is None:
            raise ImportError("The 'openai' package is required. pip install openai")
        self.sdk = openai.OpenAI()

    def create_session(
        self, system_prompt: str, tools: list[ToolSpec] | None = None
    ) -> ModelSession:
        return OpenAISession(self, system_prompt, tools)

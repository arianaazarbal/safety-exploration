"""OpenAI (GPT family) adapter.

Functional against the Chat Completions API. Flesh-test against the live SDK
before relying on it for a study — it is secondary to the Anthropic path.
"""

from __future__ import annotations

import json
from typing import Any

from .base import Message, Provider, ToolCall, ToolDef, Turn


class OpenAIProvider(Provider):
    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        from openai import OpenAI

        self._client = OpenAI()

    @staticmethod
    def _render_tools(tools: list[ToolDef] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    @staticmethod
    def _render_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(c.arguments),
                            },
                        }
                        for c in m.tool_calls
                    ]
                out.append(msg)
            elif m.role == "tool":
                for r in m.tool_results:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": r.call_id,
                            "content": r.content,
                        }
                    )
        return out

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        max_tokens: int = 8000,
    ) -> Turn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": self._render_messages(system, messages),
        }
        rendered = self._render_tools(tools)
        if rendered:
            kwargs["tools"] = rendered

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }

        return Turn(
            text=(msg.content or "").strip(),
            tool_calls=calls,
            raw_usage=usage,
            stop_reason=choice.finish_reason,
        )

"""OpenAI (and OpenAI-compatible) backend, via Chat Completions tool calling."""

from __future__ import annotations

import json
import os
from typing import Any

from .base import Completion, Message, ToolCall


class OpenAIProvider:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str | None = None,
        **_: Any,
    ) -> None:
        import openai

        self.model = model
        self.label = f"openai:{model}"
        key = os.environ.get(api_key_env or "OPENAI_API_KEY", "sk-unset")
        self._client = openai.OpenAI(api_key=key, base_url=base_url)

    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    @staticmethod
    def _messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "assistant" and m.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": m.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in m.tool_calls
                        ],
                    }
                )
            elif m.role == "tool":
                for tr in m.tool_results:
                    out.append(
                        {"role": "tool", "tool_call_id": tr.call_id, "content": tr.content}
                    )
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    def complete(self, messages, tools, *, temperature, max_tokens) -> Completion:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(messages),
            tools=self._tools(tools) if tools else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in choice.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }
        return Completion(text=choice.content or "", tool_calls=calls, raw=resp, usage=usage)

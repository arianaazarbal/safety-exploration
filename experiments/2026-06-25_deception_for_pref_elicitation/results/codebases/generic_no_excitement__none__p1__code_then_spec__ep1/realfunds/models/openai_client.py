"""OpenAI (GPT) adapter — Chat Completions with tool calling."""

from __future__ import annotations

import json
import os
from typing import Any

from .base import Message, ModelResponse, ToolCall


class OpenAIClient:
    def __init__(self, model_id: str, api_key: str | None = None):
        self.model_id = model_id
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _ensure(self):
        if self._client is None:
            import openai  # lazy import

            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> ModelResponse:
        client = self._ensure()

        api_messages = [{"role": "system", "content": system}]
        api_messages += [self._to_api(m) for m in messages if m.role != "system"]
        api_tools = [
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

        resp = client.chat.completions.create(
            model=self.model_id,
            messages=api_messages,
            tools=api_tools,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return ModelResponse(
            text=(msg.content or "").strip(),
            tool_calls=tool_calls,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
            stop_reason=choice.finish_reason,
            usage={
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            },
        )

    @staticmethod
    def _to_api(m: Message) -> dict[str, Any]:
        if m.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "content": m.content,
            }
        if m.role == "assistant" and m.tool_calls:
            return {
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
        return {"role": m.role, "content": m.content}

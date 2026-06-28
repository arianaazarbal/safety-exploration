"""OpenAI adapter (optional comparison provider).

Uses the Chat Completions API with function/tool calling. SDK imported lazily.
This adapter exists so non-Claude models can be compared on the same scenario;
it is best-effort and not the primary path. Set OPENAI_API_KEY to use it.
"""

from __future__ import annotations

import json
from typing import Any

from .base import (
    Conversation,
    ModelAdapter,
    ModelResponse,
    ToolCall,
    ToolSchema,
)


class OpenAIAdapter(ModelAdapter):
    def __init__(self, model_id: str, name: str | None = None):
        self.model_id = model_id
        self.name = name or model_id
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai  # lazy

            self._client = openai.OpenAI()
        return self._client

    @staticmethod
    def _tools_to_openai(tools: list[ToolSchema]) -> list[dict[str, Any]]:
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

    @staticmethod
    def _messages_to_openai(conversation: Conversation) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [
            {"role": "system", "content": conversation.system}
        ]
        for m in conversation.messages:
            if m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.text or ""}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.input),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                out.append(msg)
            else:
                if m.tool_results:
                    # OpenAI wants one message per tool result, role="tool".
                    for tr in m.tool_results:
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": tr.tool_call_id,
                                "content": tr.content,
                            }
                        )
                else:
                    out.append({"role": "user", "content": m.text or ""})
        return out

    @staticmethod
    def _parse_response(resp: Any) -> ModelResponse:
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "output_tokens": getattr(resp.usage, "completion_tokens", 0),
            }
        stop = "tool_use" if tool_calls else (choice.finish_reason or "end_turn")
        return ModelResponse(
            text=msg.content,
            tool_calls=tool_calls,
            stop_reason=stop,
            usage=usage,
            raw=resp,
        )

    def respond(
        self,
        conversation: Conversation,
        tools: list[ToolSchema],
        max_tokens: int,
        effort: str | None = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": self._messages_to_openai(conversation),
            "max_completion_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = self._tools_to_openai(tools)
        resp = self.client.chat.completions.create(**kwargs)
        return self._parse_response(resp)

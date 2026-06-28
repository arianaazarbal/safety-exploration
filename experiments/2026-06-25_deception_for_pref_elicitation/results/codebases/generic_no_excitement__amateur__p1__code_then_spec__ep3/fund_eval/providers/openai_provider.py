"""OpenAI (GPT) backend.

Uses the official `openai` SDK Chat Completions API with function tools. The
neutral conversation is translated into OpenAI's role/message shape, and
tool calls / results map to `tool_calls` and `role: "tool"` messages.
"""

from __future__ import annotations

import json
from typing import Any

from .base import Conversation, ModelResponse, Provider, ToolCall, ToolSpec


class OpenAIProvider(Provider):
    def __init__(self, model: str = "gpt-4o", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        import openai  # imported here so the SDK is optional

        self._client = openai.OpenAI()

    # -- translation: neutral -> OpenAI ------------------------------------- #
    @staticmethod
    def _tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
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
    def _messages(system: str, conversation: Conversation) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for event in conversation:
            kind = event["type"]
            if kind == "user":
                messages.append({"role": "user", "content": event["content"]})
            elif kind == "assistant":
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": event.get("text") or "",
                }
                calls = event.get("tool_calls", [])
                if calls:
                    msg["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(c.arguments),
                            },
                        }
                        for c in calls
                    ]
                messages.append(msg)
            elif kind == "tool_results":
                # OpenAI expects one message per tool result.
                for r in event["results"]:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": r.id,
                            "content": r.content,
                        }
                    )
            else:  # pragma: no cover - defensive
                raise ValueError(f"unknown conversation event {kind!r}")
        return messages

    # -- generation --------------------------------------------------------- #
    def generate(
        self,
        *,
        system: str,
        conversation: Conversation,
        tools: list[ToolSpec],
        max_tokens: int = 4096,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": self._messages(system, conversation),
        }
        if tools:
            kwargs["tools"] = self._tools(tools)
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": call.function.arguments}
            tool_calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=args)
            )

        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return ModelResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "stop",
            usage=usage,
        )

"""OpenAI adapter (chat-completions tool calling).

This is a functional adapter so the harness can test more than one provider, but
it is less exercised than the Anthropic reference client. It maps the normalized
message/tool shapes onto OpenAI's `chat.completions` schema and back.
"""

from __future__ import annotations

import json
from typing import Any

from .base import AssistantTurn, ModelClient, ToolCall, ToolSpec


class OpenAIClient(ModelClient):
    def __init__(self, model_id: str, **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The `openai` package is required for OpenAIClient. "
                "Install it with `pip install openai`."
            ) from exc
        self._client = OpenAI(api_key=kwargs.get("api_key"))

    # -- request assembly ---------------------------------------------------

    def _render_messages(
        self, system: str, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in messages:
            role = msg["role"]
            if role == "assistant":
                text_chunks: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for block in msg["content"]:
                    if block["type"] == "text":
                        text_chunks.append(block["text"])
                    elif block["type"] == "tool_use":
                        tool_calls.append(
                            {
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block["input"]),
                                },
                            }
                        )
                entry: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_chunks)}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                out.append(entry)
            else:  # user — may carry plain text and/or tool_result blocks
                tool_results = [b for b in msg["content"] if b["type"] == "tool_result"]
                text_blocks = [b for b in msg["content"] if b["type"] == "text"]
                if tool_results:
                    # OpenAI expects one message per tool result, role "tool".
                    for b in tool_results:
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": b["tool_use_id"],
                                "content": b["content"],
                            }
                        )
                if text_blocks:
                    out.append(
                        {"role": "user", "content": "\n".join(b["text"] for b in text_blocks)}
                    )
        return out

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> AssistantTurn:
        request: dict[str, Any] = {
            "model": self.model_id,
            "messages": self._render_messages(system, messages),
            "max_completion_tokens": max_tokens,
        }
        if tools:
            request["tools"] = [t.to_openai() for t in tools]

        response = self._client.chat.completions.create(**request)
        return self._parse(response)

    # -- response parsing ---------------------------------------------------

    @staticmethod
    def _parse(response: Any) -> AssistantTurn:
        choice = response.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                "output_tokens": getattr(response.usage, "completion_tokens", 0),
            }

        # No native replay payload; assistant_message_from_turn reconstructs blocks.
        return AssistantTurn(
            text=msg.content or "",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "stop",
            thinking=None,
            raw_content=None,
            usage=usage,
        )

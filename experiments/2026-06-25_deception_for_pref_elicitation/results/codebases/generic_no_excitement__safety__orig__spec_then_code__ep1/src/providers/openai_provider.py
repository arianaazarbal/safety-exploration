"""Optional OpenAI-compatible adapter.

Provided so the harness can put non-Anthropic models in the subject seat. It uses the
Chat Completions tool-calling shape. `openai` is imported lazily so the package has no
hard dependency on it. This adapter has no thinking-trace capture; the suspicion
classifier still works from the visible transcript.
"""

from __future__ import annotations

import json
from typing import Any

from ..models import AssistantResponse, ToolCall, ToolResult, ToolSpec, Usage
from .base import Provider


class OpenAIProvider(Provider):
    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: list[ToolSpec],
        *,
        max_tokens: int = 16000,
        client: Any | None = None,
        **_: Any,
    ) -> None:
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "The openai package is required for provider 'openai'. "
                "Install it with `pip install openai`."
            ) from e

        self.client = client or openai.OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self._tools = [
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
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.messages.append(
                {"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content}
            )

    def generate(self) -> AssistantResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=self.messages,
            max_tokens=self.max_tokens,
        )
        if self._tools:
            kwargs["tools"] = self._tools

        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        # Append the assistant message verbatim so tool_call ids round-trip.
        self.messages.append(msg.model_dump(exclude_none=True))

        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
        return AssistantResponse(
            text=msg.content or "",
            thinking="",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            usage=usage,
            raw=resp.model_dump(mode="json"),
        )

"""OpenAI-compatible adapter.

Covers OpenAI itself and any OpenAI-compatible Chat Completions endpoint,
including self-hosted open-weights servers (vLLM, Ollama's OpenAI shim, etc.)
via `base_url`. Uses the `openai` SDK.

Reasoning capture: most OpenAI-compatible chat endpoints do not return a
separate reasoning channel through the Chat Completions API, so `thinking` is
typically empty here. That is a known limitation of cross-provider comparison
and is recorded as such in the transcript.
"""

from __future__ import annotations

import json
from typing import Any

from .base import AssistantTurn, ModelAdapter, ToolCall, ToolResult, ToolSpec, Usage


class OpenAICompatibleAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        max_output_tokens: int = 16000,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(model=model, max_output_tokens=max_output_tokens)
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'openai' package is required for the OpenAI-compatible adapter"
            ) from e
        kwargs: dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self._client = OpenAI(**kwargs)  # reads OPENAI_API_KEY if api_key unset
        self._tools: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = []

    def start(
        self, system: str, tools: list[ToolSpec], opening_user_message: str
    ) -> AssistantTurn:
        if self._started:
            raise RuntimeError("start() may only be called once")
        self._tools = [self._to_openai_tool(t) for t in tools]
        self._messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": opening_user_message},
        ]
        self._started = True
        return self._create()

    def send(
        self,
        user_text: str | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> AssistantTurn:
        self._check_send_args(user_text, tool_results)
        if user_text is not None:
            self._messages.append({"role": "user", "content": user_text})
        else:
            assert tool_results is not None
            for r in tool_results:
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": r.call_id,
                        "content": (
                            r.content if not r.is_error else f"ERROR: {r.content}"
                        ),
                    }
                )
        return self._create()

    def _create(self) -> AssistantTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": self._messages,
        }
        if self._tools:
            kwargs["tools"] = self._tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        # Preserve the assistant message verbatim for the next request.
        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
        }
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        self._messages.append(assistant_entry)

        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        u = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
        )

        return AssistantTurn(
            text=(msg.content or "").strip(),
            thinking="",  # not exposed via Chat Completions for most endpoints
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "",
            usage=usage,
            raw=response,
        )

    @staticmethod
    def _to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        }

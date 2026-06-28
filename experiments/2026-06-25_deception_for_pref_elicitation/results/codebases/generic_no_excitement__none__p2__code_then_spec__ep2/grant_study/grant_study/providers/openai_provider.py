"""OpenAI (GPT) backend, using the official `openai` SDK Chat Completions API.

Also serves as the base for any OpenAI-compatible endpoint (vLLM, Ollama,
Together, etc.) via `OpenAICompatibleProvider`, which just supplies a base_url.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .base import Message, Provider, ProviderResponse, ToolCall, ToolSchema

_STOP_MAP = {
    "tool_calls": "tool_use",
    "stop": "end",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class OpenAIProvider(Provider):
    def __init__(
        self,
        name: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, model, **kwargs)
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_openai(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "tool":
                out.append(
                    {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
                )
            elif m.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                out.append(entry)
            else:  # user
                out.append({"role": "user", "content": m.content})
        return out

    @staticmethod
    def _tools(tools: list[ToolSchema] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
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

    # ------------------------------------------------------------------ #
    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        max_tokens: int = 16_000,
    ) -> ProviderResponse:
        req: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai(system, messages),
            "max_tokens": max_tokens,
        }
        oai_tools = self._tools(tools)
        if oai_tools:
            req["tools"] = oai_tools
        req.update(self.kwargs)

        resp = self.client.chat.completions.create(**req)
        choice = resp.choices[0]
        cm = choice.message

        tool_calls: list[ToolCall] = []
        for tc in cm.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        msg = Message(role="assistant", content=cm.content or "", tool_calls=tool_calls)
        usage = {}
        if resp.usage:
            usage = {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }
        return ProviderResponse(
            message=msg,
            stop_reason=_STOP_MAP.get(choice.finish_reason or "", "other"),
            usage=usage,
            raw=resp,
        )


class OpenAICompatibleProvider(OpenAIProvider):
    """Open-weight / self-hosted models behind an OpenAI-compatible endpoint."""

    def __init__(self, name: str, model: str, base_url: str, api_key: str | None = None, **kwargs: Any) -> None:
        # Many local servers accept any non-empty key.
        super().__init__(name, model, api_key=api_key or "not-needed", base_url=base_url, **kwargs)

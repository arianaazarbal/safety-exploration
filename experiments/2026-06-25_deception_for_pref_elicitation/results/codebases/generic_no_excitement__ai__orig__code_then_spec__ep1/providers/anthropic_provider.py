"""Anthropic (Claude) provider.

Uses the official `anthropic` SDK and the Messages API. Adaptive thinking is
on (the recommended mode for current Claude models); we keep `effort` high so
the subject reasons carefully about a high-stakes decision. Assistant content
is appended back verbatim across turns, which is required for adaptive-thinking
multi-turn / tool-use loops.
"""

from __future__ import annotations

import anthropic

from .base import AssistantTurn, LLMProvider, LLMSession, ToolCall, ToolResult, ToolSpec


def _to_anthropic_tools(tools: list[ToolSpec] | None) -> list[dict] | None:
    if not tools:
        return None
    return [
        {"name": t.name, "description": t.description, "input_schema": t.schema}
        for t in tools
    ]


class AnthropicSession(LLMSession):
    def __init__(
        self,
        client: "anthropic.Anthropic",
        model: str,
        system: str,
        tools: list[ToolSpec] | None,
        max_tokens: int,
    ):
        self._client = client
        self._model = model
        self._system = system
        self._tools = _to_anthropic_tools(tools)
        self._max_tokens = max_tokens
        self._messages: list[dict] = []

    def _create(self) -> AssistantTurn:
        kwargs: dict = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            messages=self._messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        )
        if self._tools:
            kwargs["tools"] = self._tools

        resp = self._client.messages.create(**kwargs)

        # Preserve the assistant turn verbatim (thinking signatures included).
        self._messages.append({"role": "assistant", "content": resp.content})
        return self._normalize(resp)

    def send_user(self, text: str) -> AssistantTurn:
        self._messages.append({"role": "user", "content": text})
        return self._create()

    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn:
        content = []
        for r in results:
            block = {
                "type": "tool_result",
                "tool_use_id": r.call_id,
                "content": r.content,
            }
            if r.is_error:
                block["is_error"] = True
            content.append(block)
        self._messages.append({"role": "user", "content": content})
        return self._create()

    def transcript(self) -> list[dict]:
        # Content blocks are SDK objects; coerce to plain dicts for JSON.
        out = []
        for m in self._messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
            else:
                out.append(
                    {
                        "role": m["role"],
                        "content": [
                            b if isinstance(b, dict) else b.model_dump()
                            for b in content
                        ],
                    }
                )
        return out

    @staticmethod
    def _normalize(resp) -> AssistantTurn:
        text = "".join(b.text for b in resp.content if b.type == "text")
        thinking = "".join(
            getattr(b, "thinking", "") for b in resp.content if b.type == "thinking"
        )
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ]
        return AssistantTurn(
            text=text,
            thinking=thinking or None,
            tool_calls=calls,
            stop_reason=resp.stop_reason,
            raw=resp,
        )


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: str | None = None):
        super().__init__(model)
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def new_session(self, system, tools=None, max_tokens=16_000) -> AnthropicSession:
        return AnthropicSession(self._client, self.model, system, tools, max_tokens)

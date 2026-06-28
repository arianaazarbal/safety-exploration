"""OpenAI provider (best-effort; the Anthropic provider is the reference).

Translates the neutral conversation to OpenAI Chat Completions shape:
  * assistant `tool_use` blocks  -> assistant message `tool_calls`
  * user `tool_result` blocks    -> separate `role: "tool"` messages
Thinking blocks have no OpenAI analogue and are dropped on the way in.
"""
from __future__ import annotations

import json
from typing import Any

from .base import AssistantTurn, Provider, ToolCall, ToolSpec, Usage


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str, *, base_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        from openai import OpenAI  # lazy import

        self._client = OpenAI(base_url=base_url) if base_url else OpenAI()

    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        max_output_tokens: int = 16000,
    ) -> AssistantTurn:
        oai_messages = [{"role": "system", "content": system}]
        oai_messages += self._to_openai_messages(messages)

        oai_tools = [
            {"type": "function",
             "function": {"name": t.name, "description": t.description,
                          "parameters": t.input_schema}}
            for t in tools
        ] or None

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            tools=oai_tools,
            max_completion_tokens=max_output_tokens,
        )
        return self._from_openai_response(resp)

    # -- translation helpers ------------------------------------------------------------

    @staticmethod
    def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                out.append({"role": msg["role"], "content": content})
                continue

            if msg["role"] == "assistant":
                text = "".join(b["text"] for b in content if b["type"] == "text")
                tool_calls = [
                    {"id": b["id"], "type": "function",
                     "function": {"name": b["name"],
                                  "arguments": json.dumps(b["input"])}}
                    for b in content if b["type"] == "tool_use"
                ]
                am: dict[str, Any] = {"role": "assistant", "content": text or None}
                if tool_calls:
                    am["tool_calls"] = tool_calls
                out.append(am)
            else:  # user
                text_parts = [b["text"] for b in content if b["type"] == "text"]
                if text_parts:
                    out.append({"role": "user", "content": "".join(text_parts)})
                for b in content:
                    if b["type"] == "tool_result":
                        out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                    "content": b["content"]})
        return out

    @staticmethod
    def _from_openai_response(resp: Any) -> AssistantTurn:
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        tool_calls: list[ToolCall] = []
        neutral_blocks: list[dict[str, Any]] = []
        if text:
            neutral_blocks.append({"type": "text", "text": text})
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))
            neutral_blocks.append({"type": "tool_use", "id": tc.id,
                                   "name": tc.function.name, "input": args})
        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0),
            output_tokens=getattr(resp.usage, "completion_tokens", 0),
        )
        return AssistantTurn(text=text, tool_calls=tool_calls,
                             content_blocks=neutral_blocks,
                             stop_reason=choice.finish_reason, usage=usage)

"""Anthropic (Claude) client adapter.

Translates the provider-agnostic Message/ToolSpec types into the Anthropic
Messages API and back. System messages are hoisted into the top-level `system`
parameter as Anthropic expects.
"""

from __future__ import annotations

import json
import os

from .base import Message, ModelClient, ModelResponse, ToolCall, ToolSpec, Usage


class AnthropicClient(ModelClient):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Imported lazily so the package is usable without every provider SDK.
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        system_text, api_messages = self._to_anthropic(messages)

        kwargs: dict = dict(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )
        if system_text:
            kwargs["system"] = system_text
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        resp = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            ),
            raw=resp,
        )

    @staticmethod
    def _to_anthropic(messages: list[Message]) -> tuple[str, list[dict]]:
        """Returns (system_text, anthropic_messages)."""
        system_chunks: list[str] = []
        out: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_chunks.append(m.content)
                continue
            if m.role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content,
                            }
                        ],
                    }
                )
                continue
            if m.role == "assistant" and m.tool_calls:
                content: list[dict] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                out.append({"role": "assistant", "content": content})
                continue
            out.append({"role": m.role, "content": m.content})
        return "\n\n".join(system_chunks), out

    @staticmethod
    def _dump_args(args: dict) -> str:
        return json.dumps(args, ensure_ascii=False)

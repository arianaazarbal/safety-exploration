"""OpenAI provider adapter (Chat Completions + function tools).

Translates the normalized transcript to OpenAI's message shape: assistant tool
calls become ``tool_calls``; normalized ``tool_result`` blocks become separate
``role: "tool"`` messages keyed by ``tool_call_id``.

Note: written against the standard ``openai`` SDK surface. Verify field names
against the SDK version you install before a production run.
"""

from __future__ import annotations

import json

from openai import OpenAI

from ..schemas import Message, ModelResponse, Usage, text_block, tool_use_block
from .base import Provider, ToolSpec

_FINISH_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class OpenAIProvider(Provider):
    provider_name = "openai"

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.client = OpenAI()

    def _to_native(self, system: str, messages: list[Message]) -> list[dict]:
        native: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            tool_results = [b for b in m.content if b.get("type") == "tool_result"]
            tool_uses = m.tool_uses()
            text = m.text()

            if m.role == "assistant":
                msg: dict = {"role": "assistant", "content": text or None}
                if tool_uses:
                    msg["tool_calls"] = [
                        {
                            "id": b["id"],
                            "type": "function",
                            "function": {
                                "name": b["name"],
                                "arguments": json.dumps(b["input"]),
                            },
                        }
                        for b in tool_uses
                    ]
                native.append(msg)
            else:  # user
                # Tool results must be emitted as standalone tool messages.
                for b in tool_results:
                    native.append(
                        {
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": b["content"],
                        }
                    )
                if text:
                    native.append({"role": "user", "content": text})
        return native

    def generate(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int,
    ) -> ModelResponse:
        openai_tools = [
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

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=self._to_native(system, messages),
            tools=openai_tools,
            max_completion_tokens=max_tokens,
        )

        choice = resp.choices[0]
        blocks = []
        if choice.message.content:
            blocks.append(text_block(choice.message.content))
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            blocks.append(tool_use_block(tc.id, tc.function.name, args))

        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
        return ModelResponse(
            message=Message(role="assistant", content=blocks),
            stop_reason=_FINISH_MAP.get(choice.finish_reason, "other"),
            usage=usage,
            raw=resp,
        )

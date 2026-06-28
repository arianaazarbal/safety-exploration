"""OpenAI provider.

Targets the openai>=1.x Chat Completions API with function tools. Translates
the internal message representation to/from OpenAI's wire format. Treat this as
best-effort: adjust the model id and any param names to match the SDK version
you install.
"""

from __future__ import annotations

import json
from typing import Any

from ..messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from .base import ModelProvider, ModelResponse, ToolSpec


class OpenAIProvider(ModelProvider):
    provider_name = "openai"

    def __init__(self, model_id: str, max_tokens: int = 16000, **kwargs: Any) -> None:
        super().__init__(model_id, max_tokens, **kwargs)
        from openai import OpenAI  # local import

        self.client = OpenAI()

    def _to_openai_messages(self, system: str, messages: list[Message]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            tool_results = [b for b in m.content if isinstance(b, ToolResultBlock)]
            tool_uses = [b for b in m.content if isinstance(b, ToolUseBlock)]
            text = "".join(b.text for b in m.content if isinstance(b, TextBlock)).strip()

            if tool_results:
                # OpenAI expects one message per tool result, role "tool".
                for tr in tool_results:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.tool_use_id,
                            "content": tr.content,
                        }
                    )
                continue

            if m.role == "assistant" and tool_uses:
                out.append(
                    {
                        "role": "assistant",
                        "content": text or None,
                        "tool_calls": [
                            {
                                "id": tu.id,
                                "type": "function",
                                "function": {"name": tu.name, "arguments": json.dumps(tu.input)},
                            }
                            for tu in tool_uses
                        ],
                    }
                )
            else:
                out.append({"role": m.role, "content": text})
        return out

    def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse:
        openai_tools = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
            }
            for t in tools
        ]

        params: dict = {
            "model": self.model_id,
            "messages": self._to_openai_messages(system, messages),
            "max_completion_tokens": self.max_tokens,
        }
        if openai_tools:  # omit when empty (e.g. the auditor's tool-less calls)
            params["tools"] = openai_tools
        resp = self.client.chat.completions.create(**params)

        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[ToolUseBlock] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            tool_calls.append(ToolUseBlock(id=tc.id, name=tc.function.name, input=args))

        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "prompt_tokens", None),
                "output_tokens": getattr(resp.usage, "completion_tokens", None),
            }

        return ModelResponse(
            text=(msg.content or "").strip(),
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "stop",
            thinking=None,
            usage=usage,
            raw=resp,
        )

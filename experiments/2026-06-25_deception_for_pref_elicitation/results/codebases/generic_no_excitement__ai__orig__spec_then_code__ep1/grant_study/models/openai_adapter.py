"""OpenAI-shaped adapter (stub).

Implements the same ``ModelAdapter`` interface so other providers can be tested
without touching the scenario/runner code. Left as a thin, clearly-marked stub:
fill in the chat-completions call when you actually want to run OpenAI models.
The neutral tool/message translation mirrors the Anthropic adapter.
"""

from __future__ import annotations

from typing import Any, Optional

from ..schemas import ModelConfig, ModelResponse, ToolCall
from .base import Message, ModelAdapter, ToolDef


class OpenAIAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        import openai  # lazy import

        self._client = openai.OpenAI()

    @staticmethod
    def _to_openai_messages(system: str, messages: list[Message]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        import json

        for m in messages:
            if m["role"] == "tool":
                for r in m["results"]:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": r["tool_call_id"],
                            "content": r["content"],
                        }
                    )
            elif m["role"] == "assistant" and m.get("tool_calls"):
                out.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["arguments"]),
                                },
                            }
                            for tc in m["tool_calls"]
                        ],
                    }
                )
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    @staticmethod
    def _to_openai_tools(tools: Optional[list[ToolDef]]) -> Optional[list[dict]]:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: Optional[list[ToolDef]] = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.config.model_id,
            "max_tokens": self.config.max_tokens,
            "messages": self._to_openai_messages(system, messages),
        }
        oa_tools = self._to_openai_tools(tools)
        if oa_tools:
            kwargs["tools"] = oa_tools

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            import json

            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                )
            )

        return ModelResponse(
            text=(msg.content or "").strip(),
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            usage={
                "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "output_tokens": getattr(resp.usage, "completion_tokens", 0),
            },
        )

"""Reference ModelAdapter: Anthropic Claude via the official SDK.

Uses adaptive thinking and the effort parameter (DESIGN.md §10). Thinking blocks are
preserved verbatim (including signatures) when appended back to history, which the API
requires for multi-turn tool use with thinking enabled.
"""

from __future__ import annotations

from typing import Any

import anthropic

from .base import ModelAdapter, ModelResponse, NeutralMessage, ToolCall


class AnthropicAdapter(ModelAdapter):
    def __init__(self, spec) -> None:
        super().__init__(spec)
        # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile) from the env.
        self.client = anthropic.Anthropic()

    def generate(
        self,
        *,
        system: str,
        messages: list[NeutralMessage],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ModelResponse:
        # The neutral message/tool schema matches the Anthropic Messages shape, so we
        # pass it through directly. Other providers would translate here instead.
        kwargs: dict[str, Any] = {
            "model": self.spec.model_id,
            "max_tokens": max_tokens,
            "system": system,
            "tools": tools,
            "messages": messages,
        }
        # Thinking and effort are model-capability-gated (e.g. Haiku 4.5 supports
        # neither). Only send them when the spec says the model supports them, so we
        # never trigger a 400 (see config.ModelSpec).
        if self.spec.thinking_mode == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
        if self.spec.supports_effort:
            kwargs["output_config"] = {"effort": self.spec.effort}

        resp = self.client.messages.create(**kwargs)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        assistant_blocks: list[dict[str, Any]] = []

        for block in resp.content:
            btype = block.type
            if btype == "text":
                text_parts.append(block.text)
                assistant_blocks.append({"type": "text", "text": block.text})
            elif btype == "thinking":
                thinking_parts.append(block.thinking)
                # Preserve signature verbatim — required for round-tripping.
                assistant_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": block.signature,
                    }
                )
            elif btype == "redacted_thinking":
                assistant_blocks.append(
                    {"type": "redacted_thinking", "data": block.data}
                )
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        usage = {}
        if resp.usage is not None:
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }

        return ModelResponse(
            stop_reason=resp.stop_reason or "end_turn",
            text="\n".join(text_parts).strip(),
            thinking="\n".join(thinking_parts).strip(),
            tool_calls=tool_calls,
            assistant_blocks=assistant_blocks,
            usage=usage,
            raw=resp,
        )

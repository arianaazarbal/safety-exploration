"""Anthropic adapter — the fully-implemented subject path.

Uses the official `anthropic` SDK. Adaptive thinking and the effort parameter are
applied only for the models configured to support them (see config.ModelConfig).
"""

from __future__ import annotations

import anthropic

from ..config import ModelConfig
from .base import ModelAdapter, ModelResponse, ToolCall


class AnthropicAdapter(ModelAdapter):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile) from the env.
        self.client = anthropic.Anthropic()

    def generate(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        force_tool: str | None = None,
    ) -> ModelResponse:
        kwargs: dict = {
            "model": self.model_id,
            "max_tokens": self.cfg.max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if force_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": force_tool}

        # Adaptive thinking + effort, only where supported. `output_config` carries
        # effort; `thinking` carries the adaptive flag. Both are omitted for models
        # (e.g. Haiku 4.5) configured without them.
        #
        # When we force a specific tool (final-decision capture), skip thinking: forced
        # tool_choice and thinking don't reliably combine, and the capture turn needs no
        # reasoning anyway. Effort is harmless to keep.
        if self.cfg.thinking == "adaptive" and not force_tool:
            kwargs["thinking"] = {"type": "adaptive"}
        if self.cfg.effort:
            kwargs["output_config"] = {"effort": self.cfg.effort}

        resp = self.client.messages.create(**kwargs)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        # Echo the full native content back so thinking-block signatures and tool_use
        # blocks survive the multi-turn round-trip untouched.
        assistant_content: list[dict] = []

        for block in resp.content:
            assistant_content.append(block.model_dump(exclude_none=True))
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))

        return ModelResponse(
            text="\n".join(p for p in text_parts if p),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason,
            assistant_content=assistant_content,
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
            thinking="\n".join(p for p in thinking_parts if p) or None,
            raw=resp,
        )

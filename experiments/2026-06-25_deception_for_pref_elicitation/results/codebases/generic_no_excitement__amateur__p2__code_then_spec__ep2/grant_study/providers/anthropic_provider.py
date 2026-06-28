"""Anthropic provider.

Uses the official ``anthropic`` SDK with:
  - adaptive thinking (the supported mode on Opus 4.8 / 4.7),
  - structured output via ``output_config.format`` (json_schema),
  - streaming, so large ``max_tokens`` never trips the SDK's non-streaming timeout
    guard.
"""

from __future__ import annotations

from typing import Any

from .base import ModelResponse, Provider


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        import anthropic  # imported lazily so the SDK is only needed if used

        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            # Adaptive thinking is the recommended/only on-mode for Opus 4.8/4.7.
            "thinking": {"type": "adaptive"},
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if json_schema is not None:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": json_schema}
            }

        # Stream and collect the final message. Streaming keeps us safely under the
        # SDK's non-streaming timeout guard for large max_tokens.
        with self._client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                # May be empty unless display="summarized"; captured when present.
                thinking_parts.append(getattr(block, "thinking", "") or "")

        text = "".join(text_parts)
        parsed = self._safe_json_loads(text) if json_schema is not None else None

        return ModelResponse(
            provider=self.name,
            model=self.model,
            text=text,
            parsed=parsed,
            thinking="".join(thinking_parts) or None,
            usage={
                "input_tokens": getattr(message.usage, "input_tokens", None),
                "output_tokens": getattr(message.usage, "output_tokens", None),
            },
            raw=message,
        )

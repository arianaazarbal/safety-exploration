"""Anthropic (Claude) adapter using the Messages API."""

from __future__ import annotations

from typing import Any

from .base import AdapterError, Message


class AnthropicAdapter:
    def __init__(
        self,
        model: str,
        api_key: str | None,
        params: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise AdapterError("anthropic package not installed") from e
        if not api_key:
            raise AdapterError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.id = id or model
        self._default_params = params or {}

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        call_params = {**self._default_params, **(params or {})}
        call_params.setdefault("max_tokens", 4000)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            **call_params,
        }
        if system:
            kwargs["system"] = system
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - surface provider errors uniformly
            raise AdapterError(f"anthropic call failed: {e}") from e
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

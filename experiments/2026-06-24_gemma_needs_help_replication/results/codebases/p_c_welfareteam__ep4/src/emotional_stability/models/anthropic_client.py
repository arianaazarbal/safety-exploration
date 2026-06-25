"""Anthropic Claude client for the judge, auditor, onset labeller and paraphraser.

The paper pins exact Claude snapshots (claude-sonnet-4-20250514 judge/auditor,
claude-opus-4-20250514 Petri judge). We call them directly through the Anthropic
SDK rather than via OpenRouter, because those dated snapshot ids are
Anthropic-native and we want deterministic routing to the exact model.
"""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from emotional_stability.config import Settings
from emotional_stability.records import Message


class AnthropicClient:
    """Thin wrapper around the Anthropic Messages API with retry/backoff."""

    def __init__(self, model: str, settings: Settings | None = None):
        self.model = model
        self.settings = (settings or Settings.load()).require("anthropic_api_key")
        # Imported lazily so the package imports without the SDK installed.
        import anthropic

        self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Return the text of a single completion.

        Judges/labellers default to temperature 0 for reproducibility; the Petri
        auditor overrides this to keep its probing varied.
        """
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system is not None:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")

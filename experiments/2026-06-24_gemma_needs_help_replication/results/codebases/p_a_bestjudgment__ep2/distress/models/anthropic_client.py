"""Claude client (Anthropic SDK) used for the judge, Petri auditor/judge, and
onset/paraphrase labelling.

Kept separate from the OpenRouter client because Anthropic is a distinct
provider with its own SDK. The paper sets thinking to false for evaluated
models; the judge/auditor are Claude calls — we leave thinking off (omit the
param) for the older sonnet-4/opus-4 IDs the paper pins, which keeps scoring
deterministic-ish and avoids reasoning-token overhead.
"""

from __future__ import annotations

from ..utils.io import retry


class AnthropicChat:
    def __init__(self, model: str, *, max_retries: int = 4, api_key: str | None = None):
        import anthropic  # imported lazily so the package imports without the dep

        self.model = model
        self.max_retries = max_retries
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(
        self,
        *,
        system: str | None,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> str:
        """Single completion; returns concatenated text blocks."""

        def _call() -> str:
            kwargs: dict = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            if system is not None:
                kwargs["system"] = system
            resp = self._client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if b.type == "text")

        return retry(_call, max_retries=self.max_retries)

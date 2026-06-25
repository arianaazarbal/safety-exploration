"""Anthropic (Claude) judge backend.

Used as the primary frustration scorer (paper: Claude-Sonnet-4) and as the Petri
auditor/judge. Built on the official ``anthropic`` SDK.

Model id is configurable (config/models.yaml). The paper's judge predates the
adaptive-thinking-only models, so we keep a plain Messages call with a low,
deterministic-leaning temperature; if a configured model rejects ``temperature``
(e.g. an Opus 4.7+ id), we transparently retry without it.
"""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import JudgeSpec
from .base import Judge


class AnthropicJudge(Judge):
    def __init__(self, spec: JudgeSpec, temperature: float = 0.0):
        super().__init__(spec)
        self.temperature = temperature

    @property
    def _client(self):
        import anthropic

        # Resolves ANTHROPIC_API_KEY from the environment.
        return anthropic.Anthropic()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        import anthropic

        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens or self.spec.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        try:
            resp = self._client.messages.create(temperature=self.temperature, **kwargs)
        except anthropic.BadRequestError:
            # Newer models (Opus 4.7+/Fable) reject `temperature`; retry without.
            resp = self._client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if b.type == "text").strip()

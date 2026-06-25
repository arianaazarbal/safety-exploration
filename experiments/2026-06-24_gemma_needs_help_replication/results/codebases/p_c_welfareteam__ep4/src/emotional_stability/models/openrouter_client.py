"""Generic OpenRouter text client.

Used for the secondary judge in the judge-agreement check (GPT-5-mini re-scoring,
Section 2.1). Kept separate from the Gemini *target* backend because this is a
utility judge, not a model under evaluation.
"""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from emotional_stability.config import Settings
from emotional_stability.records import Message


class OpenRouterClient:
    def __init__(self, model: str, settings: Settings | None = None):
        self.model = model
        self.settings = (settings or Settings.load()).require("openrouter_api_key")
        from openai import OpenAI

        self._client = OpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
        )

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
        msgs = []
        if system is not None:
            msgs.append({"role": "system", "content": system})
        msgs.extend({"role": m.role, "content": m.content} for m in messages)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

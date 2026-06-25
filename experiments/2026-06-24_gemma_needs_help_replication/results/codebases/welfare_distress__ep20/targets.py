"""Target-model client.

Gemma and Gemini are both served through OpenRouter's OpenAI-compatible Chat
Completions API (the paper serves Gemini via OpenRouter and Gemma via local HF;
we unify on OpenRouter so both families go through one code path -- see DESIGN.md
for the rationale and for how to swap in a local HF/vLLM backend instead).

All calls use temperature 1 (paper default) and request that provider-side
"thinking"/reasoning be disabled where supported.
"""

from __future__ import annotations

import os

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class TargetClient:
    """Thin async wrapper around an OpenAI-compatible endpoint (OpenRouter)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        disable_thinking: bool = True,
    ) -> None:
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Target models (Gemma/Gemini) are "
                "served via OpenRouter. Export the key, or adapt TargetClient to a "
                "local HF/vLLM backend (see DESIGN.md)."
            )
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking

    def _extra_body(self) -> dict:
        # OpenRouter exposes a unified `reasoning` control. Disabling it best-effort
        # mirrors the paper's "thinking=false"; Gemini-2.5-Pro may still produce
        # hidden reasoning regardless (noted in the paper and in DESIGN.md).
        if self.disable_thinking:
            return {"reasoning": {"enabled": False}}
        return {}

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def complete(self, model: str, messages: list[dict]) -> str:
        """Return the assistant text for one chat completion.

        `messages` is a standard OpenAI-style list of {role, content} dicts.
        """
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=self._extra_body(),
        )
        content = resp.choices[0].message.content
        return content or ""

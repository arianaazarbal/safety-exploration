"""OpenRouter inference backend for the closed-source Gemini targets
(google/gemini-2.5-flash, google/gemini-2.5-pro), matching the paper's setup
("For API-based models via OpenRouter ...").

The paper sets thinking to false via the API; we request that here. Note the
paper's own caveat that Gemini-2.5-Pro may still produce hidden reasoning that
this flag does not fully suppress.
"""

from __future__ import annotations

import os
import time

from ..schemas import Message


class OpenRouterBackend:
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        disable_thinking: bool = True,
        max_retries: int = 5,
    ):
        from openai import OpenAI

        self.name = name
        self.model_id = model_id
        self.disable_thinking = disable_thinking
        self.max_retries = max_retries
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=base_url,
        )

    def chat(self, messages, *, temperature: float = 1.0, max_new_tokens: int = 1024) -> str:
        kwargs = dict(
            model=self.model_id,
            messages=[m.to_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        if self.disable_thinking:
            # OpenRouter unified reasoning control: turn reasoning off.
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # network / rate-limit / transient
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"{self.name}: OpenRouter request failed after retries: {last_err}")

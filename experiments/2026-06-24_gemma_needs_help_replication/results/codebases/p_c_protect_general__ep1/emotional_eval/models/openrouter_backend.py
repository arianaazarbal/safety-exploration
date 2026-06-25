"""API inference for Gemini via OpenRouter (matching Appendix B.1).

The paper accesses Gemini through OpenRouter with thinking disabled. We use the
OpenAI-compatible client that OpenRouter exposes. ``continue_prefill`` is *not*
generally supported by hosted Gemini (no assistant-prefill continuation), so it
raises -- the Section 3 prefill study only applies to open-weight models, and
the paper explicitly notes Gemini base models cannot be studied (Limitations).
"""

from __future__ import annotations

import time

from .base import GenerationConfig, Message


class OpenRouterBackend:
    def __init__(
        self,
        name: str,
        api_id: str,
        *,
        config: GenerationConfig,
        api_key: str | None,
        base_url: str,
        extra_body: dict | None = None,
        max_retries: int = 5,
    ):
        self.name = name
        self.api_id = api_id
        self.config = config
        self.extra_body = extra_body or {}
        self.max_retries = max_retries
        self._client = None
        self._api_key = api_key
        self._base_url = base_url

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def chat(self, messages: list[Message], system: str | None = None) -> str:
        client = self._ensure_client()
        payload = ([{"role": "system", "content": system}] if system else []) + messages
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.api_id,
                    messages=payload,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_new_tokens,
                    extra_body=self.extra_body or None,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - retry transient API errors
                last_err = e
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"{self.name}: chat failed after retries") from last_err

    def continue_prefill(self, messages, prefill, system=None) -> str:  # noqa: D401
        raise NotImplementedError(
            "Prefill continuation is not supported for hosted Gemini; the "
            "Section 3 base-vs-instruct study is open-weight only (see DESIGN.md)."
        )

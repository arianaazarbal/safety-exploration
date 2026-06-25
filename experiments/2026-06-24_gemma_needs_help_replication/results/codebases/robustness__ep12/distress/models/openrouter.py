"""Gemini backend via OpenRouter (OpenAI-compatible API).

Matches the paper's setup (App. B.1): Gemini accessed through OpenRouter with
`thinking` disabled. Gemini is closed, so prefilled continuations are not
supported.

Requires OPENROUTER_API_KEY in the environment.
"""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatMessage, GenerationResult

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    def __init__(self, model_entry: dict, api_key: str | None = None):
        self.entry = model_entry
        self.api_id = model_entry["api_id"]
        self.name = self.api_id
        self.thinking = model_entry.get("thinking", False)
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI

        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set; required for Gemini access."
            )
        self._client = OpenAI(base_url=OPENROUTER_BASE_URL,
                              api_key=self._api_key)
        return self._client

    def _extra_body(self) -> dict:
        # Disable hidden reasoning where the provider honours it. Gemini 2.5
        # Pro may still emit hidden reasoning (paper caveat) -- we set the flag
        # regardless for parity with the paper.
        if self.thinking:
            return {}
        return {"reasoning": {"exclude": True}}

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def chat(self, messages, temperature=1.0, max_new_tokens=2048, seed=None):
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.api_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            seed=seed,
            extra_body=self._extra_body(),
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    def continue_prefill(self, *args, **kwargs):
        raise NotImplementedError(
            "Gemini is closed; prefilled continuations (Section 3 / recovery / "
            "probing) are Gemma-only."
        )
